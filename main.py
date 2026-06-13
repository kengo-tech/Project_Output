from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import get_historical_klines
from src.features import make_features


PROJECT_ROOT = Path(__file__).resolve().parent


def add_regime_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add readable trend and volatility regime labels.
    """
    regime_df = df.copy()

    regime_df = (
        regime_df
        .assign(
            trend_regime=lambda x: np.select(
                [
                    x["trend_up_6h"] & x["above_ma_6h"],
                    (~x["trend_up_6h"]) & (~x["above_ma_6h"]),
                ],
                [
                    "trend_up",
                    "trend_down",
                ],
                default="mixed_trend",
            ),
            volatility_regime=lambda x: np.select(
                [
                    x["extreme_vol_flag"],
                    x["high_vol_flag"],
                    x["low_vol_flag"],
                ],
                [
                    "extreme_vol",
                    "high_vol",
                    "low_vol",
                ],
                default="normal_vol",
            ),
        )
    )

    return regime_df


def add_candidate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add baseline and refined short-squeeze candidate signals.
    """
    signal_df = df.copy()

    baseline = (
        signal_df["breakout_forced_cover_signal"]
        & (signal_df["trend_regime"] == "trend_down")
        & (signal_df["volatility_regime"] == "high_vol")
    )

    signal_df = (
        signal_df
        .assign(
            downtrend_highvol_forced_cover_signal=baseline,
            dhv_buy60_signal=lambda x: (
                baseline
                & (x["buy_aggression_ratio_15m"] >= 0.60)
            ),
            dhv_cvd_positive_signal=lambda x: (
                baseline
                & (x["cvd_slope_15m"] > 0)
            ),
            dhv_buy60_cvdpos_signal=lambda x: (
                baseline
                & (x["buy_aggression_ratio_15m"] >= 0.60)
                & (x["cvd_slope_15m"] > 0)
            ),
            dhv_score4_signal=lambda x: (
                baseline
                & (x["forced_cover_score"] >= 4)
            ),
            dhv_score4_buy60_signal=lambda x: (
                baseline
                & (x["forced_cover_score"] >= 4)
                & (x["buy_aggression_ratio_15m"] >= 0.60)
            ),
            dhv_score4_buy60_cvdpos_signal=lambda x: (
                baseline
                & (x["forced_cover_score"] >= 4)
                & (x["buy_aggression_ratio_15m"] >= 0.60)
                & (x["cvd_slope_15m"] > 0)
            ),
        )
    )

    return signal_df


def add_sample_split(
    df: pd.DataFrame,
    split_time: str = "2026-05-08 00:00:00",
) -> pd.DataFrame:
    """
    Add in-sample and out-of-sample labels.
    """
    split_ts = pd.to_datetime(split_time, utc=True)

    split_df = (
        df
        .copy()
        .assign(
            sample_split=lambda x: np.where(
                x["open_time"] < split_ts,
                "in_sample",
                "out_of_sample",
            )
        )
    )

    return split_df


def _safe_float(value: float, default: float = 0.0) -> float:
    """
    Return a finite float, otherwise a default value.
    """
    if value is None or np.isnan(value):
        return default

    return float(value)


def estimate_round_trip_cost(
    df: pd.DataFrame,
    entry_idx: int,
    exit_idx: int,
    fee_rate: float,
    slippage_rate: float,
    cost_model: str = "fixed",
) -> float:
    """
    Estimate round-trip cost under fixed or stressed execution assumptions.

    The dataset does not include bid-ask spread, so dynamic models use
    volatility and volume-shock proxies for spread / slippage stress.
    """
    allowed_cost_models = [
        "fixed",
        "dynamic_slippage",
        "spread_stress",
    ]

    if cost_model not in allowed_cost_models:
        raise ValueError(
            f"cost_model must be one of {allowed_cost_models}. "
            f"Received: {cost_model}"
        )

    if cost_model == "fixed":
        return 2 * fee_rate + 2 * slippage_rate

    entry_vol = _safe_float(df.iloc[entry_idx].get("vol_percentile"), 0.50)
    exit_vol = _safe_float(df.iloc[exit_idx].get("vol_percentile"), 0.50)
    entry_volume_shock = _safe_float(
        df.iloc[entry_idx].get("volume_shock_15m"),
        1.0,
    )
    exit_volume_shock = _safe_float(
        df.iloc[exit_idx].get("volume_shock_15m"),
        1.0,
    )

    stress_score = max(entry_vol, exit_vol)
    thin_liquidity_proxy = min(entry_volume_shock, exit_volume_shock)

    if cost_model == "dynamic_slippage":
        slippage_multiplier = 1.0

        if stress_score >= 0.95:
            slippage_multiplier += 1.50
        elif stress_score >= 0.70:
            slippage_multiplier += 0.75
        elif stress_score >= 0.50:
            slippage_multiplier += 0.25

        if thin_liquidity_proxy < 0.80:
            slippage_multiplier += 0.50

    else:
        slippage_multiplier = 2.0

        if stress_score >= 0.70:
            slippage_multiplier += 1.0

        if thin_liquidity_proxy < 0.80:
            slippage_multiplier += 1.0

    return 2 * fee_rate + 2 * slippage_rate * slippage_multiplier


def run_fast_fixed_horizon_backtest(
    df: pd.DataFrame,
    signal_col: str,
    holding_bars: int,
    fee_rate: float,
    slippage_rate: float,
    cost_model: str = "fixed",
) -> pd.DataFrame:
    """
    Fast long-only fixed-horizon backtest.

    Entry:
    - Enter at next bar open after signal.

    Exit:
    - Exit after fixed_holding_bars.

    Overlap:
    - Skip new signal while previous trade is still open.

    Cost:
    - Round-trip cost can be fixed or dynamically stressed.
    """
    if df.empty:
        return pd.DataFrame()

    open_array = df["open"].to_numpy(dtype=float)
    close_array = df["close"].to_numpy(dtype=float)
    time_array = df["open_time"].to_numpy()
    signal_array = df[signal_col].fillna(False).to_numpy(dtype=bool)

    signal_indices = np.flatnonzero(signal_array)

    if len(signal_indices) == 0:
        return pd.DataFrame()

    round_trip_cost = 2 * fee_rate + 2 * slippage_rate

    trades = []
    next_available_idx = 0
    last_idx = len(df) - 1

    for signal_idx in signal_indices:
        if signal_idx < next_available_idx:
            continue

        entry_idx = signal_idx + 1
        exit_idx = entry_idx + holding_bars

        if entry_idx >= last_idx:
            continue

        if exit_idx > last_idx:
            continue

        entry_price = open_array[entry_idx]
        exit_price = close_array[exit_idx]

        if entry_price <= 0 or np.isnan(entry_price) or np.isnan(exit_price):
            continue

        gross_return = (exit_price / entry_price) - 1
        round_trip_cost = estimate_round_trip_cost(
            df=df,
            entry_idx=entry_idx,
            exit_idx=exit_idx,
            fee_rate=fee_rate,
            slippage_rate=slippage_rate,
            cost_model=cost_model,
        )
        net_return = gross_return - round_trip_cost

        trades.append(
            {
                "signal_name": signal_col,
                "signal_idx": signal_idx,
                "entry_idx": entry_idx,
                "exit_idx": exit_idx,
                "signal_time": time_array[signal_idx],
                "entry_time": time_array[entry_idx],
                "exit_time": time_array[exit_idx],
                "entry_price": entry_price,
                "exit_price": exit_price,
                "gross_return": gross_return,
                "net_return": net_return,
                "round_trip_cost": round_trip_cost,
                "cost_model": cost_model,
                "holding_bars": holding_bars,
                "exit_reason": f"fixed_{holding_bars}b_exit",
                "win": net_return > 0,
            }
        )

        next_available_idx = exit_idx + 1

    trade_df = pd.DataFrame(trades)

    return trade_df


def run_exit_rule_backtest(
    df: pd.DataFrame,
    signal_col: str,
    exit_rule: str,
    max_holding_bars: int,
    fee_rate: float,
    slippage_rate: float,
    cost_model: str = "fixed",
    hard_stop_return: float = -0.004,
    half_take_profit_return: float = 0.003,
    min_bars_before_thesis_stop: int = 2,
    daily_loss_limit: float | None = None,
    max_consecutive_losses: int | None = None,
    cooldown_bars: int = 0,
) -> pd.DataFrame:
    """
    Backtest fixed-entry trades with simple exit-rule variants.

    Entry is fixed at the next bar open after signal. Returns are measured on
    full-position capital, with one round-trip cost applied to the full trade.
    """
    if df.empty:
        return pd.DataFrame()

    allowed_exit_rules = [
        "fixed_6b",
        "hard_stop_6b",
        "half_tp_thesis_stop_6b",
    ]

    if exit_rule not in allowed_exit_rules:
        raise ValueError(
            f"exit_rule must be one of {allowed_exit_rules}. "
            f"Received: {exit_rule}"
        )

    open_array = df["open"].to_numpy(dtype=float)
    high_array = df["high"].to_numpy(dtype=float)
    low_array = df["low"].to_numpy(dtype=float)
    close_array = df["close"].to_numpy(dtype=float)
    rolling_high_array = df["rolling_high_12"].to_numpy(dtype=float)
    cvd_slope_array = df["cvd_slope_15m"].to_numpy(dtype=float)
    time_array = df["open_time"].to_numpy()
    signal_array = df[signal_col].fillna(False).to_numpy(dtype=bool)

    trades = []
    next_available_idx = 0
    last_idx = len(df) - 1
    daily_net_return = {}
    consecutive_losses = 0
    cooldown_until_idx = 0

    for signal_idx in np.flatnonzero(signal_array):
        if signal_idx < next_available_idx:
            continue

        if signal_idx < cooldown_until_idx:
            continue

        entry_idx = signal_idx + 1
        max_exit_idx = entry_idx + max_holding_bars

        if entry_idx >= last_idx or max_exit_idx > last_idx:
            continue

        entry_day = pd.Timestamp(time_array[entry_idx]).date()

        if (
            daily_loss_limit is not None
            and daily_net_return.get(entry_day, 0.0) <= daily_loss_limit
        ):
            continue

        entry_price = open_array[entry_idx]

        if entry_price <= 0 or np.isnan(entry_price):
            continue

        exit_idx = max_exit_idx
        exit_reason = f"fixed_{max_holding_bars}b_exit"
        gross_return = close_array[max_exit_idx] / entry_price - 1

        if exit_rule == "hard_stop_6b":
            stop_price = entry_price * (1 + hard_stop_return)

            for check_idx in range(entry_idx, max_exit_idx + 1):
                if low_array[check_idx] <= stop_price:
                    exit_idx = check_idx
                    exit_reason = "hard_stop"
                    gross_return = hard_stop_return
                    break

        elif exit_rule == "half_tp_thesis_stop_6b":
            first_leg_return = None
            second_leg_return = None
            first_exit_idx = None
            second_exit_idx = None
            first_exit_reason = None
            second_exit_reason = None
            tp_price = entry_price * (1 + half_take_profit_return)

            for check_idx in range(entry_idx, max_exit_idx + 1):
                if first_leg_return is None and high_array[check_idx] >= tp_price:
                    first_leg_return = half_take_profit_return
                    first_exit_idx = check_idx
                    first_exit_reason = "half_take_profit"

                bars_held = check_idx - entry_idx
                thesis_stop = (
                    bars_held >= min_bars_before_thesis_stop
                    and close_array[check_idx] < rolling_high_array[check_idx]
                    and cvd_slope_array[check_idx] < 0
                )

                if thesis_stop:
                    second_leg_return = close_array[check_idx] / entry_price - 1
                    second_exit_idx = check_idx
                    second_exit_reason = "thesis_stop"
                    break

            if second_leg_return is None:
                second_leg_return = close_array[max_exit_idx] / entry_price - 1
                second_exit_idx = max_exit_idx
                second_exit_reason = f"fixed_{max_holding_bars}b_exit"

            if first_leg_return is None:
                first_leg_return = second_leg_return
                first_exit_idx = second_exit_idx
                first_exit_reason = second_exit_reason

            gross_return = 0.5 * first_leg_return + 0.5 * second_leg_return
            exit_idx = max(first_exit_idx, second_exit_idx)
            exit_reason = f"{first_exit_reason}+{second_exit_reason}"

        round_trip_cost = estimate_round_trip_cost(
            df=df,
            entry_idx=entry_idx,
            exit_idx=exit_idx,
            fee_rate=fee_rate,
            slippage_rate=slippage_rate,
            cost_model=cost_model,
        )
        net_return = gross_return - round_trip_cost
        daily_net_return[entry_day] = (
            daily_net_return.get(entry_day, 0.0)
            + net_return
        )

        if net_return < 0:
            consecutive_losses += 1
        else:
            consecutive_losses = 0

        cooldown_triggered = False

        if (
            max_consecutive_losses is not None
            and consecutive_losses >= max_consecutive_losses
            and cooldown_bars > 0
        ):
            cooldown_until_idx = exit_idx + cooldown_bars
            consecutive_losses = 0
            cooldown_triggered = True

        trades.append(
            {
                "signal_name": signal_col,
                "exit_rule": exit_rule,
                "signal_idx": signal_idx,
                "entry_idx": entry_idx,
                "exit_idx": exit_idx,
                "signal_time": time_array[signal_idx],
                "entry_time": time_array[entry_idx],
                "exit_time": time_array[exit_idx],
                "entry_price": entry_price,
                "exit_price": close_array[exit_idx],
                "gross_return": gross_return,
                "net_return": net_return,
                "round_trip_cost": round_trip_cost,
                "cost_model": cost_model,
                "daily_net_return_after_trade": daily_net_return[entry_day],
                "consecutive_losses_after_trade": consecutive_losses,
                "cooldown_triggered": cooldown_triggered,
                "holding_bars": exit_idx - entry_idx,
                "exit_reason": exit_reason,
                "win": net_return > 0,
            }
        )

        next_available_idx = exit_idx + 1

    return pd.DataFrame(trades)


def calculate_profit_factor(trade_df: pd.DataFrame) -> float:
    """
    Calculate profit factor from trade-level net returns.
    """
    if trade_df.empty:
        return np.nan

    gross_profit = trade_df.loc[
        trade_df["net_return"] > 0,
        "net_return",
    ].sum()

    gross_loss = trade_df.loc[
        trade_df["net_return"] < 0,
        "net_return",
    ].sum()

    if gross_loss == 0:
        return np.nan

    return gross_profit / abs(gross_loss)


def calculate_max_drawdown_from_returns(returns: pd.Series) -> float:
    """
    Calculate max drawdown from trade-level returns.
    """
    if returns.empty:
        return np.nan

    equity_curve = (1 + returns).cumprod()
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1

    return drawdown.min()


def summarize_trade_distribution(
    trade_df: pd.DataFrame,
    sample_label: str,
    signal_name: str,
    holding_bars: int,
    cost_label: str,
) -> dict:
    """
    Summarize trade return distribution for one configuration.
    """
    if trade_df.empty:
        return {
            "sample_split": sample_label,
            "signal_name": signal_name,
            "holding_bars": holding_bars,
            "holding_minutes": holding_bars * 5,
            "cost_label": cost_label,
            "trades": 0,
            "win_rate": np.nan,
            "avg_net_return": np.nan,
            "median_net_return": np.nan,
            "std_net_return": np.nan,
            "best_trade": np.nan,
            "worst_trade": np.nan,
            "profit_factor": np.nan,
            "max_drawdown": np.nan,
            "total_return": np.nan,
        }

    returns = trade_df["net_return"]

    return {
        "sample_split": sample_label,
        "signal_name": signal_name,
        "holding_bars": holding_bars,
        "holding_minutes": holding_bars * 5,
        "cost_label": cost_label,
        "trades": len(trade_df),
        "win_rate": trade_df["win"].mean(),
        "avg_net_return": returns.mean(),
        "median_net_return": returns.median(),
        "std_net_return": returns.std(),
        "best_trade": returns.max(),
        "worst_trade": returns.min(),
        "profit_factor": calculate_profit_factor(trade_df),
        "max_drawdown": calculate_max_drawdown_from_returns(returns),
        "total_return": (1 + returns).prod() - 1,
    }


def add_cumulative_metrics(trade_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add cumulative return and drawdown to trade-level results.
    """
    if trade_df.empty:
        return trade_df.copy()

    group_cols = [
        "sample_split",
        "signal_name",
        "holding_bars_config",
        "cost_label",
    ]

    cumulative_df = (
        trade_df
        .copy()
        .sort_values(group_cols + ["entry_time"])
        .reset_index(drop=True)
    )

    cumulative_df["equity_curve"] = (
        cumulative_df
        .groupby(group_cols, dropna=False)["net_return"]
        .transform(lambda x: (1 + x).cumprod())
    )

    cumulative_df["running_max"] = (
        cumulative_df
        .groupby(group_cols, dropna=False)["equity_curve"]
        .transform("cummax")
    )

    cumulative_df["drawdown"] = (
        cumulative_df["equity_curve"]
        / cumulative_df["running_max"]
        - 1
    )

    return cumulative_df


def main():
    pd.set_option("display.max_columns", 160)
    pd.set_option("display.width", 280)

    main_entry_signal = "downtrend_exhaustion_reclaim_signal"
    main_holding_bars = 6
    main_exit_rule = "fixed_6b"
    risk_control_exit_rule = "hard_stop_6b"

    signal_candidates = [
        "downtrend_highvol_forced_cover_signal",
        "dhv_buy60_signal",
        "dhv_cvd_positive_signal",
        "dhv_buy60_cvdpos_signal",
        "dhv_score4_signal",
        "dhv_score4_buy60_signal",
        "dhv_score4_buy60_cvdpos_signal",
        "downtrend_exhaustion_reclaim_signal",
        "downtrend_exhaustion_reclaim_ret30_signal",
        "downtrend_exhaustion_reclaim_strength_signal",
        "downtrend_exhaustion_reclaim_wick_signal",
        "seller_advantage_break_signal",
        "seller_advantage_break_recent3_signal",
    ]

    holding_periods = [3, 6, 12]

    cost_configs = [
        {
            "cost_label": "no_cost",
            "fee_rate": 0.0,
            "slippage_rate": 0.0,
            "cost_model": "fixed",
        },
        {
            "cost_label": "base_cost",
            "fee_rate": 0.0004,
            "slippage_rate": 0.0002,
            "cost_model": "fixed",
        },
        {
            "cost_label": "dynamic_slippage",
            "fee_rate": 0.0004,
            "slippage_rate": 0.0002,
            "cost_model": "dynamic_slippage",
        },
        {
            "cost_label": "spread_stress",
            "fee_rate": 0.0004,
            "slippage_rate": 0.0002,
            "cost_model": "spread_stress",
        },
    ]

    ### A1: Download about 90 days of BTCUSDT 5-minute klines
    raw_df = get_historical_klines(
        symbol="BTCUSDT",
        interval="5m",
        start_time="2026-03-10",
        end_time="2026-06-07 23:59:59",
        limit=1000,
        sleep_seconds=0.0,
        verbose=False,
    )

    print("\nA1: Raw data shape")
    print(raw_df.shape)

    if raw_df.empty:
        print("No data was downloaded.")
        return

    print("\nA2: Raw data time range")
    print("Start:", raw_df["open_time"].min())
    print("End:  ", raw_df["open_time"].max())

    ### A3: Create features, regimes, candidate signals, and sample split
    feature_df = make_features(raw_df)
    feature_df = add_regime_labels(feature_df)
    feature_df = add_candidate_signals(feature_df)
    feature_df = add_sample_split(
        feature_df,
        split_time="2026-05-08 00:00:00",
    )

    print("\nA3: Feature data shape")
    print(feature_df.shape)

    ### A4: IS/OOS split summary
    sample_split_df = (
        feature_df
        .groupby("sample_split")
        .agg(
            rows=("open_time", "count"),
            start_time=("open_time", "min"),
            end_time=("open_time", "max"),
        )
        .reset_index()
    )

    print("\nA4: IS/OOS split summary")
    print(sample_split_df)

    ### A5: Signal count comparison
    signal_count_rows = []

    for signal_name in signal_candidates:
        signal_count_part = (
            feature_df
            .groupby("sample_split")
            .agg(
                true_count=(signal_name, "sum"),
                sample_size=(signal_name, "count"),
            )
            .reset_index()
            .assign(
                signal_name=signal_name,
                signal_rate=lambda x: x["true_count"] / x["sample_size"],
            )
            [
                [
                    "sample_split",
                    "signal_name",
                    "true_count",
                    "sample_size",
                    "signal_rate",
                ]
            ]
        )

        signal_count_rows.append(signal_count_part)

    signal_count_df = (
        pd.concat(signal_count_rows, ignore_index=True)
        .sort_values(
            ["sample_split", "true_count"],
            ascending=[True, False],
        )
        .reset_index(drop=True)
    )

    print("\nA5: Signal count comparison")
    print(signal_count_df)

    ### A6: Lightweight final candidate signal rows
    target_signal = main_entry_signal

    candidate_cols = [
        "open_time",
        "close",
        "sample_split",
        "trend_regime",
        "volatility_regime",
        "downtrend_ma_6h",
        "return_30m",
        "no_new_low_12",
        "reclaim_strength_12",
        "lower_wick_ratio",
        "seller_exhaustion_setup_signal",
        "sell_pressure_60m",
        "sell_pressure_60m_zscore",
        "absorption_flag",
        "cvd_slope_60m",
        "cvd_flip_15m",
        "cvd_flip_recent_3",
        "reclaim_breakout_12",
        "forced_cover_score",
        "buy_aggression_ratio_15m",
        "cvd_slope_15m",
        "volume_shock_15m",
        "vol_percentile",
        target_signal,
    ]

    candidate_rows_df = (
        feature_df
        .loc[feature_df[target_signal], candidate_cols]
        .copy()
        .reset_index(drop=True)
    )

    print("\nA6: Final candidate signal rows")
    print(candidate_rows_df.head(20))
    print("...")
    print(candidate_rows_df.tail(10))

    ### A7: Fast holding period and cost sensitivity
    all_trade_results = []
    distribution_rows = []

    for signal_name in signal_candidates:
        for sample_label in ["in_sample", "out_of_sample"]:
            sample_df = (
                feature_df
                .loc[feature_df["sample_split"] == sample_label]
                .copy()
                .reset_index(drop=True)
            )

            for holding_bars in holding_periods:
                for cost_config in cost_configs:
                    trade_df = run_fast_fixed_horizon_backtest(
                        df=sample_df,
                        signal_col=signal_name,
                        holding_bars=holding_bars,
                        fee_rate=cost_config["fee_rate"],
                        slippage_rate=cost_config["slippage_rate"],
                        cost_model=cost_config["cost_model"],
                    )

                    if not trade_df.empty:
                        trade_df = (
                            trade_df
                            .copy()
                            .assign(
                                sample_split=sample_label,
                                signal_name=signal_name,
                                holding_bars_config=holding_bars,
                                holding_minutes=holding_bars * 5,
                                cost_label=cost_config["cost_label"],
                                cost_model=cost_config["cost_model"],
                            )
                        )

                        all_trade_results.append(trade_df)

                    distribution_rows.append(
                        summarize_trade_distribution(
                            trade_df=trade_df,
                            sample_label=sample_label,
                            signal_name=signal_name,
                            holding_bars=holding_bars,
                            cost_label=cost_config["cost_label"],
                        )
                    )

    trade_distribution_df = pd.DataFrame(distribution_rows)

    print("\nA7: Trade distribution summary")
    print(trade_distribution_df)

    if all_trade_results:
        combined_trade_df = pd.concat(
            all_trade_results,
            ignore_index=True,
        )
    else:
        combined_trade_df = pd.DataFrame()

    cumulative_trade_df = add_cumulative_metrics(combined_trade_df)

    ### A8: Base cost ranking
    base_cost_df = (
        trade_distribution_df
        .loc[lambda x: x["cost_label"] == "base_cost"]
        .copy()
        .sort_values(
            ["sample_split", "avg_net_return"],
            ascending=[True, False],
        )
        .reset_index(drop=True)
    )

    print("\nA8: Base cost ranking")
    print(
        base_cost_df
        [
            [
                "sample_split",
                "signal_name",
                "holding_bars",
                "holding_minutes",
                "trades",
                "win_rate",
                "avg_net_return",
                "median_net_return",
                "total_return",
                "profit_factor",
                "max_drawdown",
                "best_trade",
                "worst_trade",
            ]
        ]
    )

    ### A9: Out-of-sample base cost ranking
    oos_base_cost_df = (
        base_cost_df
        .loc[lambda x: x["sample_split"] == "out_of_sample"]
        .copy()
        .sort_values(
            ["avg_net_return", "profit_factor", "trades"],
            ascending=[False, False, False],
        )
        .reset_index(drop=True)
    )

    print("\nA9: Out-of-sample base cost ranking")
    print(
        oos_base_cost_df
        [
            [
                "sample_split",
                "signal_name",
                "holding_bars",
                "holding_minutes",
                "trades",
                "win_rate",
                "avg_net_return",
                "median_net_return",
                "total_return",
                "profit_factor",
                "max_drawdown",
                "best_trade",
                "worst_trade",
            ]
        ]
    )

    ### A10: Cost sensitivity pivot by avg_net_return
    cost_pivot_df = (
        trade_distribution_df
        .pivot_table(
            index=[
                "sample_split",
                "signal_name",
                "holding_bars",
                "holding_minutes",
            ],
            columns="cost_label",
            values="avg_net_return",
            aggfunc="first",
        )
        .reset_index()
    )

    print("\nA10: Cost sensitivity pivot by avg_net_return")
    print(cost_pivot_df)

    ### A11: Signal robustness table
    robustness_df = (
        trade_distribution_df
        .loc[lambda x: x["cost_label"] == "base_cost"]
        .copy()
        .assign(
            positive_after_base_cost=lambda x: x["avg_net_return"] > 0,
            enough_trades=lambda x: x["trades"] >= 5,
        )
        .pivot_table(
            index=[
                "signal_name",
                "holding_bars",
                "holding_minutes",
            ],
            columns="sample_split",
            values=[
                "positive_after_base_cost",
                "enough_trades",
            ],
            aggfunc="first",
        )
        .reset_index()
    )

    robustness_df.columns = [
        "_".join([str(part) for part in col if str(part) != ""])
        for col in robustness_df.columns
    ]

    if (
        "positive_after_base_cost_in_sample" in robustness_df.columns
        and "positive_after_base_cost_out_of_sample" in robustness_df.columns
    ):
        robustness_df = (
            robustness_df
            .assign(
                positive_in_both=lambda x: (
                    x["positive_after_base_cost_in_sample"]
                    & x["positive_after_base_cost_out_of_sample"]
                ),
            )
        )

    if (
        "enough_trades_in_sample" in robustness_df.columns
        and "enough_trades_out_of_sample" in robustness_df.columns
    ):
        robustness_df = (
            robustness_df
            .assign(
                enough_trades_in_both=lambda x: (
                    x["enough_trades_in_sample"]
                    & x["enough_trades_out_of_sample"]
                ),
            )
        )

    print("\nA11: Signal robustness table")
    print(robustness_df)

    ### A12: Final recommended refined signals
    recommended_signal_df = (
        base_cost_df
        .merge(
            robustness_df
            [
                [
                    "signal_name",
                    "holding_bars",
                    "holding_minutes",
                    "positive_in_both",
                    "enough_trades_in_both",
                ]
            ],
            on=[
                "signal_name",
                "holding_bars",
                "holding_minutes",
            ],
            how="left",
        )
        .loc[
            lambda x: (
                x["positive_in_both"]
                & x["enough_trades_in_both"]
            )
        ]
        .copy()
        .sort_values(
            ["sample_split", "avg_net_return"],
            ascending=[True, False],
        )
        .reset_index(drop=True)
    )

    print("\nA12: Final recommended refined signals")
    print(recommended_signal_df)

    ### A13: Final candidate curve
    final_signal = main_entry_signal
    final_holding_bars = main_holding_bars

    final_candidate_curve_df = (
        cumulative_trade_df
        .loc[
            (cumulative_trade_df["cost_label"] == "base_cost")
            & (cumulative_trade_df["signal_name"] == final_signal)
            & (cumulative_trade_df["holding_bars_config"] == final_holding_bars)
        ]
        .copy()
        .reset_index(drop=True)
    )

    print("\nA13: Cumulative return and drawdown table for final candidate")
    print("Final signal:", final_signal)
    print("Final holding_bars:", final_holding_bars)
    print("Primary exit rule:", main_exit_rule)
    print("Risk-control exit rule:", risk_control_exit_rule)

    if final_candidate_curve_df.empty:
        print("No final candidate trades.")
    else:
        print(
            final_candidate_curve_df
            [
                [
                    "sample_split",
                    "signal_name",
                    "entry_time",
                    "exit_time",
                    "net_return",
                    "equity_curve",
                    "drawdown",
                    "win",
                    "exit_reason",
                ]
            ]
        )

    ### A14: Exit rule comparison for the main candidate
    exit_rule_signal = main_entry_signal
    exit_rules = [
        main_exit_rule,
        risk_control_exit_rule,
        "half_tp_thesis_stop_6b",
    ]
    exit_rule_cost_configs = [
        cost_config
        for cost_config in cost_configs
        if cost_config["cost_label"] != "no_cost"
    ]
    exit_rule_trade_rows = []
    exit_rule_summary_rows = []

    for sample_label in ["in_sample", "out_of_sample"]:
        sample_df = (
            feature_df
            .loc[feature_df["sample_split"] == sample_label]
            .copy()
            .reset_index(drop=True)
        )

        for exit_rule in exit_rules:
            for cost_config in exit_rule_cost_configs:
                trade_df = run_exit_rule_backtest(
                    df=sample_df,
                    signal_col=exit_rule_signal,
                    exit_rule=exit_rule,
                    max_holding_bars=6,
                    fee_rate=cost_config["fee_rate"],
                    slippage_rate=cost_config["slippage_rate"],
                    cost_model=cost_config["cost_model"],
                )

                if not trade_df.empty:
                    trade_df = (
                        trade_df
                        .copy()
                        .assign(
                            sample_split=sample_label,
                            signal_name=exit_rule_signal,
                            holding_bars_config=6,
                            holding_minutes=30,
                            cost_label=cost_config["cost_label"],
                            cost_model=cost_config["cost_model"],
                        )
                    )

                    exit_rule_trade_rows.append(trade_df)

                summary = summarize_trade_distribution(
                    trade_df=trade_df,
                    sample_label=sample_label,
                    signal_name=exit_rule_signal,
                    holding_bars=6,
                    cost_label=cost_config["cost_label"],
                )
                summary["exit_rule"] = exit_rule
                summary["cost_model"] = cost_config["cost_model"]
                exit_rule_summary_rows.append(summary)

    if exit_rule_trade_rows:
        exit_rule_trade_df = pd.concat(
            exit_rule_trade_rows,
            ignore_index=True,
        )
    else:
        exit_rule_trade_df = pd.DataFrame()

    exit_rule_summary_df = (
        pd.DataFrame(exit_rule_summary_rows)
        .sort_values(
            ["sample_split", "cost_label", "avg_net_return"],
            ascending=[True, True, False],
        )
        .reset_index(drop=True)
    )

    print("\nA14: Exit rule comparison for main candidate")
    print(
        exit_rule_summary_df
        [
            [
                "sample_split",
                "signal_name",
                "exit_rule",
                "cost_label",
                "cost_model",
                "holding_bars",
                "trades",
                "win_rate",
                "avg_net_return",
                "median_net_return",
                "total_return",
                "profit_factor",
                "max_drawdown",
                "best_trade",
                "worst_trade",
            ]
        ]
    )

    ### A15: Risk overlay comparison for the final candidate
    risk_overlay_configs = [
        {
            "risk_overlay": "none",
            "daily_loss_limit": None,
            "max_consecutive_losses": None,
            "cooldown_bars": 0,
        },
        {
            "risk_overlay": "daily_loss_1pct",
            "daily_loss_limit": -0.01,
            "max_consecutive_losses": None,
            "cooldown_bars": 0,
        },
        {
            "risk_overlay": "loss2_cooldown_12b",
            "daily_loss_limit": None,
            "max_consecutive_losses": 2,
            "cooldown_bars": 12,
        },
        {
            "risk_overlay": "daily_loss_1pct_loss2_cooldown_12b",
            "daily_loss_limit": -0.01,
            "max_consecutive_losses": 2,
            "cooldown_bars": 12,
        },
    ]
    risk_overlay_trade_rows = []
    risk_overlay_summary_rows = []

    for sample_label in ["in_sample", "out_of_sample"]:
        sample_df = (
            feature_df
            .loc[feature_df["sample_split"] == sample_label]
            .copy()
            .reset_index(drop=True)
        )

        for risk_config in risk_overlay_configs:
            for cost_config in exit_rule_cost_configs:
                trade_df = run_exit_rule_backtest(
                    df=sample_df,
                    signal_col=main_entry_signal,
                    exit_rule=main_exit_rule,
                    max_holding_bars=main_holding_bars,
                    fee_rate=cost_config["fee_rate"],
                    slippage_rate=cost_config["slippage_rate"],
                    cost_model=cost_config["cost_model"],
                    daily_loss_limit=risk_config["daily_loss_limit"],
                    max_consecutive_losses=risk_config["max_consecutive_losses"],
                    cooldown_bars=risk_config["cooldown_bars"],
                )

                if not trade_df.empty:
                    trade_df = (
                        trade_df
                        .copy()
                        .assign(
                            sample_split=sample_label,
                            signal_name=main_entry_signal,
                            exit_rule=main_exit_rule,
                            holding_bars_config=main_holding_bars,
                            holding_minutes=main_holding_bars * 5,
                            cost_label=cost_config["cost_label"],
                            cost_model=cost_config["cost_model"],
                            risk_overlay=risk_config["risk_overlay"],
                        )
                    )

                    risk_overlay_trade_rows.append(trade_df)

                summary = summarize_trade_distribution(
                    trade_df=trade_df,
                    sample_label=sample_label,
                    signal_name=main_entry_signal,
                    holding_bars=main_holding_bars,
                    cost_label=cost_config["cost_label"],
                )
                summary["exit_rule"] = main_exit_rule
                summary["cost_model"] = cost_config["cost_model"]
                summary["risk_overlay"] = risk_config["risk_overlay"]
                summary["cooldown_triggers"] = (
                    int(trade_df["cooldown_triggered"].sum())
                    if not trade_df.empty
                    else 0
                )
                risk_overlay_summary_rows.append(summary)

    if risk_overlay_trade_rows:
        risk_overlay_trade_df = pd.concat(
            risk_overlay_trade_rows,
            ignore_index=True,
        )
    else:
        risk_overlay_trade_df = pd.DataFrame()

    risk_overlay_summary_df = (
        pd.DataFrame(risk_overlay_summary_rows)
        .sort_values(
            ["sample_split", "cost_label", "avg_net_return"],
            ascending=[True, True, False],
        )
        .reset_index(drop=True)
    )

    print("\nA15: Risk overlay comparison for final candidate")
    print(
        risk_overlay_summary_df
        [
            [
                "sample_split",
                "signal_name",
                "exit_rule",
                "cost_label",
                "risk_overlay",
                "trades",
                "cooldown_triggers",
                "win_rate",
                "avg_net_return",
                "median_net_return",
                "total_return",
                "profit_factor",
                "max_drawdown",
                "best_trade",
                "worst_trade",
            ]
        ]
    )

    final_strategy_config_df = pd.DataFrame(
        [
            {
                "config_item": "main_entry_signal",
                "value": main_entry_signal,
                "comment": "Final working entry signal.",
            },
            {
                "config_item": "main_exit_rule",
                "value": main_exit_rule,
                "comment": "Primary exit rule: 6 bars / 30 minutes.",
            },
            {
                "config_item": "risk_control_exit_rule",
                "value": risk_control_exit_rule,
                "comment": "Conservative variant with -0.40% gross hard stop.",
            },
            {
                "config_item": "main_holding_bars",
                "value": str(main_holding_bars),
                "comment": "Fixed holding horizon for the primary exit.",
            },
        ]
    )

    ### A16: Save outputs
    processed_dir = PROJECT_ROOT / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    feature_output_path = (
        processed_dir
        / "btcusdt_5m_features_90d_fast_refined_signals.csv"
    )

    feature_df.to_csv(feature_output_path, index=False)

    print(f"\nA16: Saved processed feature file to: {feature_output_path}")

    results_dir = PROJECT_ROOT / "results" / "tables"
    results_dir.mkdir(parents=True, exist_ok=True)

    output_tables = {
        "fast_refined_90d_sample_split_summary.csv": sample_split_df,
        "fast_refined_90d_signal_count.csv": signal_count_df,
        "fast_refined_90d_baseline_candidate_rows.csv": candidate_rows_df,
        "fast_refined_90d_trade_distribution_summary.csv": trade_distribution_df,
        "fast_refined_90d_base_cost_ranking.csv": base_cost_df,
        "fast_refined_90d_oos_base_cost_ranking.csv": oos_base_cost_df,
        "fast_refined_90d_cost_sensitivity_pivot.csv": cost_pivot_df,
        "fast_refined_90d_signal_robustness.csv": robustness_df,
        "fast_refined_90d_recommended_signals.csv": recommended_signal_df,
        "fast_refined_90d_trade_results.csv": cumulative_trade_df,
        "fast_refined_90d_final_candidate_curve.csv": final_candidate_curve_df,
        "fast_refined_90d_final_strategy_config.csv": final_strategy_config_df,
        "fast_refined_90d_exit_rule_summary.csv": exit_rule_summary_df,
        "fast_refined_90d_exit_rule_trades.csv": exit_rule_trade_df,
        "fast_refined_90d_risk_overlay_summary.csv": risk_overlay_summary_df,
        "fast_refined_90d_risk_overlay_trades.csv": risk_overlay_trade_df,
    }

    for filename, table_df in output_tables.items():
        output_path = results_dir / filename
        table_df.to_csv(output_path, index=False)

        print(f"A16: Saved {filename} to: {output_path}")


if __name__ == "__main__":
    main()
