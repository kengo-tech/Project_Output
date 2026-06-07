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


def run_fast_fixed_horizon_backtest(
    df: pd.DataFrame,
    signal_col: str,
    holding_bars: int,
    fee_rate: float,
    slippage_rate: float,
) -> pd.DataFrame:
    """
    Fast long-only fixed-horizon backtest.

    Entry:
    - Enter at next bar close after signal.

    Exit:
    - Exit after fixed_holding_bars.

    Overlap:
    - Skip new signal while previous trade is still open.

    Cost:
    - Round-trip cost = 2 * fee_rate + 2 * slippage_rate.
    """
    if df.empty:
        return pd.DataFrame()

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

        entry_price = close_array[entry_idx]
        exit_price = close_array[exit_idx]

        if entry_price <= 0 or np.isnan(entry_price) or np.isnan(exit_price):
            continue

        gross_return = (exit_price / entry_price) - 1
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
                "holding_bars": holding_bars,
                "exit_reason": f"fixed_{holding_bars}b_exit",
                "win": net_return > 0,
            }
        )

        next_available_idx = exit_idx + 1

    trade_df = pd.DataFrame(trades)

    return trade_df


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

    signal_candidates = [
        "downtrend_highvol_forced_cover_signal",
        "dhv_buy60_signal",
        "dhv_cvd_positive_signal",
        "dhv_buy60_cvdpos_signal",
        "dhv_score4_signal",
        "dhv_score4_buy60_signal",
        "dhv_score4_buy60_cvdpos_signal",
    ]

    holding_periods = [3, 6, 12]

    cost_configs = [
        {
            "cost_label": "no_cost",
            "fee_rate": 0.0,
            "slippage_rate": 0.0,
        },
        {
            "cost_label": "base_cost",
            "fee_rate": 0.0004,
            "slippage_rate": 0.0002,
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

    ### A6: Lightweight baseline candidate signal rows
    target_signal = "downtrend_highvol_forced_cover_signal"

    candidate_cols = [
        "open_time",
        "close",
        "sample_split",
        "trend_regime",
        "volatility_regime",
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

    print("\nA6: Baseline candidate signal rows")
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
                                round_trip_cost=(
                                    2 * cost_config["fee_rate"]
                                    + 2 * cost_config["slippage_rate"]
                                ),
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
    if recommended_signal_df.empty:
        final_signal = target_signal
        final_holding_bars = 6
    else:
        final_signal = (
            recommended_signal_df
            .loc[lambda x: x["sample_split"] == "out_of_sample"]
            .sort_values("avg_net_return", ascending=False)
            .iloc[0]["signal_name"]
        )

        final_holding_bars = int(
            recommended_signal_df
            .loc[
                lambda x: (
                    (x["sample_split"] == "out_of_sample")
                    & (x["signal_name"] == final_signal)
                )
            ]
            .sort_values("avg_net_return", ascending=False)
            .iloc[0]["holding_bars"]
        )

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

    ### A14: Save outputs
    processed_dir = PROJECT_ROOT / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    feature_output_path = (
        processed_dir
        / "btcusdt_5m_features_90d_fast_refined_signals.csv"
    )

    feature_df.to_csv(feature_output_path, index=False)

    print(f"\nA14: Saved processed feature file to: {feature_output_path}")

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
    }

    for filename, table_df in output_tables.items():
        output_path = results_dir / filename
        table_df.to_csv(output_path, index=False)

        print(f"A14: Saved {filename} to: {output_path}")


if __name__ == "__main__":
    main()