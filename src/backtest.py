import pandas as pd


def run_long_only_backtest(
    df: pd.DataFrame,
    signal_col: str = "long_signal",
    exit_mode: str = "adaptive",
    fee_rate: float = 0.0004,
    slippage_rate: float = 0.0002,
    stop_loss: float = 0.005,
    take_profit: float = 0.010,
    max_holding_bars: int = 12,
    fixed_holding_bars: int = 6,
    min_holding_bars_before_soft_exit: int = 2,
    false_squeeze_score_threshold: int = 2,
    use_risk_exits_in_fixed_mode: bool = False,
    close_open_position_at_end: bool = True,
) -> pd.DataFrame:
    """
    Run a long-only backtest for BTCUSDT short-squeeze momentum signals.

    This backtest is designed for hypothesis testing, not for optimizing returns.

    Parameters
    ----------
    df : pd.DataFrame
        Feature dataframe.

    signal_col : str
        Signal column used for entry.
        Examples:
        - "current_short_squeeze_signal"
        - "trend_filtered_short_squeeze_signal"
        - "early_squeeze_candidate_signal"

    exit_mode : str
        Exit mode.
        - "adaptive":
            Uses stop loss, take profit, extreme volatility exit,
            false squeeze exit, and time stop.
        - "fixed_horizon":
            Exits after fixed_holding_bars.
            By default, this mode does not apply stop loss / take profit.
            This is useful for testing signal quality.

    fee_rate : float
        Fee per side. 0.0004 = 0.04%.

    slippage_rate : float
        Slippage per side. 0.0002 = 0.02%.

    stop_loss : float
        Stop loss from entry price. 0.005 = 0.5%.

    take_profit : float
        Take profit from entry price. 0.010 = 1.0%.

    max_holding_bars : int
        Maximum holding period for adaptive mode.

    fixed_holding_bars : int
        Holding period for fixed_horizon mode.
        6 bars = 30 minutes for 5-minute data.
        12 bars = 60 minutes for 5-minute data.

    min_holding_bars_before_soft_exit : int
        Minimum holding period before false_squeeze exit can trigger.

    false_squeeze_score_threshold : int
        False squeeze exit is triggered when the false_squeeze_score is
        greater than or equal to this threshold.

    use_risk_exits_in_fixed_mode : bool
        If True, fixed_horizon mode also applies stop loss, take profit,
        and extreme volatility exit.
        If False, fixed_horizon mode exits only after fixed_holding_bars.
        False is useful for pure signal-quality testing.

    close_open_position_at_end : bool
        If True, close any open position at the final row.

    Returns
    -------
    pd.DataFrame
        Trade-level backtest results.
    """

    allowed_exit_modes = ["adaptive", "fixed_horizon"]

    if exit_mode not in allowed_exit_modes:
        raise ValueError(
            f"exit_mode must be one of {allowed_exit_modes}. "
            f"Received: {exit_mode}"
        )

    required_columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        signal_col,
        "rolling_high_12",
        "buy_aggression_ratio_15m",
        "cvd_15m",
        "cvd_slope_15m",
        "extreme_vol_flag",
        "forced_cover_score",
    ]

    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    ### Staging
    bt_df = (
        df
        .copy()
        .sort_values("open_time")
        .reset_index(drop=True)
    )

    trades = []

    ### State variables
    in_position = False

    signal_idx = None
    entry_idx = None

    signal_time = None
    entry_time = None

    entry_price = None
    breakout_level = None

    entry_forced_cover_score = None
    entry_buy_aggression_ratio_15m = None
    entry_cvd_15m = None
    entry_cvd_slope_15m = None

    for i in range(len(bt_df) - 1):
        row = bt_df.iloc[i]

        ### Entry logic
        if not in_position:
            if bool(row[signal_col]):
                next_row = bt_df.iloc[i + 1]

                signal_idx = i
                entry_idx = i + 1

                signal_time = row["open_time"]
                entry_time = next_row["open_time"]

                # Buy at next bar open with slippage.
                entry_price = next_row["open"] * (1 + slippage_rate)

                # Store signal context from signal bar.
                breakout_level = row["rolling_high_12"]

                entry_forced_cover_score = row["forced_cover_score"]
                entry_buy_aggression_ratio_15m = row["buy_aggression_ratio_15m"]
                entry_cvd_15m = row["cvd_15m"]
                entry_cvd_slope_15m = row["cvd_slope_15m"]

                in_position = True

            continue

        ### Exit logic
        if in_position:
            current_row = bt_df.iloc[i]

            holding_bars = i - entry_idx

            current_time = current_row["open_time"]
            current_close = current_row["close"]

            stop_price = entry_price * (1 - stop_loss)
            target_price = entry_price * (1 + take_profit)

            exit_reason = None
            exit_price = None
            false_squeeze_score = 0

            ### Fixed-horizon mode
            if exit_mode == "fixed_horizon":
                if use_risk_exits_in_fixed_mode:
                    ### Optional hard stop loss
                    if current_row["low"] <= stop_price:
                        exit_reason = "stop_loss"
                        exit_price = stop_price * (1 - slippage_rate)

                    ### Optional take profit
                    elif current_row["high"] >= target_price:
                        exit_reason = "take_profit"
                        exit_price = target_price * (1 - slippage_rate)

                    ### Optional extreme volatility exit
                    elif bool(current_row["extreme_vol_flag"]):
                        exit_reason = "extreme_vol_exit"
                        exit_price = current_close * (1 - slippage_rate)

                if exit_reason is None and holding_bars >= fixed_holding_bars:
                    exit_reason = f"fixed_{fixed_holding_bars}b_exit"
                    exit_price = current_close * (1 - slippage_rate)

            ### Adaptive mode
            elif exit_mode == "adaptive":
                ### Hard exit 1: stop loss
                # Conservative assumption:
                # if stop loss and take profit are both touched in the same bar,
                # stop loss is prioritized.
                if current_row["low"] <= stop_price:
                    exit_reason = "stop_loss"
                    exit_price = stop_price * (1 - slippage_rate)

                ### Hard exit 2: take profit
                elif current_row["high"] >= target_price:
                    exit_reason = "take_profit"
                    exit_price = target_price * (1 - slippage_rate)

                ### Hard exit 3: extreme volatility
                elif bool(current_row["extreme_vol_flag"]):
                    exit_reason = "extreme_vol_exit"
                    exit_price = current_close * (1 - slippage_rate)

                else:
                    ### Soft false-squeeze diagnostics
                    price_back_below_breakout = current_close < breakout_level

                    buy_aggression_weakened = (
                        current_row["buy_aggression_ratio_15m"] < 0.50
                    )

                    cvd_turned_negative = current_row["cvd_15m"] < 0

                    cvd_slope_weakened = current_row["cvd_slope_15m"] < 0

                    if price_back_below_breakout:
                        false_squeeze_score += 1

                    if buy_aggression_weakened:
                        false_squeeze_score += 1

                    if cvd_turned_negative:
                        false_squeeze_score += 1

                    if cvd_slope_weakened:
                        false_squeeze_score += 1

                    ### Soft exit: false squeeze
                    if (
                        holding_bars >= min_holding_bars_before_soft_exit
                        and false_squeeze_score >= false_squeeze_score_threshold
                    ):
                        exit_reason = "false_squeeze"
                        exit_price = current_close * (1 - slippage_rate)

                    ### Time exit
                    elif holding_bars >= max_holding_bars:
                        exit_reason = "time_stop"
                        exit_price = current_close * (1 - slippage_rate)

            ### Record trade if exit is triggered
            if exit_reason is not None:
                gross_return = exit_price / entry_price - 1

                # Fee is charged both on entry and exit.
                net_return = gross_return - (2 * fee_rate)

                trades.append(
                    {
                        "signal_name": signal_col,
                        "exit_mode": exit_mode,
                        "fixed_holding_bars": fixed_holding_bars
                        if exit_mode == "fixed_horizon"
                        else None,
                        "signal_idx": signal_idx,
                        "entry_idx": entry_idx,
                        "exit_idx": i,
                        "signal_time": signal_time,
                        "entry_time": entry_time,
                        "exit_time": current_time,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "gross_return": gross_return,
                        "net_return": net_return,
                        "holding_bars": holding_bars,
                        "exit_reason": exit_reason,
                        "false_squeeze_score": false_squeeze_score,
                        "breakout_level": breakout_level,
                        "entry_forced_cover_score": entry_forced_cover_score,
                        "entry_buy_aggression_ratio_15m": entry_buy_aggression_ratio_15m,
                        "entry_cvd_15m": entry_cvd_15m,
                        "entry_cvd_slope_15m": entry_cvd_slope_15m,
                        "exit_buy_aggression_ratio_15m": current_row[
                            "buy_aggression_ratio_15m"
                        ],
                        "exit_cvd_15m": current_row["cvd_15m"],
                        "exit_cvd_slope_15m": current_row["cvd_slope_15m"],
                    }
                )

                ### Reset state
                in_position = False

                signal_idx = None
                entry_idx = None

                signal_time = None
                entry_time = None

                entry_price = None
                breakout_level = None

                entry_forced_cover_score = None
                entry_buy_aggression_ratio_15m = None
                entry_cvd_15m = None
                entry_cvd_slope_15m = None

    ### Close open position at the end of the sample if needed
    if in_position and close_open_position_at_end:
        final_row = bt_df.iloc[-1]

        exit_price = final_row["close"] * (1 - slippage_rate)
        gross_return = exit_price / entry_price - 1
        net_return = gross_return - (2 * fee_rate)

        holding_bars = len(bt_df) - 1 - entry_idx

        trades.append(
            {
                "signal_name": signal_col,
                "exit_mode": exit_mode,
                "fixed_holding_bars": fixed_holding_bars
                if exit_mode == "fixed_horizon"
                else None,
                "signal_idx": signal_idx,
                "entry_idx": entry_idx,
                "exit_idx": len(bt_df) - 1,
                "signal_time": signal_time,
                "entry_time": entry_time,
                "exit_time": final_row["open_time"],
                "entry_price": entry_price,
                "exit_price": exit_price,
                "gross_return": gross_return,
                "net_return": net_return,
                "holding_bars": holding_bars,
                "exit_reason": "end_of_sample",
                "false_squeeze_score": None,
                "breakout_level": breakout_level,
                "entry_forced_cover_score": entry_forced_cover_score,
                "entry_buy_aggression_ratio_15m": entry_buy_aggression_ratio_15m,
                "entry_cvd_15m": entry_cvd_15m,
                "entry_cvd_slope_15m": entry_cvd_slope_15m,
                "exit_buy_aggression_ratio_15m": final_row[
                    "buy_aggression_ratio_15m"
                ],
                "exit_cvd_15m": final_row["cvd_15m"],
                "exit_cvd_slope_15m": final_row["cvd_slope_15m"],
            }
        )

    trade_df = pd.DataFrame(trades)

    if trade_df.empty:
        return trade_df

    trade_df = (
        trade_df
        .copy()
        .assign(
            cumulative_return=lambda x: (
                x
                .groupby(["signal_name", "exit_mode", "fixed_holding_bars"], dropna=False)[
                    "net_return"
                ]
                .transform(lambda s: (1 + s).cumprod() - 1)
            ),
            win=lambda x: x["net_return"] > 0,
        )
    )

    return trade_df