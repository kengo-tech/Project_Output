import pandas as pd


def run_event_study(
    df: pd.DataFrame,
    signal_col: str = "long_signal",
    horizons: tuple[int, ...] = (1, 2, 3, 6, 12),
    slippage_rate: float = 0.0002,
) -> pd.DataFrame:
    """
    Run event study after trading signals.

    This is not a trading backtest.
    This checks whether price tends to move favorably after a signal.

    Entry assumption:
        Signal at bar t close.
        Hypothetical entry at bar t+1 open with slippage.

    Parameters
    ----------
    df : pd.DataFrame
        Feature dataframe.
    signal_col : str
        Column name used as event signal.
    horizons : tuple[int, ...]
        Forward horizons in bars.
        For 5-minute data:
        1 = 5 min, 2 = 10 min, 3 = 15 min, 6 = 30 min, 12 = 60 min.
    slippage_rate : float
        Entry slippage assumption.

    Returns
    -------
    pd.DataFrame
        Event-level forward return summary.
    """

    required_columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        signal_col,
        "forced_cover_score",
        "buy_aggression_ratio_15m",
        "cvd_15m",
        "cvd_slope_15m",
        "volume_shock_15m",
        "vol_percentile",
    ]

    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    ### Staging
    event_base_df = (
        df
        .copy()
        .sort_values("open_time")
        .reset_index(drop=True)
    )

    events = []

    max_horizon = max(horizons)

    for signal_idx in range(len(event_base_df) - max_horizon - 1):
        row = event_base_df.iloc[signal_idx]

        if not bool(row[signal_col]):
            continue

        entry_idx = signal_idx + 1
        entry_row = event_base_df.iloc[entry_idx]

        entry_price = entry_row["open"] * (1 + slippage_rate)

        event = {
            "signal_time": row["open_time"],
            "entry_time": entry_row["open_time"],
            "entry_price": entry_price,
            "signal_col": signal_col,
            "forced_cover_score": row["forced_cover_score"],
            "buy_aggression_ratio_15m": row["buy_aggression_ratio_15m"],
            "cvd_15m": row["cvd_15m"],
            "cvd_slope_15m": row["cvd_slope_15m"],
            "volume_shock_15m": row["volume_shock_15m"],
            "vol_percentile": row["vol_percentile"],
        }

        for h in horizons:
            exit_idx = entry_idx + h

            future_close = event_base_df.iloc[exit_idx]["close"]
            future_high = event_base_df.iloc[entry_idx: exit_idx + 1]["high"].max()
            future_low = event_base_df.iloc[entry_idx: exit_idx + 1]["low"].min()

            event[f"ret_{h}b"] = future_close / entry_price - 1
            event[f"mfe_{h}b"] = future_high / entry_price - 1
            event[f"mae_{h}b"] = future_low / entry_price - 1

        events.append(event)

    return pd.DataFrame(events)


def summarize_event_study(event_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize event study output.
    """

    if event_df.empty:
        return pd.DataFrame()

    return_cols = [col for col in event_df.columns if col.startswith("ret_")]
    mfe_cols = [col for col in event_df.columns if col.startswith("mfe_")]
    mae_cols = [col for col in event_df.columns if col.startswith("mae_")]

    summary_rows = []

    for col in return_cols:
        horizon = col.replace("ret_", "")

        summary_rows.append(
            {
                "horizon": horizon,
                "events": len(event_df),
                "avg_return": event_df[col].mean(),
                "median_return": event_df[col].median(),
                "win_rate": (event_df[col] > 0).mean(),
                "avg_mfe": event_df[f"mfe_{horizon}"].mean()
                if f"mfe_{horizon}" in mfe_cols else None,
                "avg_mae": event_df[f"mae_{horizon}"].mean()
                if f"mae_{horizon}" in mae_cols else None,
            }
        )

    return pd.DataFrame(summary_rows)