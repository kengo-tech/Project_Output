import numpy as np
import pandas as pd


def _last_percentile(values: np.ndarray) -> float:
    """
    Return the percentile rank of the last value within a rolling window.

    This uses only data inside the current rolling window, so it avoids
    using future information.
    """
    s = pd.Series(values).dropna()

    if len(s) == 0:
        return np.nan

    return s.rank(pct=True).iloc[-1]


def make_features(
    df: pd.DataFrame,
    volume_window: int = 288,
    volatility_percentile_window: int = 288,
    near_high_threshold: float = 0.001,
) -> pd.DataFrame:
    """
    Create features for the BTCUSDT short-squeeze momentum strategy.

    Parameters
    ----------
    df : pd.DataFrame
        Binance kline dataframe.

    volume_window : int
        Rolling window for average volume calculation.
        288 bars = 24 hours for 5-minute data.

    volatility_percentile_window : int
        Rolling window for volatility percentile calculation.
        288 bars = 24 hours for 5-minute data.

    near_high_threshold : float
        Distance threshold for "near rolling high".
        0.001 means within 0.10% below the rolling high.

    Returns
    -------
    pd.DataFrame
        Feature-enriched dataframe.
    """

    required_columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "number_of_trades",
        "taker_buy_base_volume",
    ]

    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    ### Staging
    staged_df = (
        df
        .copy()
        .sort_values("open_time")
        .reset_index(drop=True)
    )

    ### Cleaning
    clean_df = (
        staged_df
        .dropna(how="all")
        .assign(
            open=lambda x: pd.to_numeric(x["open"], errors="coerce"),
            high=lambda x: pd.to_numeric(x["high"], errors="coerce"),
            low=lambda x: pd.to_numeric(x["low"], errors="coerce"),
            close=lambda x: pd.to_numeric(x["close"], errors="coerce"),
            volume=lambda x: pd.to_numeric(x["volume"], errors="coerce"),
            number_of_trades=lambda x: pd.to_numeric(
                x["number_of_trades"],
                errors="coerce",
            ),
            taker_buy_base_volume=lambda x: pd.to_numeric(
                x["taker_buy_base_volume"],
                errors="coerce",
            ),
        )
    )

    ### Features: price and return
    feature_df = (
        clean_df
        .assign(
            return_5m=lambda x: x["close"].pct_change(),
            return_15m=lambda x: x["close"].pct_change(3),
            return_30m=lambda x: x["close"].pct_change(6),
            return_60m=lambda x: x["close"].pct_change(12),
            return_3h=lambda x: x["close"].pct_change(36),
            return_6h=lambda x: x["close"].pct_change(72),
            return_24h=lambda x: x["close"].pct_change(288),
            rolling_high_12=lambda x: x["high"].shift(1).rolling(12).max(),
            rolling_high_24=lambda x: x["high"].shift(1).rolling(24).max(),
            rolling_low_12=lambda x: x["low"].shift(1).rolling(12).min(),
            rolling_low_24=lambda x: x["low"].shift(1).rolling(24).min(),
        )
    )

    ### Features: trend and moving averages
    feature_df = (
        feature_df
        .assign(
            ma_1h=lambda x: x["close"].shift(1).rolling(12).mean(),
            ma_3h=lambda x: x["close"].shift(1).rolling(36).mean(),
            ma_6h=lambda x: x["close"].shift(1).rolling(72).mean(),
            above_ma_6h=lambda x: x["close"] > x["ma_6h"],
            downtrend_ma_6h=lambda x: x["close"] < x["ma_6h"],
            trend_up_6h=lambda x: x["return_6h"] > 0,
        )
    )

    ### Features: distance to breakout levels
    feature_df = (
        feature_df
        .assign(
            distance_to_rolling_high_12=lambda x: (
                x["close"] / x["rolling_high_12"] - 1
            ),
            distance_to_rolling_high_24=lambda x: (
                x["close"] / x["rolling_high_24"] - 1
            ),
            reclaim_strength_12=lambda x: (
                x["close"] / x["rolling_high_12"] - 1
            ),
            near_rolling_high_12=lambda x: (
                (x["close"] >= x["rolling_high_12"] * (1 - near_high_threshold))
                & (x["close"] <= x["rolling_high_12"])
            ),
            near_rolling_high_24=lambda x: (
                (x["close"] >= x["rolling_high_24"] * (1 - near_high_threshold))
                & (x["close"] <= x["rolling_high_24"])
            ),
            no_new_low_12=lambda x: x["low"] > x["rolling_low_12"],
            no_new_low_24=lambda x: x["low"] > x["rolling_low_24"],
        )
    )

    ### Features: candle structure
    feature_df = (
        feature_df
        .assign(
            candle_range=lambda x: x["high"] - x["low"],
            candle_body=lambda x: (x["close"] - x["open"]).abs(),
            upper_wick=lambda x: x["high"] - x[["open", "close"]].max(axis=1),
            lower_wick=lambda x: x[["open", "close"]].min(axis=1) - x["low"],
        )
        .assign(
            upper_wick_ratio=lambda x: np.where(
                x["candle_range"] > 0,
                x["upper_wick"] / x["candle_range"],
                np.nan,
            ),
            lower_wick_ratio=lambda x: np.where(
                x["candle_range"] > 0,
                x["lower_wick"] / x["candle_range"],
                np.nan,
            ),
        )
    )

    ### Features: taker buy/sell volume and CVD proxy
    feature_df = (
        feature_df
        .assign(
            taker_sell_base_volume=lambda x: (
                x["volume"] - x["taker_buy_base_volume"]
            ).clip(lower=0),
            buy_aggression_ratio=lambda x: np.where(
                x["volume"] > 0,
                x["taker_buy_base_volume"] / x["volume"],
                np.nan,
            ),
            sell_aggression_ratio=lambda x: np.where(
                x["volume"] > 0,
                (x["volume"] - x["taker_buy_base_volume"]) / x["volume"],
                np.nan,
            ),
            volume_delta=lambda x: (
                x["taker_buy_base_volume"] - x["taker_sell_base_volume"]
            ),
        )
        .assign(
            cvd=lambda x: x["volume_delta"].cumsum(),
            cvd_15m=lambda x: x["volume_delta"].rolling(3).sum(),
            cvd_30m=lambda x: x["volume_delta"].rolling(6).sum(),
            cvd_60m=lambda x: x["volume_delta"].rolling(12).sum(),
        )
        .assign(
            cvd_slope_15m=lambda x: x["cvd_15m"] - x["cvd_15m"].shift(1),
            cvd_slope_30m=lambda x: x["cvd_30m"] - x["cvd_30m"].shift(1),
            cvd_slope_60m=lambda x: x["cvd_60m"] - x["cvd_60m"].shift(1),
            rolling_cvd_change_15m=lambda x: x["cvd"] - x["cvd"].shift(3),
            rolling_cvd_change_60m=lambda x: x["cvd"] - x["cvd"].shift(12),
            cvd_slope_improving=lambda x: (
                x["cvd_slope_15m"] > x["cvd_slope_15m"].shift(1)
            ),
        )
    )

    ### Features: cumulative volume and volume shock
    feature_df = (
        feature_df
        .assign(
            volume_15m=lambda x: x["volume"].rolling(3).sum(),
            volume_30m=lambda x: x["volume"].rolling(6).sum(),
            volume_60m=lambda x: x["volume"].rolling(12).sum(),
            taker_buy_volume_15m=lambda x: (
                x["taker_buy_base_volume"].rolling(3).sum()
            ),
            taker_buy_volume_30m=lambda x: (
                x["taker_buy_base_volume"].rolling(6).sum()
            ),
            taker_buy_volume_60m=lambda x: (
                x["taker_buy_base_volume"].rolling(12).sum()
            ),
            taker_sell_volume_15m=lambda x: (
                x["taker_sell_base_volume"].rolling(3).sum()
            ),
            taker_sell_volume_30m=lambda x: (
                x["taker_sell_base_volume"].rolling(6).sum()
            ),
            taker_sell_volume_60m=lambda x: (
                x["taker_sell_base_volume"].rolling(12).sum()
            ),
        )
        .assign(
            avg_volume_15m=lambda x: (
                x["volume_15m"].shift(1).rolling(volume_window).mean()
            ),
            avg_taker_buy_volume_15m=lambda x: (
                x["taker_buy_volume_15m"].shift(1).rolling(volume_window).mean()
            ),
            avg_number_of_trades=lambda x: (
                x["number_of_trades"].shift(1).rolling(volume_window).mean()
            ),
        )
        .assign(
            volume_shock_15m=lambda x: np.where(
                x["avg_volume_15m"] > 0,
                x["volume_15m"] / x["avg_volume_15m"],
                np.nan,
            ),
            taker_buy_volume_shock_15m=lambda x: np.where(
                x["avg_taker_buy_volume_15m"] > 0,
                x["taker_buy_volume_15m"] / x["avg_taker_buy_volume_15m"],
                np.nan,
            ),
            number_of_trades_shock=lambda x: np.where(
                x["avg_number_of_trades"] > 0,
                x["number_of_trades"] / x["avg_number_of_trades"],
                np.nan,
            ),
            buy_aggression_ratio_15m=lambda x: np.where(
                x["volume_15m"] > 0,
                x["taker_buy_volume_15m"] / x["volume_15m"],
                np.nan,
            ),
            sell_aggression_ratio_15m=lambda x: np.where(
                x["volume_15m"] > 0,
                x["taker_sell_volume_15m"] / x["volume_15m"],
                np.nan,
            ),
            buy_aggression_ratio_30m=lambda x: np.where(
                x["volume_30m"] > 0,
                x["taker_buy_volume_30m"] / x["volume_30m"],
                np.nan,
            ),
            buy_aggression_ratio_60m=lambda x: np.where(
                x["volume_60m"] > 0,
                x["taker_buy_volume_60m"] / x["volume_60m"],
                np.nan,
            ),
            sell_aggression_ratio_30m=lambda x: np.where(
                x["volume_30m"] > 0,
                x["taker_sell_volume_30m"] / x["volume_30m"],
                np.nan,
            ),
            sell_aggression_ratio_60m=lambda x: np.where(
                x["volume_60m"] > 0,
                x["taker_sell_volume_60m"] / x["volume_60m"],
                np.nan,
            ),
            sell_pressure_60m=lambda x: np.where(
                x["volume_60m"] > 0,
                (
                    x["taker_sell_volume_60m"]
                    - x["taker_buy_volume_60m"]
                )
                / x["volume_60m"],
                np.nan,
            ),
            anchored_cvd_60m=lambda x: x["cvd"] - x["cvd"].shift(12),
        )
        .assign(
            sell_pressure_60m_zscore=lambda x: (
                (
                    x["sell_pressure_60m"]
                    - x["sell_pressure_60m"].shift(1).rolling(288, min_periods=50).mean()
                )
                / x["sell_pressure_60m"].shift(1).rolling(288, min_periods=50).std()
            ),
            buy_aggression_improving=lambda x: (
                x["buy_aggression_ratio_15m"]
                > x["buy_aggression_ratio_15m"].shift(1)
            ),
            sell_aggression_improving=lambda x: (
                x["sell_aggression_ratio_15m"]
                > x["sell_aggression_ratio_15m"].shift(1)
            ),
        )
    )

    ### Features: volatility
    feature_df = (
        feature_df
        .assign(
            rolling_vol_1h=lambda x: x["return_5m"].rolling(12).std(),
            rolling_vol_3h=lambda x: x["return_5m"].rolling(36).std(),
            rolling_vol_6h=lambda x: x["return_5m"].rolling(72).std(),
            vol_expansion=lambda x: (
                x["rolling_vol_3h"] > x["rolling_vol_3h"].shift(1)
            ),
        )
    )

    feature_df["vol_percentile"] = (
        feature_df["rolling_vol_3h"]
        .rolling(volatility_percentile_window, min_periods=50)
        .apply(_last_percentile, raw=True)
    )

    ### Features: volatility flags
    feature_df = (
        feature_df
        .assign(
            extreme_vol_flag=lambda x: x["vol_percentile"] >= 0.95,
            high_vol_flag=lambda x: (
                (x["vol_percentile"] >= 0.70)
                & (x["vol_percentile"] < 0.95)
            ),
            low_vol_flag=lambda x: x["vol_percentile"] < 0.30,
        )
    )

    ### Features: short build-up and sell absorption
    feature_df = (
        feature_df
        .assign(
            sell_absorption_signal=lambda x: (
                (x["cvd_30m"] <= 0)
                & (x["return_30m"] >= -0.001)
            ),
            sell_aggression_absorption_signal=lambda x: (
                (x["sell_aggression_ratio_30m"] > 0.55)
                & (x["return_30m"] >= -0.0015)
            ),
        )
        .assign(
            short_build_up=lambda x: (
                x["sell_absorption_signal"]
                | x["sell_aggression_absorption_signal"]
            ),
            seller_exhaustion_setup_signal=lambda x: (
                x["downtrend_ma_6h"]
                & (x["cvd_30m"] <= 0)
                & (x["return_30m"] >= -0.0015)
                & x["no_new_low_12"]
            ),
        )
    )

    ### Features: squeeze triggers
    feature_df = (
        feature_df
        .assign(
            squeeze_trigger_12=lambda x: x["close"] > x["rolling_high_12"],
            squeeze_trigger_24=lambda x: x["close"] > x["rolling_high_24"],
            reclaim_breakout_12=lambda x: x["close"] > x["rolling_high_12"],
        )
    )

    ### Features: forced-cover confirmation score
    feature_df = (
        feature_df
        .assign(
            forced_cover_score=lambda x: (
                (x["buy_aggression_ratio_15m"] > 0.60).astype(int)
                + (x["taker_buy_volume_shock_15m"] > 1.50).astype(int)
                + (x["cvd_15m"] > 0).astype(int)
                + (x["cvd_slope_15m"] > 0).astype(int)
                + (x["volume_shock_15m"] > 1.50).astype(int)
            )
        )
    )

    ### Features: short build-up memory
    feature_df = (
        feature_df
        .assign(
            short_build_up_recent=lambda x: (
                x["short_build_up"]
                .astype(float)
                .shift(1)
                .rolling(6)
                .max()
                .fillna(0)
                .astype(bool)
            )
        )
    )

    ### Main current short-squeeze signal
    feature_df = (
        feature_df
        .assign(
            long_signal=lambda x: (
                x["short_build_up_recent"]
                & x["squeeze_trigger_12"]
                & (x["forced_cover_score"] >= 3)
                & (~x["extreme_vol_flag"])
            )
        )
    )

    ### Alternative signals for hypothesis testing
    feature_df = (
        feature_df
        .assign(
            simple_breakout_signal=lambda x: (
                x["squeeze_trigger_12"]
                & (~x["extreme_vol_flag"])
            ),
            breakout_forced_cover_signal=lambda x: (
                x["squeeze_trigger_12"]
                & (x["forced_cover_score"] >= 3)
                & (~x["extreme_vol_flag"])
            ),
            current_short_squeeze_signal=lambda x: x["long_signal"],
            trend_filtered_short_squeeze_signal=lambda x: (
                x["long_signal"]
                & x["trend_up_6h"]
                & x["above_ma_6h"]
            ),
            early_squeeze_candidate_signal=lambda x: (
                x["short_build_up_recent"]
                & x["near_rolling_high_12"]
                & x["cvd_slope_improving"]
                & x["buy_aggression_improving"]
                & (x["forced_cover_score"] >= 2)
                & (~x["extreme_vol_flag"])
            ),
            downtrend_exhaustion_reclaim_signal=lambda x: (
                x["downtrend_ma_6h"]
                & x["no_new_low_12"]
                & x["squeeze_trigger_12"]
                & (x["forced_cover_score"] >= 3)
                & (x["buy_aggression_ratio_15m"] >= 0.58)
                & (x["cvd_slope_15m"] > 0)
                & (x["volume_shock_15m"] >= 1.20)
                & (x["vol_percentile"] >= 0.50)
                & (x["vol_percentile"] < 0.95)
            ),
            downtrend_exhaustion_reclaim_ret30_signal=lambda x: (
                x["downtrend_exhaustion_reclaim_signal"]
                & (x["return_30m"] > -0.002)
            ),
            downtrend_exhaustion_reclaim_strength_signal=lambda x: (
                x["downtrend_exhaustion_reclaim_signal"]
                & (x["reclaim_strength_12"] >= 0.0003)
            ),
            downtrend_exhaustion_reclaim_wick_signal=lambda x: (
                x["downtrend_exhaustion_reclaim_signal"]
                & (x["lower_wick_ratio"] >= 0.10)
            ),
            absorption_flag=lambda x: (
                (x["sell_pressure_60m"] > 0.10)
                & (x["return_60m"] > -0.003)
            ),
            cvd_flip_15m=lambda x: (
                (x["cvd_slope_15m"] > 0)
                & (x["cvd_slope_60m"] < 0)
            ),
            cvd_flip_recent_3=lambda x: (
                x["cvd_flip_15m"]
                .astype(float)
                .shift(1)
                .rolling(3)
                .max()
                .fillna(0)
                .astype(bool)
            ),
            seller_advantage_break_signal=lambda x: (
                x["downtrend_ma_6h"]
                & x["absorption_flag"]
                & x["cvd_flip_15m"]
                & x["reclaim_breakout_12"]
                & (x["buy_aggression_ratio_15m"] > 0.55)
            ),
            seller_advantage_break_recent3_signal=lambda x: (
                x["downtrend_ma_6h"]
                & x["absorption_flag"]
                & x["cvd_flip_recent_3"]
                & x["reclaim_breakout_12"]
                & (x["buy_aggression_ratio_15m"] > 0.55)
            ),
        )
    )

    return feature_df
