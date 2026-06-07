# Experiment Log

## 1. Initial Hypothesis

We tested whether BTCUSDT 5-minute short-squeeze behavior can be captured using downtrend, high volatility, forced-cover pressure, and buy aggression features.

## 2. Baseline Signal

downtrend_highvol_forced_cover_signal

## 3. Refined Signal

dhv_score4_buy60_signal

## 4. Key Improvement

The refined signal filters weak squeeze candidates by requiring:

- forced_cover_score >= 4
- buy_aggression_ratio_15m >= 0.60

## 5. Current Result

The refined signal showed positive average net return in both in-sample and out-of-sample periods under a 6-bar fixed holding period after base transaction costs.

## 6. Limitation

The out-of-sample trade count is small, so the result should be treated as a promising candidate rather than a fully validated strategy.
