# QF635 BTCUSDT Short-Squeeze Strategy

This project tests a BTCUSDT 5-minute short-squeeze style algorithmic trading strategy using Binance API data.

## Main Hypothesis

Short-term rebounds may occur when BTCUSDT is in a downtrend and high-volatility regime, especially when forced-cover pressure and buy aggression are strong.

## Current Refined Signal

- Signal: dhv_score4_buy60_signal
- Conditions:
  - downtrend high-volatility forced-cover setup
  - forced_cover_score >= 4
  - buy_aggression_ratio_15m >= 0.60
- Holding period: 6 bars / 30 minutes
- Cost setting: base transaction cost
- Validation: in-sample and out-of-sample comparison

## Next Candidate Signal

- Signal: downtrend_exhaustion_reclaim_signal
- Idea:
  - detect downtrend exhaustion rather than downtrend alone
  - require price to stop making short-term new lows
  - require short-term high reclaim
  - require taker-buy / CVD / volume confirmation
- Conditions:
  - close < ma_6h
  - low > rolling_low_12
  - close > rolling_high_12
  - forced_cover_score >= 3
  - buy_aggression_ratio_15m >= 0.58
  - cvd_slope_15m > 0
  - volume_shock_15m >= 1.20
  - 0.50 <= vol_percentile < 0.95
- Primary holding period: 6 bars / 30 minutes
- Comparison holding period: 12 bars / 60 minutes

## Main Finding

The refined and exhaustion-reclaim signals improved out-of-sample performance compared with the baseline signal, but the out-of-sample trade count remains small.

## Current Conclusion

The main working candidate is downtrend_exhaustion_reclaim_signal with a fixed
6-bar / 30-minute exit. hard_stop_6b is kept as a conservative risk-control
variant.

See docs/final_strategy_summary.md for the report-ready summary, including
execution-cost stress, regime fit, risk scenarios, and live-execution
limitations.

## How to Run

python main.py
