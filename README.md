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

## Main Finding

The refined signal improved out-of-sample performance compared with the baseline signal, but the out-of-sample trade count remains small.

## How to Run

python main.py
