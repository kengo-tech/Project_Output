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

## 7. Next Candidate: Downtrend Exhaustion Reclaim

The next hypothesis is to avoid using downtrend alone as an entry condition.
Instead, the strategy looks for a downtrend that is starting to lose downside
momentum:

- price is below the 6-hour moving average
- price no longer makes a new short-term low
- price reclaims a short-term rolling high
- forced-cover confirmation appears through taker-buy aggression, positive CVD
  slope, and volume expansion

Candidate signal:

downtrend_exhaustion_reclaim_signal

Primary holding period:

- 6 bars / 30 minutes

Comparison holding period:

- 12 bars / 60 minutes

Current limitation:

- Open interest is not available in the current spot kline dataset.
- If futures open interest is added, the preferred confirmation is price up,
  open interest down, CVD up, and taker-buy ratio up after breakout.

## 8. Seller Advantage Break Candidate

The next narrow test starts from this rule:

if downtrend and seller_exhaustion and breakout and forced_cover_confirmation,
enter long at next bar open.

Implemented candidate:

seller_advantage_break_signal

Core conditions:

- downtrend_ma_6h
- absorption_flag
- cvd_flip_15m
- reclaim_breakout_12
- buy_aggression_ratio_15m > 0.55

New diagnostics:

- sell_pressure_60m
- sell_pressure_60m_zscore
- rolling_cvd_change_15m
- rolling_cvd_change_60m
- cvd_slope_60m
- anchored_cvd_60m

Initial read:

- The strict version fires very rarely in the current 90-day sample.
- It should remain a diagnostic candidate until more trades or futures open
  interest confirmation are available.

Recent CVD flip variant:

- seller_advantage_break_recent3_signal treats cvd_flip_15m as valid if it
  occurred within the previous 3 bars.
- This increases the number of signals but does not improve the first 90-day
  validation result.
- The result suggests that CVD flip plus reclaim breakout is not sufficient by
  itself in the current spot-only dataset.

## 9. Downtrend Exhaustion Reclaim Variants

The current main candidate remains:

downtrend_exhaustion_reclaim_signal

Four narrow variants were compared:

- A: original downtrend_exhaustion_reclaim_signal
- B: A + return_30m > -0.002
- C: A + reclaim_strength_12 >= 0.0003
- D: A + lower_wick_ratio >= 0.10

Initial read:

- B is identical to A in the current sample because all A signals already pass
  the return_30m filter.
- C and D improve out-of-sample average return, but they weaken in-sample
  performance and reduce trade count.
- For now, A should remain the main candidate. C and D should be treated as
  diagnostic filters rather than promoted rules.

## 10. Exit Rule Comparison for Main Candidate

The entry remains fixed:

downtrend_exhaustion_reclaim_signal

Compared exit rules:

- fixed_6b: exit after 6 bars / 30 minutes
- hard_stop_6b: exit after 6 bars unless gross return hits -0.40%
- half_tp_thesis_stop_6b: take half at +0.30%, then exit the rest on thesis
  stop or after 6 bars

Thesis stop:

- close < rolling_high_12
- cvd_slope_15m < 0
- active only after at least 2 bars of holding

Initial read:

- hard_stop_6b slightly improves in-sample results by cutting the worst losses.
- hard_stop_6b is unchanged out-of-sample because no OOS trade hits the stop.
- half_tp_thesis_stop_6b reduces both in-sample and out-of-sample expectancy.
- The current best simple exit remains fixed_6b, with hard_stop_6b as a
  conservative risk-control variant.

## 11. Final Candidate Lock

Priority 1 is locked as:

- main_entry_signal: downtrend_exhaustion_reclaim_signal
- main_exit_rule: fixed_6b
- risk_control_exit_rule: hard_stop_6b
- main_holding_bars: 6

The ranking tables remain useful for diagnostics, but the final candidate curve
is now explicitly fixed to the main entry signal and 6-bar exit instead of being
selected automatically from the out-of-sample ranking.

## 12. Dynamic Slippage and Spread Stress

Priority 2 adds execution-cost stress testing.

The current dataset does not include bid-ask spread or order book depth, so the
stress model uses volatility percentile and volume-shock proxies:

- base_cost: fixed fee and fixed slippage
- dynamic_slippage: slippage increases when volatility percentile is high or
  volume-shock liquidity proxy is weak
- spread_stress: a more conservative spread/slippage stress case

Initial read for downtrend_exhaustion_reclaim_signal with fixed_6b:

- out-of-sample average net return remains positive under dynamic_slippage and
  spread_stress
- in-sample performance is much more sensitive and becomes roughly flat under
  spread_stress
- this means the signal still has promise, but the edge is not large enough to
  ignore execution quality

Implementation note:

- spread is not directly observed in the current kline dataset
- this is a proxy stress test, not a full order book execution simulation

## 13. Daily Loss Limit and Consecutive-Loss Cooldown

Priority 3 adds research-level risk overlays:

- daily_loss_1pct: stop opening new trades for the day after daily net return
  reaches -1.00%
- loss2_cooldown_12b: after 2 consecutive losing trades, skip new entries for
  12 bars / 60 minutes
- daily_loss_1pct_loss2_cooldown_12b: combined overlay

Initial read:

- The current final candidate trades infrequently, so these overlays do not
  materially change the first 90-day backtest result.
- Cooldown can be triggered in some cost scenarios, but there are usually no
  immediately following signals to skip.
- The overlays should be kept as live-risk controls, not as alpha-improving
  strategy rules.

Implementation note:

- The overlays prevent new entries only.
- They do not simulate forced liquidation of an existing open trade.
- Live trading still requires account-level position reconciliation and kill
  switch logic.

## 14. Partial Fill and Residual Position Design Note

Priority 4 is treated as a live-execution design requirement rather than a
backtest rule.

Reason:

- The current data is 5-minute bar data.
- It does not contain order book queue position, exchange execution reports, or
  partial-fill events.
- Therefore, adding a partial-fill simulator here would be more assumption than
  evidence.

Live risk:

- An entry order may fill only partially.
- An exit order may close only part of the position.
- A residual long position may remain after the strategy believes it is flat.
- API delay, cancel/reject events, or retry logic may create a mismatch between
  the local strategy state and the exchange-reported position.

Required live behavior before deployment:

- Track submitted quantity, confirmed filled quantity, and remaining quantity.
- Size all exits from confirmed filled quantity only.
- Use reduce-only exit orders where supported.
- Reconcile local position against exchange-reported position before allowing a
  new entry.
- If residual position exists after an exit, send a forced flatten order.
- If order state is unknown, stop new entries, cancel open orders, reconcile
  position, and resume only after state is clean.
- Use idempotent client order IDs so retries do not create duplicate exposure.

Research implication:

The final candidate intentionally keeps execution simple:

- no averaging down
- no scale-in
- no half-exit
- fixed_6b as the main exit
- hard_stop_6b as the conservative risk-control version

This reduces the number of live order states, but it does not eliminate the
need for partial-fill handling and residual-position detection.
