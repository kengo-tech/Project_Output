# Final Strategy Summary

## 1. Main Candidate

The current main candidate is:

downtrend_exhaustion_reclaim_signal

This signal is designed to capture a short-cover rebound after a downtrend
starts to lose downside momentum.

Entry rule:

if downtrend_exhaustion_reclaim_signal is true, enter long at the next bar open.

Core conditions:

- close < ma_6h
- low > rolling_low_12
- close > rolling_high_12
- forced_cover_score >= 3
- buy_aggression_ratio_15m >= 0.58
- cvd_slope_15m > 0
- volume_shock_15m >= 1.20
- 0.50 <= vol_percentile < 0.95

## 2. Why 30-Minute Holding Is Natural

The economic hypothesis is that the edge comes from short-cover pressure, not
from a long trend-following move.

Short covering is expected to be short-lived:

- price stops making new short-term lows
- price reclaims a short-term high
- taker-buy aggression and CVD turn positive
- volume expands during the reclaim

This is closer to a short burst of forced buying than a long directional trend.
Therefore, a 6-bar / 30-minute fixed holding period is more consistent with the
hypothesis than a 12-bar / 60-minute holding period.

## 3. Validation Read

Under base transaction cost, the 6-bar fixed exit is the cleanest current exit.

For downtrend_exhaustion_reclaim_signal:

- In-sample, 6 bars: positive average net return with positive median net return
- Out-of-sample, 6 bars: positive average net return with positive median net return
- 12 bars can still work in some cases, but it is less aligned with the
  short-cover burst hypothesis

The result should still be treated as a promising candidate rather than a fully
validated strategy because the out-of-sample trade count remains small.

## 4. Exit Decision

Main exit:

fixed_6b

- exit after 6 bars / 30 minutes
- simple and aligned with the short-cover burst hypothesis
- currently the best clean exit rule

Risk-control variant:

hard_stop_6b

- exit after 6 bars unless gross return hits -0.40%
- keeps the same 30-minute structure
- slightly improves in-sample results by cutting the worst losses
- unchanged out-of-sample in the current sample because no OOS trade hit the stop

Rejected for now:

half_tp_thesis_stop_6b

- taking half profit and exiting on thesis stop reduced expectancy in the first
  validation
- the rule exits too early for this signal's current behavior

## 5. Execution Cost Stress

The strategy was also checked under execution-cost stress.

Because the current dataset uses Binance kline data and does not include bid-ask
spread or order book depth, spread stress is modeled with proxies:

- volatility percentile
- volume shock

Cost scenarios:

- base_cost: fixed fee and fixed slippage
- dynamic_slippage: slippage increases in high-volatility or weaker-liquidity
  proxy conditions
- spread_stress: more conservative stressed slippage

Initial read:

- out-of-sample average net return remains positive under dynamic_slippage and
  spread_stress
- in-sample performance is sensitive to spread_stress and becomes close to flat
- execution quality is therefore a key risk, not a secondary detail

This supports keeping the strategy simple and short-horizon. It also means a
live implementation should not assume the base backtest cost is always
achievable.

## 6. Risk Overlay

The strategy was also checked with simple risk overlays:

- daily_loss_1pct
- loss2_cooldown_12b
- daily_loss_1pct_loss2_cooldown_12b

Definitions:

- daily_loss_1pct stops new entries for the day after daily net return reaches
  -1.00%
- loss2_cooldown_12b skips new entries for 12 bars / 60 minutes after 2
  consecutive losing trades

Initial read:

- These overlays do not materially change the current 90-day backtest because
  the final candidate trades infrequently.
- They are still useful as live-risk controls.
- They should not be presented as an alpha-improving rule.

Important limitation:

- These overlays only stop new entries.
- They do not solve partial fills, residual positions, exchange rejects, or
  emergency flattening.

## 7. Current Conclusion

The final working candidate is:

Entry:

downtrend_exhaustion_reclaim_signal

Primary exit:

fixed_6b

Risk-control exit:

hard_stop_6b

This keeps the strategy narrow: enter only when downtrend exhaustion is followed
by a reclaim and forced-cover confirmation, then exit quickly because the edge
is expected to come from a short-cover burst.

## 8. Regime Fit and Risk Scenarios

### When The Strategy Should Be Strong

The strategy should be strongest when:

- BTCUSDT is still in a downtrend, but downside momentum is fading
- price stops making short-term new lows
- price reclaims a short-term high
- taker-buy aggression rises
- CVD slope turns positive
- volume expands, but volatility is not extreme
- the move is driven by short-cover pressure rather than a slow trend-following
  long build-up

This is why the strategy uses a short 6-bar / 30-minute holding period. The
expected edge is a burst, not a long campaign.

### When The Strategy Is Vulnerable

The strategy is vulnerable when:

- the reclaim is a false breakout and price quickly falls back below the
  rolling high
- the market is in a true crash regime where short-cover rebounds fail quickly
- volatility is extreme and the fixed stop cannot be filled near the assumed
  price
- liquidity thins and slippage is larger than the base cost assumption
- the signal is triggered by short-term noise rather than real forced-cover flow
- spot-only data misclassifies the move because open interest is unavailable

### Operational Risk Scenarios

Partial fill or residual position risk:

- A live order may be partially filled.
- If the exit order only closes part of the position, a leftover long position
  can remain.
- The current research backtest assumes full fills and does not model residual
  positions.
- The final strategy avoids scale-in and half-exit logic for now, which reduces
  this risk, but live execution still needs position reconciliation.

Live design requirement:

- The live system must not assume that submitted quantity equals filled
  quantity.
- Entry, stop, and time-exit orders must be sized from confirmed filled
  position size only.
- If an exit order partially fills, the remaining position must be detected and
  flattened with a reduce-only order where supported.
- If order state is unknown, the system should stop opening new entries, cancel
  open orders, reconcile with the exchange position, and only resume after the
  position state is clean.
- One symbol should have only one active strategy position at a time.
- Client order IDs should be idempotent so retry logic does not accidentally
  create duplicate exposure.

Minimum live state to track:

- intended position size
- submitted order quantity
- confirmed filled quantity
- remaining unfilled quantity
- local strategy position
- exchange-reported position
- open reduce-only exit orders
- last reconciliation timestamp

Research decision:

Partial fill and residual position handling is not added to the current
backtest because the dataset is bar-level and does not contain order book queue
or exchange execution events. It is therefore treated as a mandatory live
execution design item, not as an alpha condition.

Stop execution risk:

- hard_stop_6b assumes the stop can be executed when the bar low reaches the
  stop level.
- In live trading, a stop-market order can slip, especially during a fast selloff.
- The hard stop is a risk-control idea, not a guaranteed loss cap.

API and order-state risk:

- The exchange may reject, delay, or partially execute an entry or exit order.
- Network/API failures can leave the system unaware of the true position.
- The current research code does not include order idempotency, retry logic,
  position reconciliation, or a kill switch.

Loss clustering risk:

- Multiple false reclaims can occur during a strong downtrend.
- The current research backtest evaluates trade-level results, but a live system
  should also enforce daily loss limits, maximum consecutive losses, and a
  cooldown after stop-outs.

Model risk:

- The current signal uses spot taker-buy and CVD proxies.
- Without futures open interest, the strategy cannot directly confirm that price
  up plus OI down is occurring.
- This means the short-cover interpretation is plausible, but not fully proven.

### Current Mitigation Status

Already addressed in the research design:

- no averaging down
- no scale-in in the final candidate
- no half-exit in the final candidate
- short 30-minute time stop
- optional hard_stop_6b risk-control variant
- extreme volatility excluded by the entry condition

Not yet addressed and required before live trading:

- partial fill handling
- residual position detection and forced flattening
- reduce-only exit order handling
- live stop-market / stop-limit behavior testing
- slippage stress test beyond the base cost assumption
- order retry and idempotency logic
- exchange position reconciliation
- max daily loss limit
- max consecutive loss cooldown
- emergency kill switch

## 9. Next Validation Step

The next step is not to add more rules immediately. The priority is to validate
the same entry and exit structure on more out-of-sample data.

If futures data becomes available, open interest should be added as confirmation:

- price up
- open interest down
- CVD up
- taker-buy ratio up

That would make the short-cover interpretation stronger than the current
spot-only proxy.
