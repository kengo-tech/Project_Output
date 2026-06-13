# 取引頻度とアルゴ実装メモ

## 1. 目的

このメモは、現在の主候補である
`downtrend_exhaustion_reclaim_signal` について、取引回数が少ないことを
どう解釈するか、また今回の仮説がどのようにコードへ落とし込まれているかを
READMEとは別に整理したものです。

中心となる問いは次の通りです。

> 取引回数が少ないのは、機会損失なのか。それとも、仮説に沿って質の高い局面だけを選別できているのか。

現時点の結論は、次の通りです。

取引回数が少ないこと自体は、今回の研究段階では問題ではありません。今回の仮説は
かなり限定されたショートカバー局面を狙うものなので、頻度が低くなるのは自然です。
ただし、サンプル数が少ないため、統計的な信頼性はまだ十分ではありません。

したがって、この戦略は現時点では「完成した本番戦略」ではなく、
「仮説に沿って実装された有望なプロトタイプ」として扱うのが妥当です。

## 2. 現在の主候補

主エントリー:

```text
downtrend_exhaustion_reclaim_signal
```

主出口:

```text
fixed_6b
```

意味:

- 下落トレンド中である
- ただし、価格が短期安値を更新できなくなっている
- 短期高値を上抜ける
- taker-buy、CVD、volume expansion によって買い戻しらしさを確認する
- 5分足で6本、つまり約30分だけ保有する

保守的な比較対象:

```text
hard_stop_6b
```

ただし、現時点の主出口は `fixed_6b` のままです。

## 3. 取引頻度

現在の主候補を、90日分のBTCUSDT 5分足データで確認した結果は以下です。

| 区分 | 期間 | シグナル発火数 | 実取引数 | 頻度 |
|---|---:|---:|---:|---:|
| In-sample | 約59.0日 | 14回 | 13回 | 約4.5日に1回 |
| Out-of-sample | 約30.7日 | 7回 | 6回 | 約5.1日に1回 |
| 合計 | 約89.7日 | 21回 | 19回 | 約4.7日に1回 |

月換算:

```text
約6.3回 / 30日
```

数字の意味:

- `21回` は、条件を満たしたシグナルバーの数
- `20回` は、連続シグナルを1つの機会としてまとめた場合の取引機会数
- `19回` は、保有中の重複シグナルを除いた実際のバックテスト取引数

取引にならなかったシグナルは2回です。

- 1回は、前の30分トレードを保有中だったためスキップ
- 1回は、直前バーからの連続シグナルであり、同一機会として扱うのが自然

## 4. 取引回数が少ないことの評価

取引回数が少ないことは、必ずしも悪いことではありません。

今回の仮説は、次のようなかなり限定された局面を狙っています。

- まだ下落トレンド中である
- 売りが出ている
- しかし価格が新安値を更新できない
- 短期高値を上抜ける
- CVDが改善する
- taker-buy が増える
- volume が拡大する

これは高頻度に何度も出るシグナルではなく、下落局面の中で一部のショート勢が
買い戻し始める瞬間を狙う戦略です。

そのため、頻度が低いことは仮説と整合しています。

良い面:

- ノイズトレードを減らしている
- 仮説に忠実な局面だけを選別している
- 初回実装として検証しやすい
- ナンピン、買い増し、半分利確を使わないため、ライブ実装リスクが小さい
- 保有時間が短く、ショートカバーの「火花」を取りに行く設計と合っている

弱い面:

- サンプル数が少ない
- 統計的な信頼度がまだ低い
- 数回のトレードに結果が左右されやすい
- 条件が厳しすぎて機会損失している可能性は残る
- ライブ運用では待つ時間が長くなり、ルール外トレードをしたくなる心理的リスクがある

## 5. 今回のアルゴ実装としての評価

初めて作るアルゴとしては、現在の構造はかなり良いです。

理由は、以下の要素が分離できているからです。

- entry: どの局面で入るか
- exit: どのくらい保有するか
- risk-control exit: 損失をどこで切るか
- execution cost: slippageやspread stressに耐えられるか
- portfolio risk: 日次損失や連敗時cooldownをどう扱うか
- live risk: partial fillやresidual positionをどう扱うか

最初から取引回数を増やすことを目的にすると、戦略の良し悪しなのか、
実装ミスなのか、執行コストなのかが分かりにくくなります。

その意味で、今回の実装は「狭い仮説を、まず安全に検証する」設計になっています。

## 6. エントリー条件の実装コード

`src/features.py` では、主候補のシグナルは次のように実装されています。

```python
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
)
```

自然言語に直すと、次のようになります。

```text
if downtrend
and price is no longer making a fresh short-term low
and close reclaims a short-term high
and forced-cover score is strong
and taker-buy aggression is high
and CVD slope turns positive
and volume expands
and volatility is high but not extreme:
    enter long at next bar open
```

## 7. 30分固定出口の実装コード

`main.py` のバックテストでは、シグナルの次の足のopenで入り、6本後のcloseで出ます。

```python
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

    next_available_idx = exit_idx + 1
```

この実装上の重要点:

- 同じ足ではなく、次の足のopenで入る
- 6本、つまり約30分で固定退出する
- 保有中のシグナルは新規トレードにしない
- コストは固定コストまたは動的slippage / spread stressで評価できる

## 8. 取引回数を再現するコード

以下のコードで、現在の主候補について、シグナル発火数と実取引数を再計算できます。

```python
import pandas as pd

from src.features import make_features
from main import run_fast_fixed_horizon_backtest


feature_path = "data/processed/btcusdt_5m_features_90d_fast_refined_signals.csv"
base_df = pd.read_csv(
    feature_path,
    parse_dates=["open_time", "close_time"],
)

sample_split = base_df["sample_split"].copy()
feature_df = make_features(base_df)
feature_df["sample_split"] = sample_split

signal_col = "downtrend_exhaustion_reclaim_signal"
holding_bars = 6
fee_rate = 0.0004
slippage_rate = 0.0002

rows = []

for sample_label in ["in_sample", "out_of_sample"]:
    sample_df = (
        feature_df
        .loc[feature_df["sample_split"] == sample_label]
        .reset_index(drop=True)
    )

    signal_series = sample_df[signal_col].fillna(False).astype(bool)
    signal_bars = int(signal_series.sum())

    signal_clusters = int(
        (signal_series & ~signal_series.shift(1, fill_value=False)).sum()
    )

    trade_df = run_fast_fixed_horizon_backtest(
        df=sample_df,
        signal_col=signal_col,
        holding_bars=holding_bars,
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
        cost_model="fixed",
    )

    start_time = pd.to_datetime(sample_df["open_time"].iloc[0])
    end_time = pd.to_datetime(sample_df["open_time"].iloc[-1])
    sample_days = (end_time - start_time).total_seconds() / 86400 + 5 / 1440

    rows.append(
        {
            "sample_split": sample_label,
            "sample_days": sample_days,
            "signal_bars": signal_bars,
            "signal_clusters": signal_clusters,
            "executed_trades": len(trade_df),
            "trades_per_30d": len(trade_df) / sample_days * 30,
            "days_per_trade": sample_days / len(trade_df),
        }
    )

summary_df = pd.DataFrame(rows)
print(summary_df)
```

期待される結果:

```text
in_sample:
  signal_bars = 14
  signal_clusters = 14
  executed_trades = 13

out_of_sample:
  signal_bars = 7
  signal_clusters = 6
  executed_trades = 6

total:
  signal_bars = 21
  signal_clusters = 20
  executed_trades = 19
```

## 9. 現時点の結論

今回の戦略は、取引回数を最大化するための戦略ではありません。

狙っているのは、下落トレンド中に売りの力が弱まり、短期高値の上抜けと
買い成行の増加によってショートカバーの可能性が出る局面です。

したがって、現時点の最もフェアな結論は次の通りです。

```text
取引回数が少ないのは、仮説に忠実な結果である。
ただし、サンプル数が少ないため、完成版ではなく有望な初期候補である。
```

次にやるべきことは、単純に取引回数を増やすことではありません。

優先順位は次の通りです。

1. 現在の主候補を固定したまま、より長い期間で検証する
2. `fixed_6b` と `hard_stop_6b` の比較を維持する
3. 少し条件を緩めた診断候補と比較する
4. 期待値が残る範囲で取引機会を増やせるかを見る
5. futures open interest を追加し、short-cover 解釈を補強する

この順番なら、初回アルゴとしての安全性と、研究としての説得力の両方を保てます。
