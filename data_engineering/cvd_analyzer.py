"""
cvd_analyzer.py — Institutional Absorption Classifier

Tests whether aggressive order flow (CVD) at structural touches improves
the baseline bounce rate. Reports:

  1. Baseline bounce rate on the full dataset (no filter)
  2. Bounce rate per CVD quintile (all touches, no trend filter)
  3. Same, restricted to trend-aligned subset
  4. Bootstrap-free 95% confidence intervals on each rate
  5. Train/test split by time (first 9 months / last 3 months)

If CVD carries information, the top and bottom quintiles should differ
by more than their confidence intervals, and the ordering should hold
out-of-sample.
"""

import math
import polars as pl
import numpy as np

CVD_FILE = "data_store/touches_with_cvd.parquet"

TRAIN_END = "2024-06-30"   # first ~9 months train, rest test


def add_delta_pct(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.when(pl.col("vol_pre_5m") > 0)
          .then(pl.col("cvd_pre_5m") / pl.col("vol_pre_5m"))
          .otherwise(0.0)
          .alias("delta_pct_pre_5m")
    )


def ci95(p: float, n: int) -> tuple[float, float]:
    """Wald 95% CI for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    se = math.sqrt(p * (1 - p) / n)
    return (max(0.0, p - 1.96 * se), min(1.0, p + 1.96 * se))


def report_quintiles(df: pl.DataFrame, title: str):
    print(f"\n{title}")
    print(f"  Total decisive touches: {len(df):,}")
    if len(df) == 0:
        print("  (empty)")
        return

    baseline = df["outcome"].eq("bounce").mean()
    lo, hi = ci95(baseline, len(df))
    print(f"  Baseline bounce rate:   {baseline:.2%}  "
          f"[{lo:.2%}, {hi:.2%}]")
    print()

    if len(df) < 50:
        print("  (too few samples for quintile breakdown)")
        return

    # Quintile by delta_pct_pre_5m
    df = df.with_columns(
        pl.col("delta_pct_pre_5m")
          .qcut(5, labels=["Q1 ExtremeSell", "Q2 HeavySell", "Q3 Neutral",
                           "Q4 HeavyBuy", "Q5 ExtremeBuy"],
                allow_duplicates=True)
          .alias("cvd_quintile")
    )

    rows = []
    for q in ["Q1 ExtremeSell", "Q2 HeavySell", "Q3 Neutral",
              "Q4 HeavyBuy", "Q5 ExtremeBuy"]:
        sub = df.filter(pl.col("cvd_quintile") == q)
        n = len(sub)
        if n == 0:
            continue
        bounces = sub["outcome"].eq("bounce").sum()
        p = bounces / n
        lo, hi = ci95(p, n)
        avg = sub["delta_pct_pre_5m"].mean()
        rows.append((q, n, avg, p, lo, hi))

    print(f"  {'quintile':<16}  {'n':>6}  {'avg_delta':>10}  "
          f"{'win_rate':>9}  {'95% CI':>18}")
    for q, n, avg, p, lo, hi in rows:
        print(f"  {q:<16}  {n:>6}  {avg:>10.3f}  "
              f"{p:>8.2%}  [{lo:>6.2%}, {hi:>6.2%}]")

    # Spread check: does the extreme top differ from extreme bottom?
    if len(rows) >= 2:
        q1 = rows[0]
        q5 = rows[-1]
        spread = q5[3] - q1[3]
        # Rough significance: do the CIs overlap?
        overlap = not (q1[5] < q5[4] or q5[5] < q1[4])
        print()
        print(f"  Extreme Buy − Extreme Sell spread: {spread:+.2%}")
        print(f"  CIs overlap: {'YES (not significant)' if overlap else 'NO (signal present)'}")


def analyze():
    df = pl.read_parquet(CVD_FILE)
    df = df.filter(pl.col("outcome").is_in(["bounce", "break"]))
    df = add_delta_pct(df)

    # Add a date column for train/test split
    df = df.with_columns(
        pl.from_epoch("touch_time", time_unit="ms")
          .dt.strftime("%Y-%m-%d")
          .alias("touch_date")
    )

    train = df.filter(pl.col("touch_date") <= TRAIN_END)
    test  = df.filter(pl.col("touch_date") >  TRAIN_END)

    print("=" * 78)
    print("CVD ABSORPTION ANALYSIS — STRUCTURAL TOUCHES")
    print("=" * 78)
    print(f"Total decisive touches: {len(df):,}")
    print(f"  Train (≤{TRAIN_END}): {len(train):,}")
    print(f"  Test  (> {TRAIN_END}): {len(test):,}")

    # ─── 1. All touches, no trend filter ─────────────────────────────────
    print("\n" + "=" * 78)
    print("ANALYSIS A: ALL TOUCHES (no trend filter, no side filter)")
    print("=" * 78)
    report_quintiles(df, "Quintile breakdown")

    # ─── 2. Split by side ────────────────────────────────────────────────
    for side, label in [("low", "SUPPORT"), ("high", "RESISTANCE")]:
        sub = df.filter(pl.col("side") == side)
        print("\n" + "=" * 78)
        print(f"ANALYSIS B: {label} ONLY (all trends)")
        print("=" * 78)
        report_quintiles(sub, f"{label} quintile breakdown")

    # ─── 3. Trend-aligned subsets ────────────────────────────────────────
    sup_aligned = df.filter(
        (pl.col("side") == "low") &
        (pl.col("trend_bucket").is_in(["up", "flat"]))
    )
    res_aligned = df.filter(
        (pl.col("side") == "high") &
        (pl.col("trend_bucket").is_in(["down", "flat"]))
    )

    print("\n" + "=" * 78)
    print("ANALYSIS C: TREND-ALIGNED SUBSETS")
    print("=" * 78)
    report_quintiles(sup_aligned, "Support aligned (up/flat trend)")
    report_quintiles(res_aligned, "Resistance aligned (down/flat trend)")

    # ─── 4. Out-of-sample stability check ────────────────────────────────
    print("\n" + "=" * 78)
    print("ANALYSIS D: TRAIN / TEST STABILITY (all touches)")
    print("=" * 78)

    # Fit quintile cutoffs on train, apply to test
    if len(train) >= 100 and len(test) >= 100:
        cuts = train["delta_pct_pre_5m"].quantile([0.2, 0.4, 0.6, 0.8])
        cut_values = [float(cuts[i]) for i in range(4)]

        def bucket_of(x: float) -> int:
            if x < cut_values[0]:   return 1
            if x < cut_values[1]:   return 2
            if x < cut_values[2]:   return 3
            if x < cut_values[3]:   return 4
            return 5

        for name, subset in [("TRAIN", train), ("TEST", test)]:
            arr = subset.select(["delta_pct_pre_5m"]).to_series().to_list()
            outcomes = subset.select(["outcome"]).to_series().to_list()
            buckets = {i: [0, 0] for i in range(1, 6)}   # [total, bounces]
            for x, o in zip(arr, outcomes):
                b = bucket_of(x)
                buckets[b][0] += 1
                if o == "bounce":
                    buckets[b][1] += 1

            print(f"\n  {name} (n={len(subset):,}):")
            print(f"  {'bucket':<8}  {'n':>6}  {'bounce_rate':>12}")
            for b in range(1, 6):
                n, bo = buckets[b]
                rate = (bo / n) if n > 0 else 0.0
                print(f"  Q{b:<7}  {n:>6}  {rate:>11.2%}")

        print("\n  If Q5−Q1 spread shrinks in TEST vs TRAIN, the signal is unstable.")
    else:
        print("  (Not enough data for train/test split)")


if __name__ == "__main__":
    analyze()