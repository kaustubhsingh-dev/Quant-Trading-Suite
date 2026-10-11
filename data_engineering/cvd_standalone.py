"""
cvd_standalone.py — Does CVD predict forward returns, without any structural filter?

Uses 1m klines: taker_buy_volume gives us per-bar CVD directly.
No levels. No excursion. No side. Pure signal → future return.

Reports:
  1. IC matrix (Spearman) — rolling CVD vs forward returns
  2. Top/bottom quintile forward-return spread
  3. Same split by trend bucket (up/down/flat) to catch trend contamination
"""

import glob
import numpy as np
import polars as pl
from scipy.stats import spearmanr


DATA_GLOB = "data_store/klines/BTCUSDT_1m/*.parquet"

LOOKBACKS = [5, 15, 60, 240]     # minutes
HORIZONS  = [5, 15, 60, 240]     # minutes
TREND_LOOKBACK_BARS = 1440       # 24h for trend classification
TREND_THRESH_BPS    = 100.0      # ±1% separates up/down from flat


def load_data() -> pl.DataFrame:
    files = sorted(glob.glob(DATA_GLOB))
    if not files:
        raise SystemExit(f"No files matched {DATA_GLOB}")
    print(f"Loading {len(files)} files...")
    df = pl.concat([pl.read_parquet(f) for f in files]).sort("open_time")

    # Per-bar delta from kline fields
    # volume = taker_buy + taker_sell
    # delta  = taker_buy - taker_sell = 2*taker_buy - volume
    df = df.with_columns(
        (2 * pl.col("taker_buy_volume") - pl.col("volume")).alias("delta")
    )
    print(f"Loaded {len(df):,} bars | Range: {df['time'].min()} → {df['time'].max()}")
    return df


def add_features(df: pl.DataFrame) -> pl.DataFrame:
    # Rolling CVD ratio: cvd / total_volume over the window
    # Scale-invariant, bounded [-1, +1]
    for lb in LOOKBACKS:
        df = df.with_columns([
            pl.col("delta").rolling_sum(window_size=lb).alias(f"cvd_{lb}"),
            pl.col("volume").rolling_sum(window_size=lb).alias(f"vol_{lb}"),
        ])
        df = df.with_columns(
            (pl.col(f"cvd_{lb}") / pl.col(f"vol_{lb}")).alias(f"cvd_ratio_{lb}")
        )

    # Forward returns
    for h in HORIZONS:
        df = df.with_columns(
            ((pl.col("close").shift(-h) / pl.col("close")) - 1.0).alias(f"fwd_{h}")
        )

    # Trend bucket at each bar (trailing 24h)
    df = df.with_columns(
        ((pl.col("close") / pl.col("close").shift(TREND_LOOKBACK_BARS)) - 1.0)
        .alias("trailing_ret")
    )
    df = df.with_columns(
        pl.when(pl.col("trailing_ret") >  TREND_THRESH_BPS / 1e4).then(pl.lit("up"))
          .when(pl.col("trailing_ret") < -TREND_THRESH_BPS / 1e4).then(pl.lit("down"))
          .otherwise(pl.lit("flat"))
          .alias("trend_bucket")
    )
    return df


def ic_matrix(df: pl.DataFrame, subset_name: str):
    print(f"\n{'=' * 72}")
    print(f"IC MATRIX — {subset_name}")
    print(f"{'=' * 72}")
    print(f"{'lookback':>10} | " + " | ".join(f"{h:>5}m" for h in HORIZONS))
    print("-" * 72)

    for lb in LOOKBACKS:
        row = []
        for h in HORIZONS:
            sub = df.drop_nulls(subset=[f"cvd_ratio_{lb}", f"fwd_{h}"])
            if len(sub) < 1000:
                row.append("   n/a")
                continue
            ic, _ = spearmanr(
                sub[f"cvd_ratio_{lb}"].to_numpy(),
                sub[f"fwd_{h}"].to_numpy() * 1e4,  # bps
            )
            row.append(f"{ic:>6.4f}")
        print(f"{lb:>8}m | " + " | ".join(row))


def quintile_spread(df: pl.DataFrame, lookback: int, horizon: int):
    sub = df.drop_nulls(subset=[f"cvd_ratio_{lookback}", f"fwd_{horizon}"]).with_columns(
        ((pl.col(f"fwd_{horizon}") * 1e4).round(2)).alias("ret_bps")
    )
    sub = sub.with_columns(
        pl.col(f"cvd_ratio_{lookback}")
          .qcut(5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"], allow_duplicates=True)
          .alias("q")
    )
    stats = (
        sub.group_by("q")
           .agg([
               pl.col("ret_bps").mean().alias("mean_ret_bps"),
               pl.col("ret_bps").std().alias("std"),
               pl.len().alias("n"),
           ])
           .sort("q")
    )
    spread = stats.filter(pl.col("q") == "Q5")["mean_ret_bps"][0] - \
             stats.filter(pl.col("q") == "Q1")["mean_ret_bps"][0]
    return stats, spread


def main():
    df = load_data()
    df = add_features(df)

    # Full dataset
    ic_matrix(df, "ALL DATA")

    # Trend-bucketed
    for bucket in ["up", "down", "flat"]:
        sub = df.filter(pl.col("trend_bucket") == bucket)
        if len(sub) < 10000:
            continue
        ic_matrix(sub, f"TREND = {bucket.upper()}")

    # Quintile spread at a representative (lookback, horizon)
    print(f"\n{'=' * 72}")
    print("QUINTILE SPREAD — lookback=15m, horizon=60m")
    print(f"{'=' * 72}")
    stats, spread = quintile_spread(df, 15, 60)
    print(stats)
    print(f"\nQ5 − Q1 spread: {spread:+.2f} bps")

    # Also check the train/test side
    print(f"\n{'=' * 72}")
    print("QUINTILE SPREAD — lookback=15m, horizon=60m  (test half only)")
    print(f"{'=' * 72}")
    n = len(df)
    test = df.tail(n // 3)   # last ~4 months
    stats_test, spread_test = quintile_spread(test, 15, 60)
    print(stats_test)
    print(f"\nQ5 − Q1 spread (test): {spread_test:+.2f} bps")


if __name__ == "__main__":
    main()