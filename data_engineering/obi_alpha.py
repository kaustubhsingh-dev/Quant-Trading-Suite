"""obi_alpha.py — OBI vs forward return IC matrix, streaming-safe.

Hybrid timeline:
  - State matching (trade → book) uses recv_time_ns to avoid lookahead bias.
  - Forward return measurement uses event_time to keep the horizon independent.

Usage:
    python obi_alpha.py --split all
    python obi_alpha.py --split train --hourly
    python obi_alpha.py --split test  --hourly
"""

import argparse
import polars as pl
import numpy as np
import tables as tb
from pathlib import Path
from scipy.stats import spearmanr

# ─── Config ──────────────────────────────────────────────────────────────────

HORIZONS_NS = [100_000_000, 500_000_000, 1_000_000_000,
               2_000_000_000, 5_000_000_000]

DEPTHS = [1, 3, 5, 10, 20]
IC_SIGNALS = [f"obi_{n}" for n in DEPTHS] + ["obi_diff"]

TRADE_COLS = ["agg_trade_id", "event_time", "recv_time_ns",
              "price", "quantity", "is_buyer_mm"]

DEPTH_COLS = ["recv_time_ns", "event_time", "msg_seq",
              "side", "level", "price", "quantity"]


def parse_args():
    p = argparse.ArgumentParser(description="OBI forward-return IC matrix.")
    p.add_argument("--data-dir", default="data/btcusdt/aggtrade")
    p.add_argument("--split", choices=["all", "train", "test"], default="all",
                   help="Chronological split: first half = train, second = test")
    p.add_argument("--split-pct", type=float, default=0.5,
                   help="Fraction of files used for train (default 0.5)")
    p.add_argument("--best-signal", type=str, default="obi_diff",
                   help="Feature for decile analysis")
    p.add_argument("--best-horizon-ns", type=int, default=500_000_000,
                   help="Horizon for decile and hourly analysis")
    p.add_argument("--hourly", action="store_true",
                   help="Print per-hour IC breakdown for best-signal/horizon")
    return p.parse_args()


def iter_files(args):
    path = Path(args.data_dir)
    files = sorted(path.glob("*.h5"))
    if not files:
        raise SystemExit(f"No .h5 files in {path.resolve()}")

    n = len(files)
    split_idx = int(n * args.split_pct)

    if args.split == "train":
        files = files[:split_idx]
        print(f"[SPLIT] TRAIN — {len(files)}/{n} files")
    elif args.split == "test":
        files = files[split_idx:]
        print(f"[SPLIT] TEST  — {len(files)}/{n} files")
    else:
        print(f"[SPLIT] ALL   — {len(files)} files")
    return files


# ─── Streaming Loads ────────────────────────────────────────────────────────

def load_trades(files):
    parts = []
    for f in files:
        try:
            with tb.open_file(str(f), mode="r") as h:
                if len(h.root.trades) == 0:
                    continue
                cols = {c: h.root.trades.col(c)[:] for c in TRADE_COLS}
                t = pl.DataFrame(cols)
                parts.append(t)
                print(f"  {f.name}: {len(t):>8,} trades")
        except (OSError, tb.HDF5ExtError) as e:
            print(f"  {f.name}: SKIPPED ({type(e).__name__})")

    if not parts:
        return pl.DataFrame()

    return (
        pl.concat(parts)
        .unique(subset=["agg_trade_id"], keep="first")
        .sort("recv_time_ns")
    )


def load_book(files):
    parts = []
    for f in files:
        try:
            with tb.open_file(str(f), mode="r") as h:
                if len(h.root.depth) == 0:
                    continue
                cols = {c: h.root.depth.col(c)[:] for c in DEPTH_COLS}
                d = pl.DataFrame(cols)
        except (OSError, tb.HDF5ExtError) as e:
            print(f"  {f.name}: SKIPPED ({type(e).__name__})")
            continue

        summary = (
            d
            .group_by(["recv_time_ns", "msg_seq"])
            .agg([
                pl.first("event_time").alias("event_time"),
                pl.col("price").filter(pl.col("side") == 0).max().alias("best_bid"),
                pl.col("price").filter(pl.col("side") == 1).min().alias("best_ask"),
                *[
                    pl.col("quantity")
                      .filter((pl.col("side") == 0) & (pl.col("level") < n))
                      .sum().alias(f"bid_{n}")
                    for n in DEPTHS
                ],
                *[
                    pl.col("quantity")
                      .filter((pl.col("side") == 1) & (pl.col("level") < n))
                      .sum().alias(f"ask_{n}")
                    for n in DEPTHS
                ],
            ])
        )
        parts.append(summary)
        print(f"  {f.name}: {len(d):>10,} rows → {len(summary):>7,} msgs")
        del d

    if not parts:
        return pl.DataFrame()

    book = pl.concat(parts)
    book = book.filter((pl.col("bid_1") > 0) & (pl.col("ask_1") > 0))

    book = (
        book
        .with_columns([
            ((pl.col(f"bid_{n}") - pl.col(f"ask_{n}")) /
             (pl.col(f"bid_{n}") + pl.col(f"ask_{n}"))).alias(f"obi_{n}")
            for n in DEPTHS
        ])
        .with_columns(
            ((pl.col("best_ask") * pl.col("bid_1") +
              pl.col("best_bid") * pl.col("ask_1"))
             / (pl.col("bid_1") + pl.col("ask_1"))).alias("micro_price")
        )
        .with_columns(
            (pl.col("obi_3") - pl.col("obi_1")).alias("obi_diff")
        )
        .drop_nulls(subset=IC_SIGNALS + ["micro_price"])
        .unique(subset=["recv_time_ns", "msg_seq"], keep="first")
        .sort("recv_time_ns")
    )
    return book


# ─── Analysis ───────────────────────────────────────────────────────────────

def _forward_return_df(merged: pl.DataFrame, book: pl.DataFrame, h_ns: int) -> pl.DataFrame:
    h_ms = h_ns // 1_000_000

    fwd = (
        book
        .select([
            (pl.col("event_time") - h_ms).alias("event_time"),
            pl.col("micro_price").alias("fwd_micro"),
            pl.col("event_time").alias("snap_time"),
        ])
        .sort("event_time")
    )

    return (
        merged
        .join_asof(fwd, on="event_time", strategy="forward")
        .drop_nulls(subset=["fwd_micro"])
        .with_columns(
            (pl.col("snap_time") - pl.col("event_time")).alias("actual_h_ms")
        )
        .filter((pl.col("actual_h_ms") - h_ms).abs() < 50)
        .with_columns(
            ((pl.col("fwd_micro") - pl.col("micro_price"))
             / pl.col("micro_price") * 10_000.0).alias("ret_bps")
        )
    )


def compute_ic_matrix(merged: pl.DataFrame, book: pl.DataFrame):
    print("\n=== IC Matrix (Spearman, joins on Binance event_time) ===")

    col_w = max(max(len(c) for c in IC_SIGNALS), 8)

    header = f"{'horizon':>9} | " + " | ".join(f"{c:>{col_w}}" for c in IC_SIGNALS)
    print(header)
    print("-" * len(header))

    for h in HORIZONS_NS:
        a = _forward_return_df(merged, book, h)
        row_cells = []
        n_samples = len(a)
        for sig in IC_SIGNALS:
            if n_samples < 100:
                row_cells.append(f"{'n/a':>{col_w}}")
                continue
            ic, _ = spearmanr(a[sig].to_numpy(), a["ret_bps"].to_numpy())
            row_cells.append(f"{ic:>{col_w}.4f}")
        print(f"{h/1e9:>8.2f}s | " + " | ".join(row_cells)
              + f"   (n={n_samples:,})")


def decile_analysis(merged: pl.DataFrame, book: pl.DataFrame,
                    signal_name: str, horizon_ns: int):
    print(f"\n=== Decile returns — {signal_name}, h={horizon_ns/1e9:.2f}s ===")

    a = (
        _forward_return_df(merged, book, horizon_ns)
        .with_columns(
            pl.col(signal_name)
              .qcut(10, labels=[f"D{i:02d}" for i in range(1, 11)],
                    allow_duplicates=True)
              .alias("decile")
        )
    )

    return (
        a.group_by("decile")
         .agg([
             pl.col(signal_name).mean().alias(f"avg_{signal_name}"),
             pl.col("ret_bps").mean().alias("mean_ret_bps"),
             pl.col("ret_bps").std().alias("std_ret_bps"),
             pl.len().alias("n"),
         ])
         .sort("decile")
    )


def hourly_ic_breakdown(merged: pl.DataFrame, book: pl.DataFrame,
                        signal_name: str, horizon_ns: int):
    """Compute IC chronologically for every hour of the dataset."""
    a = _forward_return_df(merged, book, horizon_ns)

    # 1. Truncate to native Datetime — DO NOT convert to string here
    a = a.with_columns(
        pl.from_epoch(pl.col("event_time"), time_unit="ms")
          .dt.truncate("1h")
          .alias("hour_ts")
    )

    print(f"\n=== Hourly IC — {signal_name}, h={horizon_ns/1e9:.2f}s ===")
    print(f"{'hour_utc':>16} | {'IC':>9} | {'n':>9} | {'bar':<40}")
    print("-" * 85)

    rows = []
    # 2. Native temporal sort: works across days, months, and year rollovers
    for dt in sorted(a["hour_ts"].unique().to_list()):
        sub = a.filter(pl.col("hour_ts") == dt)
        n = len(sub)
        if n < 100:
            rows.append((dt, None, n))
            continue
        ic, _ = spearmanr(sub[signal_name].to_numpy(), sub["ret_bps"].to_numpy())
        rows.append((dt, ic, n))

    max_abs = max((abs(r[1]) for r in rows if r[1] is not None), default=1.0)
    
    # 3. Format as ISO string only when rendering to stdout
    for dt, ic, n in rows:
        dt_str = dt.strftime("%Y-%m-%d %H:00")
        if ic is None:
            print(f"{dt_str:>16} | {'n/a':>9} | {n:>9} | (insufficient data)")
            continue
        width = int(round(abs(ic) / max_abs * 40))
        bar_char = "█" if ic > 0 else "▒"
        bar = bar_char * width
        print(f"{dt_str:>16} | {ic:>9.4f} | {n:>9,} | {bar}")


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    files = iter_files(args)

    print("\nTrades:")
    df_trades = load_trades(files)

    print("\nBook (per-file reduction, grouped by recv_time_ns + msg_seq):")
    book = load_book(files)

    print(f"\nTrades loaded: {len(df_trades):,}")
    print(f"Book msgs:     {len(book):,}")

    if df_trades.is_empty() or book.is_empty():
        raise SystemExit("Not enough data to analyze.")

    merged = (
        df_trades
        .join_asof(book, on="recv_time_ns", strategy="backward")
        .drop_nulls(subset=["micro_price"])
        .sort("event_time")
    )
    print(f"Merged:        {len(merged):,} trades with book state")

    compute_ic_matrix(merged, book)

    deciles = decile_analysis(merged, book, args.best_signal, args.best_horizon_ns)
    print(deciles)

    if args.hourly:
        hourly_ic_breakdown(merged, book, args.best_signal, args.best_horizon_ns)


if __name__ == "__main__":
    main()