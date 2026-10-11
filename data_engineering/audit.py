"""audit.py — Stream-based trade continuity + depth cadence check. (Bulletproof Edition)"""

import argparse
import polars as pl
import tables as tb
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

NS_PER_MS = 1_000_000

def iso_ns(ns):
    return datetime.fromtimestamp(int(ns) / 1e9, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

def iso_ms(ms):
    return datetime.fromtimestamp(int(ms) / 1e3, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def parse_args():
    parser = argparse.ArgumentParser(description="Audit HDF5 market data for continuity and cadence.")
    parser.add_argument("--data-dir", type=str, default="data/btcusdt/aggtrade", help="Path to HDF5 directory")
    parser.add_argument("--stutter-ms", type=float, default=250.0, help="Threshold for minor stutter (ms)")
    return parser.parse_args()


# ─── Stage 1: Columnar Extraction & Global Deduplication ─────────────────────

def iter_files(data_dir):
    path = Path(data_dir)
    files = sorted(path.glob("*.h5"))
    if not files:
        raise SystemExit(f"No .h5 files found in {path.resolve()}")
    return files


def load_trade_ids(files):
    """Loads specific columns directly from HDF5 disk to prevent OOM crashes."""
    parts = []
    for f in files:
        try:
            with tb.open_file(str(f), mode="r") as h:
                if len(h.root.trades) == 0:
                    continue
                # TRUE COLUMNAR LOADING: Extract 1D arrays, bypass the full table
                ids = h.root.trades.col("agg_trade_id")
                evs = h.root.trades.col("event_time")
                
                t = pl.DataFrame({
                    "agg_trade_id": ids,
                    "event_time": evs,
                    "src_file": f.name
                })
                parts.append(t)
                print(f"  {f.name}: {len(t):>8,} trade ids")
        except (OSError, tb.HDF5ExtError) as e:
            print(f"  {f.name}: SKIPPED ({type(e).__name__}) - May be actively writing")
    
    if not parts:
        return pl.DataFrame()
        
    # GLOBAL DEDUPLICATION: Run after concatenation to catch overlaps at file boundaries
    df = pl.concat(parts)
    return df.unique(subset=["agg_trade_id"], keep="first").sort("agg_trade_id")


def load_depth_summaries(files):
    """Loads specific columns directly with a composite dedup key."""
    parts = []
    for f in files:
        try:
            with tb.open_file(str(f), mode="r") as h:
                if len(h.root.depth) == 0:
                    continue
                
                recv = h.root.depth.col("recv_time_ns")
                ev = h.root.depth.col("event_time")
                txn = h.root.depth.col("txn_time")
                seq = h.root.depth.col("msg_seq")
                
                d = pl.DataFrame({
                    "recv_time_ns": recv,
                    "event_time": ev,
                    "txn_time": txn,
                    "msg_seq": seq,
                    "src_file": f.name
                })
                
                # Composite key prevents dropping valid back-to-back packets in the same 100ns tick
                d = d.unique(subset=["recv_time_ns", "msg_seq"], keep="first")
                parts.append(d)
                print(f"  {f.name}: {len(d):>8,} depth messages")
        except (OSError, tb.HDF5ExtError) as e:
            print(f"  {f.name}: SKIPPED ({type(e).__name__})")
    
    if not parts:
        return pl.DataFrame()

    # Enforce global sequence deduplication
    df = pl.concat(parts)
    return df.unique(subset=["recv_time_ns", "msg_seq"], keep="first").sort("recv_time_ns")


# ─── Stage 2: Trade Continuity (Idiomatic Polars) ────────────────────────────

def check_trades(trades: pl.DataFrame):
    print("\n" + "=" * 125)
    print("TRADE CONTINUITY   (agg_trade_id gaps)")
    print("=" * 125)

    n = trades.height
    if n < 2:
        print("Not enough trades.")
        return

    # Use native lazy/vectorized expressions instead of numpy dumps
    trades = trades.with_columns([
        pl.col("agg_trade_id").diff().alias("id_diff"),
        pl.col("event_time").diff().alias("time_diff_ms"),
        pl.col("event_time").shift(1).alias("prev_event_time")
    ])

    # Filter strictly for gaps > 1. This prevents overlap collisions (diff == 0) from ruining math.
    gaps_df = trades.filter(pl.col("id_diff") > 1)
    
    ids = trades["agg_trade_id"]
    span = int(ids.max() - ids.min() + 1)
    coverage = 100.0 * n / span

    print(f"Trades loaded:    {n:>14,}")
    print(f"ID range:         {ids.min():>14,}  →  {ids.max():>14,}")
    print(f"Expected total:   {span:>14,}")
    print(f"Coverage:         {coverage:>13.5f}%")

    if gaps_df.height == 0:
        print("\n✅  No gaps. Perfect continuity.")
        return

    total_missing = int((gaps_df["id_diff"] - 1).sum())
    print(f"\n❌  {gaps_df.height} gaps  |  {total_missing:,} missing trades total\n")
    
    print(f"{'#':>4}  {'gap':>7}  {'missing':>10}  {'duration_ms':>12}  {'gap_window_utc':>54}  {'src_file (post-gap)':>22}")
    print("-" * 125)
    
    for i, row in enumerate(gaps_df.iter_rows(named=True)):
        if i >= 40:
            print(f"   ... and {gaps_df.height - 40} more gaps")
            break
            
        gap_size = int(row["id_diff"])
        miss = gap_size - 1
        dur_ms = int(row["time_diff_ms"])
        window = f"{iso_ms(row['prev_event_time'])} -> {iso_ms(row['event_time'])}"
        print(f"{i+1:>4}  {gap_size:>7}  {miss:>10,}  {dur_ms:>12,d}  {window:>54}  {row['src_file']:>22}")


# ─── Stage 3: Depth Cadence & Attribution ────────────────────────────────────

def check_depth(depth: pl.DataFrame, stutter_ms: float):
    print("\n" + "=" * 120)
    print("DEPTH CADENCE   (recv_time_ns vs Binance event_time)")
    print("=" * 120)

    n = depth.height
    if n < 2:
        print("Not enough messages.")
        return

    # Calculate native deltas directly in Polars
    depth = depth.with_columns([
        (pl.col("recv_time_ns").diff() / 1e6).alias("d_recv_ms"),
        pl.col("event_time").diff().alias("d_event_ms")
    ])

    d_recv = depth["d_recv_ms"].drop_nulls().to_numpy()
    d_event = depth["d_event_ms"].drop_nulls().to_numpy()

    recv_min_ns, recv_max_ns = depth["recv_time_ns"].min(), depth["recv_time_ns"].max()
    window_min = (recv_max_ns - recv_min_ns) / 1e9 / 60
    
    print(f"Messages:         {n:>14,}")
    print(f"Window:           {window_min:>13.2f} min")

    def print_dist(name, arr):
        print(f"\n{name} delta distribution (ms):")
        if len(arr) == 0: return
        percentiles = [50, 90, 99, 99.9]
        vals = np.percentile(arr, percentiles)
        print(f"  min    {arr.min():>10.2f}")
        for p, v in zip(percentiles, vals):
            print(f"  p{p:<5} {v:>10.2f}")
        print(f"  max    {arr.max():>10.2f}")

    print_dist("Local (recv)", d_recv)
    print_dist("Binance (event)", d_event)

    flagged = depth.filter(pl.col("d_recv_ms") > stutter_ms)
    total_gap_s = flagged["d_recv_ms"].sum() / 1000 if flagged.height > 0 else 0.0
    loss_pct = (100 * total_gap_s / (window_min * 60)) if window_min > 0 else 0.0

    print(f"\nStutters > {stutter_ms} ms (recv-side):  {flagged.height} events")
    print(f"Total time lost:  {total_gap_s:.2f} s  ({loss_pct:.4f}% of window)")

    if flagged.height == 0:
        print("\n✅  No stutters. Cadence is healthy.")
        return

    print(f"\n{'#':>4}  {'d_recv':>8}  {'d_event':>8}  {'lost':>6}  {'cause':>14}  {'time_utc':>26}  {'src_file':>22}")
    print("-" * 110)
    
    for i, row in enumerate(flagged.iter_rows(named=True)):
        if i >= 50:
            print(f"   ... and {flagged.height - 50} more stutters")
            break
            
        dr = row["d_recv_ms"]
        de = row["d_event_ms"]
        lost = max(0, int(round(dr / 100.0)) - 1)

        # Logic now safely handles out-of-order execution prints
        if de < 0:
            cause = "OUT_OF_ORDER"
        elif dr > stutter_ms and de <= 150:
            cause = "LOCAL_LAG"
        elif de > stutter_ms and abs(dr - de) < 60:
            cause = "EXCHANGE"
        elif de > stutter_ms and dr > de + 60:
            cause = "BOTH"
        else:
            cause = "BURST_RECOVERY"

        print(f"{i+1:>4}  {dr:>8.1f}  {de:>8.1f}  {lost:>6}  {cause:>14}  {iso_ns(row['recv_time_ns']):>26}  {row['src_file']:>22}")


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    files = iter_files(args.data_dir)
    
    print("Loading trade IDs (Columnar read, True streaming)...")
    trades = load_trade_ids(files)
    check_trades(trades)
    del trades

    print("\nLoading depth summaries (Columnar read, Composite Dedup)...")
    depth = load_depth_summaries(files)
    check_depth(depth, args.stutter_ms)
    print()