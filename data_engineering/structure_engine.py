"""
structure_engine.py — Rigorous Chronological Validation of Structural Barriers.

Tests whether clustered liquidity pools (Equal Highs / Equal Lows) act as barriers
under zero-lookahead, realistic market mechanics:
  1. Multi-pivot cluster requirement (>= 2 pivots within epsilon tolerance).
  2. Excursion requirement (price must pull away before pool is armed for retest).
  3. Strict directional approach (resistance tested from below, support from above).
  4. Outcome measurement starting at t+1 with distinct targets and ambiguity tracking.
  5. Trend bucketing: outcome rate split by trailing 24h return context.
  6. Touch export: writes resolved touches (with metadata) to parquet for CVD enrichment.

Usage:
    python structure_engine.py --data "data_store/BTCUSDT_1h.parquet"
"""

import argparse
import glob
import time
import numpy as np
import polars as pl
from collections import defaultdict


# ─── Configuration Defaults ──────────────────────────────────────────────────

DATA_GLOB         = "data_store/*_1h.parquet"
TOUCHES_OUTPUT    = "data_store/valid_touches.parquet"

K                 = 15
MIN_PIVOTS        = 2
EPSILON_BPS       = 5.0
EXCURSION_BPS     = 15.0
TOUCH_TOL_BPS     = 3.0
BOUNCE_BPS        = 25.0
BREAK_BPS         = 25.0
TIMEOUT_BARS      = 120
POOL_EXPIRY_BARS  = 1440
TREND_LOOKBACK    = 1440
TREND_THRESH_BPS  = 100.0


# ─── Data Ingestion ──────────────────────────────────────────────────────────

def load_data(data_path: str) -> pl.DataFrame:
    # Handle both comma-separated strings and standard glob patterns
    files = []
    for pattern in data_path.split(","):
        files.extend(glob.glob(pattern.strip()))
    
    files = sorted(list(set(files)))
    
    if not files:
        raise SystemExit(f"No files matched {data_path}")
    print(f"Loading {len(files)} files from {data_path}...")
    df = pl.concat([pl.read_parquet(f) for f in files]).sort("open_time")
    print(f"Loaded {len(df):,} bars | Range: {df['open_time'].min()} → {df['open_time'].max()}")
    return df


# ─── Vectorized Pivot Detection ──────────────────────────────────────────────

def detect_pivots_vectorized(highs: np.ndarray, lows: np.ndarray, k: int):
    n = len(highs)
    is_high = np.ones(n, dtype=bool)
    is_low  = np.ones(n, dtype=bool)

    for shift in range(-k, k + 1):
        if shift == 0:
            continue
        is_high &= (highs > np.roll(highs, shift))
        is_low  &= (lows < np.roll(lows, shift))

    is_high[:k]  = False
    is_high[-k:] = False
    is_low[:k]   = False
    is_low[-k:]  = False

    return np.where(is_high)[0], np.where(is_low)[0]


# ─── Pool Clustering & State Updates ─────────────────────────────────────────

def upsert_pivot(active_pools: list, level: float, side: str, t: int,
                 epsilon_bps: float, min_pivots: int):
    tol = epsilon_bps / 1e4
    for pool in active_pools:
        if pool["side"] != side or pool["consumed"]:
            continue
        if abs(level - pool["level"]) / pool["level"] <= tol:
            pool["_sum"] += level
            pool["_count"] += 1
            pool["level"] = pool["_sum"] / pool["_count"]
            pool["last_reinforced_at"] = t
            if pool["_count"] >= min_pivots:
                pool["is_cluster"] = True
            return

    active_pools.append({
        "side": side,
        "level": level,
        "born_at": t,
        "last_reinforced_at": t,
        "_sum": level,
        "_count": 1,
        "is_cluster": (1 >= min_pivots),
        "armed": False,
        "consumed": False,
    })


# ─── Chronological Walk-Forward Engine ───────────────────────────────────────

def _classify_trend(trailing_ret_bps: float, threshold_bps: float) -> str:
    if trailing_ret_bps > threshold_bps:
        return "up"
    if trailing_ret_bps < -threshold_bps:
        return "down"
    return "flat"


def _empty_stats():
    return {
        "pivots_confirmed": 0, "pools_formed": 0, "pools_armed": 0,
        "touches": 0, "bounces": 0, "breaks": 0,
        "ambiguous": 0, "timeouts": 0, "expired_untested": 0,
        "trend": {
            "up":   {"bounces": 0, "breaks": 0, "timeouts": 0},
            "down": {"bounces": 0, "breaks": 0, "timeouts": 0},
            "flat": {"bounces": 0, "breaks": 0, "timeouts": 0},
        },
    }


def walk_forward_validation(
    highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
    open_times: np.ndarray,
    confirm_high: dict, confirm_low: dict,
    k: int, min_pivots: int, epsilon_bps: float,
    excursion_bps: float, touch_tol_bps: float,
    bounce_bps: float, break_bps: float,
    timeout_bars: int, pool_expiry_bars: int,
    trend_lookback: int, trend_threshold_bps: float,
):
    n = len(closes)
    active_pools = []
    pending_measurements = []
    resolved_touches = []
    stats = {"high": _empty_stats(), "low": _empty_stats()}

    t0 = time.perf_counter()

    for t in range(n):
        cur_high  = highs[t]
        cur_low   = lows[t]
        cur_close = closes[t]

        # 1. Register confirmed pivots at bar t
        for lvl in confirm_high.get(t, ()):
            stats["high"]["pivots_confirmed"] += 1
            upsert_pivot(active_pools, lvl, "high", t, epsilon_bps, min_pivots)
        for lvl in confirm_low.get(t, ()):
            stats["low"]["pivots_confirmed"] += 1
            upsert_pivot(active_pools, lvl, "low", t, epsilon_bps, min_pivots)

        # 2. Resolve pending measurements
        still_pending = []
        for m in pending_measurements:
            side = m["side"]
            if side == "high":
                hit_bounce = (cur_low <= m["bounce_target"])
                hit_break  = (cur_high >= m["break_target"])
            else:
                hit_bounce = (cur_high >= m["bounce_target"])
                hit_break  = (cur_low <= m["break_target"])

            bucket = m["trend_bucket"]
            outcome = None

            if hit_bounce and hit_break:
                stats[side]["ambiguous"] += 1
                outcome = "ambiguous"
            elif hit_bounce:
                stats[side]["bounces"] += 1
                stats[side]["trend"][bucket]["bounces"] += 1
                outcome = "bounce"
            elif hit_break:
                stats[side]["breaks"] += 1
                stats[side]["trend"][bucket]["breaks"] += 1
                outcome = "break"
            elif (t - m["touch_bar"]) >= timeout_bars:
                stats[side]["timeouts"] += 1
                stats[side]["trend"][bucket]["timeouts"] += 1
                outcome = "timeout"
            else:
                still_pending.append(m)
                continue  # no outcome yet

            resolved_touches.append({
                "touch_time":   int(open_times[m["touch_bar"]]),
                "resolve_time": int(open_times[t]),
                "side":         side,
                "level":        float(m["level"]),
                "trend_bucket": bucket,
                "outcome":      outcome,
            })

        pending_measurements = still_pending

        # 3. Update pool state and check for touches
        for pool in active_pools:
            if pool["consumed"]:
                continue
            if (t - pool["last_reinforced_at"]) > pool_expiry_bars:
                pool["consumed"] = True
                stats[pool["side"]]["expired_untested"] += 1
                continue
            if not pool["is_cluster"]:
                continue

            side  = pool["side"]
            level = pool["level"]

            if not pool["armed"]:
                if side == "high":
                    if cur_low <= level * (1.0 - excursion_bps / 1e4):
                        pool["armed"] = True
                        stats["high"]["pools_armed"] += 1
                else:
                    if cur_high >= level * (1.0 + excursion_bps / 1e4):
                        pool["armed"] = True
                        stats["low"]["pools_armed"] += 1
                continue

            touch_zone_dist = level * (touch_tol_bps / 1e4)

            if t == 0:
                continue

            touched = False
            bounce_target = break_target = 0.0
            if side == "high":
                if closes[t - 1] < (level - touch_zone_dist) and cur_high >= (level - touch_zone_dist):
                    touched = True
                    bounce_target = level * (1.0 - bounce_bps / 1e4)
                    break_target  = level * (1.0 + break_bps / 1e4)
            else:
                if closes[t - 1] > (level + touch_zone_dist) and cur_low <= (level + touch_zone_dist):
                    touched = True
                    bounce_target = level * (1.0 + bounce_bps / 1e4)
                    break_target  = level * (1.0 - break_bps / 1e4)

            if not touched:
                continue

            if t >= trend_lookback:
                past_close = closes[t - trend_lookback]
                trailing_ret_bps = (cur_close / past_close - 1.0) * 1e4
                trend_bucket = _classify_trend(trailing_ret_bps, trend_threshold_bps)
            else:
                trend_bucket = "flat"

            pool["consumed"] = True
            stats[side]["touches"] += 1
            pending_measurements.append({
                "side": side,
                "level": level,
                "touch_bar": t,
                "bounce_target": bounce_target,
                "break_target":  break_target,
                "trend_bucket":  trend_bucket,
            })

        if t % 5000 == 0:
            active_pools = [p for p in active_pools if not p["consumed"]]

    elapsed = (time.perf_counter() - t0) * 1000.0
    print(f"Processed {n:,} bars in {elapsed:.1f} ms\n")

    touches_df = pl.DataFrame(resolved_touches) if resolved_touches else pl.DataFrame()
    return stats, touches_df


# ─── Reporting ───────────────────────────────────────────────────────────────

def _rate(b, k):
    d = b + k
    return (b / d) if d > 0 else 0.0


def report(stats, trend_lookback, trend_threshold_bps):
    print("=" * 72)
    print("STRUCTURAL BARRIER VALIDATION (ZERO LOOKAHEAD / TREND-SPLIT)")
    print("=" * 72)
    print(f"Trend lookback: {trend_lookback} bars  "
          f"| Up if +{trend_threshold_bps:.0f} bps, Down if -{trend_threshold_bps:.0f} bps")

    for label, side in [("RESISTANCE (Equal Highs)", "high"),
                        ("SUPPORT    (Equal Lows)",  "low")]:
        s = stats[side]
        decisive = s["bounces"] + s["breaks"]
        rate = _rate(s["bounces"], s["breaks"])

        print(f"\n{label}")
        print(f"  Pivots Confirmed:     {s['pivots_confirmed']:>8,}")
        print(f"  Pools Armed:          {s['pools_armed']:>8,}")
        print(f"  Valid Touches:        {s['touches']:>8,}")
        print(f"  Bounces:              {s['bounces']:>8,}")
        print(f"  Breaks:               {s['breaks']:>8,}")
        print(f"  Ambiguous:            {s['ambiguous']:>8,}")
        print(f"  Timeouts:             {s['timeouts']:>8,}")
        print(f"  Decisive:             {decisive:>8,}")
        print(f"  OVERALL BOUNCE RATE:  {rate:>8.2%}")

        print(f"\n  -- Trend buckets --")
        print(f"  {'bucket':<8}  {'bounces':>8}  {'breaks':>8}  {'timeouts':>8}  {'bounce_rate':>12}")
        for bucket in ("up", "down", "flat"):
            tb = s["trend"][bucket]
            d = tb["bounces"] + tb["breaks"]
            r = (tb["bounces"] / d) if d > 0 else 0.0
            print(f"  {bucket:<8}  {tb['bounces']:>8,}  {tb['breaks']:>8,}  "
                  f"{tb['timeouts']:>8,}  {r:>11.2%}")

    h = stats["high"]; l = stats["low"]
    agg_b = h["bounces"] + l["bounces"]
    agg_k = h["breaks"]  + l["breaks"]
    agg = (agg_b / (agg_b + agg_k)) if (agg_b + agg_k) > 0 else 0.0

    print("\n" + "=" * 72)
    print(f"AGGREGATE BOUNCE RATE:  {agg:.2%}")
    print("=" * 72 + "\n")


# ─── Execution Entry ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=DATA_GLOB,
                        help="Path to kline parquet file(s), supports wildcards or comma-separated lists")
    parser.add_argument("--k", type=int, default=K)
    parser.add_argument("--min-pivots", type=int, default=MIN_PIVOTS)
    parser.add_argument("--epsilon-bps", type=float, default=EPSILON_BPS)
    parser.add_argument("--excursion-bps", type=float, default=EXCURSION_BPS)
    parser.add_argument("--touch-tol-bps", type=float, default=TOUCH_TOL_BPS)
    parser.add_argument("--bounce-bps", type=float, default=BOUNCE_BPS)
    parser.add_argument("--break-bps", type=float, default=BREAK_BPS)
    parser.add_argument("--timeout-bars", type=int, default=TIMEOUT_BARS)
    parser.add_argument("--pool-expiry-bars", type=int, default=POOL_EXPIRY_BARS)
    parser.add_argument("--trend-lookback-bars", type=int, default=TREND_LOOKBACK)
    parser.add_argument("--trend-threshold-bps", type=float, default=TREND_THRESH_BPS)
    parser.add_argument("--out", type=str, default=TOUCHES_OUTPUT,
                        help="Path to write resolved touches parquet")
    args = parser.parse_args()

    df = load_data(args.data)
    highs      = df["high"].to_numpy()
    lows       = df["low"].to_numpy()
    closes     = df["close"].to_numpy()
    open_times = df["open_time"].to_numpy()

    print(f"\n[*] Detecting swing pivots (k={args.k} bars)...")
    sh_indices, sl_indices = detect_pivots_vectorized(highs, lows, args.k)
    print(f"    Swing Highs found: {len(sh_indices):,}")
    print(f"    Swing Lows found:  {len(sl_indices):,}")

    confirm_high = defaultdict(list)
    confirm_low  = defaultdict(list)
    for p in sh_indices:
        confirm_high[p + args.k].append(highs[p])
    for p in sl_indices:
        confirm_low[p + args.k].append(lows[p])

    print("\n[*] Starting walk-forward barrier validation...")
    stats, touches_df = walk_forward_validation(
        highs, lows, closes, open_times,
        confirm_high, confirm_low,
        k=args.k,
        min_pivots=args.min_pivots,
        epsilon_bps=args.epsilon_bps,
        excursion_bps=args.excursion_bps,
        touch_tol_bps=args.touch_tol_bps,
        bounce_bps=args.bounce_bps,
        break_bps=args.break_bps,
        timeout_bars=args.timeout_bars,
        pool_expiry_bars=args.pool_expiry_bars,
        trend_lookback=args.trend_lookback_bars,
        trend_threshold_bps=args.trend_threshold_bps,
    )

    report(stats, args.trend_lookback_bars, args.trend_threshold_bps)

    if not touches_df.is_empty():
        touches_df.write_parquet(args.out)
        print(f"[+] Exported {len(touches_df):,} resolved touches to {args.out}\n")
    else:
        print("[!] No resolved touches to export.\n")


if __name__ == "__main__":
    main()