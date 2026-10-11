"""
breakout_engine.py — Trend-Aligned Breakout System

Trades the pure expansion phase. 
Trigger: Candle closes completely THROUGH the structural barrier.
Filter: Must be aligned with the 14-day (336-hour) macro trend.
Target: Fixed 1:3 R:R (150 bps stop, 450 bps target) to outrun all exchange fees.
"""

import argparse
import glob
import time
import numpy as np
import polars as pl
from collections import defaultdict

DATA_GLOB         = "data_store/*_1h.parquet"
K                 = 12
MIN_PIVOTS        = 2
EPSILON_BPS       = 25.0
STOP_BPS          = 150.0
TARGET_BPS        = 450.0  # 1:3 RR
TREND_LOOKBACK    = 336    # 14 days on 1h
TREND_THRESH_BPS  = 300.0  # Must be trending at least 3% 
TIMEOUT_BARS      = 168    # 7 days max hold

def load_data(data_path: str) -> pl.DataFrame:
    files = []
    for pattern in data_path.split(","):
        files.extend(glob.glob(pattern.strip()))
    files = sorted(list(set(files)))
    if not files: raise SystemExit(f"No files matched {data_path}")
    print(f"Loading {len(files)} files from {data_path}...")
    df = pl.concat([pl.read_parquet(f) for f in files]).sort("open_time")
    print(f"Loaded {len(df):,} bars | Range: {df['open_time'].min()} → {df['open_time'].max()}")
    return df

def detect_pivots(highs: np.ndarray, lows: np.ndarray, k: int):
    n = len(highs)
    is_high, is_low = np.ones(n, dtype=bool), np.ones(n, dtype=bool)
    for shift in range(-k, k + 1):
        if shift == 0: continue
        is_high &= (highs > np.roll(highs, shift))
        is_low  &= (lows < np.roll(lows, shift))
    is_high[:k], is_high[-k:] = False, False
    is_low[:k],  is_low[-k:]  = False, False
    return np.where(is_high)[0], np.where(is_low)[0]

def upsert_pivot(active_pools: list, level: float, side: str, t: int, eps_bps: float, min_p: int):
    tol = eps_bps / 1e4
    for pool in active_pools:
        if pool["side"] != side or pool["consumed"]: continue
        if abs(level - pool["level"]) / pool["level"] <= tol:
            pool["_sum"] += level
            pool["_count"] += 1
            pool["level"] = pool["_sum"] / pool["_count"]
            if pool["_count"] >= min_p: pool["is_cluster"] = True
            return
    active_pools.append({
        "side": side, "level": level, "_sum": level, "_count": 1,
        "is_cluster": (1 >= min_p), "consumed": False
    })

def walk_forward_breakouts(
    highs, lows, closes, open_times, ch, cl,
    k, min_p, eps_bps, target_bps, stop_bps, trend_lb, trend_th, timeout
):
    n = len(closes)
    active_pools, pending_trades = [], []
    stats = {"long_breakouts": {"trades": 0, "wins": 0, "losses": 0, "timeouts": 0},
             "short_breakdowns": {"trades": 0, "wins": 0, "losses": 0, "timeouts": 0}}

    for t in range(n):
        cur_high, cur_low, cur_close = highs[t], lows[t], closes[t]

        for lvl in ch.get(t, ()): upsert_pivot(active_pools, lvl, "high", t, eps_bps, min_p)
        for lvl in cl.get(t, ()): upsert_pivot(active_pools, lvl, "low", t, eps_bps, min_p)

        still_pending = []
        for trade in pending_trades:
            side = trade["side"]
            stat_key = "long_breakouts" if side == "long" else "short_breakdowns"
            
            if side == "long":
                hit_tp, hit_sl = (cur_high >= trade["tp"]), (cur_low <= trade["sl"])
            else:
                hit_tp, hit_sl = (cur_low <= trade["tp"]), (cur_high >= trade["sl"])

            if hit_tp and hit_sl: stats[stat_key]["losses"] += 1
            elif hit_tp:          stats[stat_key]["wins"] += 1
            elif hit_sl:          stats[stat_key]["losses"] += 1
            elif (t - trade["entry_bar"]) >= timeout: stats[stat_key]["timeouts"] += 1
            else: still_pending.append(trade)
        pending_trades = still_pending

        if t < trend_lb: continue
        macro_trend_bps = (cur_close / closes[t - trend_lb] - 1.0) * 1e4

        for pool in active_pools:
            if pool["consumed"] or not pool["is_cluster"]: continue
            side, level = pool["side"], pool["level"]

            # LONG BREAKOUT: Price closes ABOVE resistance & Macro Trend is UP
            if side == "high" and cur_close > level and macro_trend_bps > trend_th:
                pool["consumed"] = True
                stats["long_breakouts"]["trades"] += 1
                pending_trades.append({
                    "side": "long",
                    "tp": cur_close * (1.0 + target_bps / 1e4),
                    "sl": cur_close * (1.0 - stop_bps / 1e4),
                    "entry_bar": t
                })

            # SHORT BREAKDOWN: Price closes BELOW support & Macro Trend is DOWN
            elif side == "low" and cur_close < level and macro_trend_bps < -trend_th:
                pool["consumed"] = True
                stats["short_breakdowns"]["trades"] += 1
                pending_trades.append({
                    "side": "short",
                    "tp": cur_close * (1.0 - target_bps / 1e4),
                    "sl": cur_close * (1.0 + stop_bps / 1e4),
                    "entry_bar": t
                })

    return stats

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=DATA_GLOB)
    args = parser.parse_args()

    df = load_data(args.data)
    highs, lows, closes = df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy()
    open_times = df["open_time"].to_numpy()

    print(f"[*] Detecting swing pivots (k={K})...")
    sh, sl = detect_pivots(highs, lows, K)
    ch, cl = defaultdict(list), defaultdict(list)
    for p in sh: ch[p + K].append(highs[p])
    for p in sl: cl[p + K].append(lows[p])

    print("[*] Simulating Trend-Aligned Breakouts...")
    stats = walk_forward_breakouts(
        highs, lows, closes, open_times, ch, cl,
        K, MIN_PIVOTS, EPSILON_BPS, TARGET_BPS, STOP_BPS, TREND_LOOKBACK, TREND_THRESH_BPS, TIMEOUT_BARS
    )

    print("\n" + "=" * 60)
    print("TREND-ALIGNED BREAKOUT PERFORMANCE")
    print(f"Target: {TARGET_BPS} bps | Stop: {STOP_BPS} bps | R:R: 3.0")
    print("=" * 60)
    
    for key, label in [("long_breakouts", "Long Breakouts (Above Resistance)"), 
                       ("short_breakdowns", "Short Breakdowns (Below Support)")]:
        s = stats[key]
        decisive = s["wins"] + s["losses"]
        wr = (s["wins"] / decisive) if decisive > 0 else 0
        print(f"\n{label}")
        print(f"  Trades Executed: {s['trades']}")
        print(f"  Wins (TP):       {s['wins']}")
        print(f"  Losses (SL):     {s['losses']}")
        print(f"  Timeouts:        {s['timeouts']}")
        print(f"  Win Rate:        {wr:.2%}")

if __name__ == "__main__":
    main()