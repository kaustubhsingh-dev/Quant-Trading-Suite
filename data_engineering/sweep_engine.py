"""
sweep_engine_dynamic.py — Dynamic Target & Structural Stop Sweep Engine.

Instead of fixed targets, this engine:
1. Places the Stop Loss strictly at the extreme of the sweep wick (±5 bps).
2. Scans active market structure for the nearest opposing liquidity pool to set Take Profit.
3. Calculates actual R:R.
4. Executes ONLY if 1.5 <= R:R <= MAX_RR.
"""

import argparse
import glob
import time
import numpy as np
import polars as pl
from collections import defaultdict

# ─── Configuration Defaults ──────────────────────────────────────────────────
DATA_GLOB         = "data_store/*_1h.parquet"
K                 = 12
MIN_PIVOTS        = 2
EPSILON_BPS       = 25.0
EXCURSION_BPS     = 100.0
TIMEOUT_BARS      = 72
VWAP_WINDOW       = 72
MIN_RR            = 1.5
MAX_RR            = 3.0
STOP_BUFFER_BPS   = 5.0  # 5 bps breathing room beyond the wick

def load_data(data_path: str, vwap_window: int) -> pl.DataFrame:
    files = []
    for pattern in data_path.split(","):
        files.extend(glob.glob(pattern.strip()))
    files = sorted(list(set(files)))
    if not files:
        raise SystemExit(f"No files matched {data_path}")
    
    print(f"Loading {len(files)} files from {data_path}...")
    df = pl.concat([pl.read_parquet(f) for f in files]).sort("open_time")
    
    df = df.with_columns([
        (pl.col("close") * pl.col("volume")).rolling_sum(window_size=vwap_window).alias("vol_price"),
        pl.col("volume").rolling_sum(window_size=vwap_window).alias("vol_sum")
    ]).with_columns(
        (pl.col("vol_price") / pl.col("vol_sum")).alias("vwap")
    ).drop(["vol_price", "vol_sum"]).fill_null(strategy="forward")
    
    print(f"Loaded {len(df):,} bars | Range: {df['open_time'].min()} → {df['open_time'].max()}")
    return df

def detect_pivots(highs: np.ndarray, lows: np.ndarray, k: int):
    n = len(highs)
    is_high = np.ones(n, dtype=bool)
    is_low  = np.ones(n, dtype=bool)
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
            pool["last_reinforced"] = t
            if pool["_count"] >= min_p:
                pool["is_cluster"] = True
            return
    active_pools.append({
        "side": side, "level": level, "born_at": t, "last_reinforced": t,
        "_sum": level, "_count": 1, "is_cluster": (1 >= min_p),
        "armed": False, "consumed": False,
    })

def walk_forward_dynamic_sweeps(
    highs, lows, closes, vwaps, open_times, confirm_high, confirm_low,
    k, min_p, eps_bps, exc_bps, min_rr, max_rr, timeout_bars, stop_buffer_bps
):
    n = len(closes)
    active_pools = []
    pending_trades = []
    
    stats = {
        "high": {"sweeps": 0, "wins": 0, "losses": 0, "timeouts": 0, "avg_rr": []},
        "low":  {"sweeps": 0, "wins": 0, "losses": 0, "timeouts": 0, "avg_rr": []}
    }

    for t in range(n):
        cur_high, cur_low, cur_close, cur_vwap = highs[t], lows[t], closes[t], vwaps[t]

        # 1. Register Pivots
        for lvl in confirm_high.get(t, ()): upsert_pivot(active_pools, lvl, "high", t, eps_bps, min_p)
        for lvl in confirm_low.get(t, ()):  upsert_pivot(active_pools, lvl, "low", t, eps_bps, min_p)

        # 2. Track Open Trades
        still_pending = []
        for trade in pending_trades:
            side = trade["side"]
            if side == "high": # Short trade
                hit_tp = (cur_low <= trade["tp"])
                hit_sl = (cur_high >= trade["sl"])
            else: # Long trade
                hit_tp = (cur_high >= trade["tp"])
                hit_sl = (cur_low <= trade["sl"])

            if hit_tp and hit_sl:  stats[side]["losses"] += 1 
            elif hit_tp:           stats[side]["wins"] += 1
            elif hit_sl:           stats[side]["losses"] += 1
            elif (t - trade["entry_bar"]) >= timeout_bars: stats[side]["timeouts"] += 1
            else: still_pending.append(trade)
        pending_trades = still_pending

        # 3. Check for Liquidity Sweeps with Dynamic Targeting
        for pool in active_pools:
            if pool["consumed"] or not pool["is_cluster"]: continue
            side, level = pool["side"], pool["level"]

            if not pool["armed"]:
                if side == "high" and cur_low <= level * (1.0 - exc_bps / 1e4): pool["armed"] = True
                if side == "low" and cur_high >= level * (1.0 + exc_bps / 1e4): pool["armed"] = True
                continue

            swept = False
            tp = sl = rr = 0.0

            # RESISTANCE SWEEP (Short Setup)
            if side == "high":
                if cur_high >= level and cur_close < level and cur_close < cur_vwap:
                    sl = cur_high * (1.0 + stop_buffer_bps / 1e4)
                    risk = sl - cur_close
                    
                    # Find highest active support pool below current price
                    valid_targets = [p["level"] for p in active_pools if p["side"] == "low" and p["level"] < cur_close and not p["consumed"]]
                    if valid_targets and risk > 0:
                        tp = max(valid_targets)
                        reward = cur_close - tp
                        rr = reward / risk
                        if min_rr <= rr <= max_rr: swept = True

            # SUPPORT SWEEP (Long Setup)
            elif side == "low":
                if cur_low <= level and cur_close > level and cur_close > cur_vwap:
                    sl = cur_low * (1.0 - stop_buffer_bps / 1e4)
                    risk = cur_close - sl
                    
                    # Find lowest active resistance pool above current price
                    valid_targets = [p["level"] for p in active_pools if p["side"] == "high" and p["level"] > cur_close and not p["consumed"]]
                    if valid_targets and risk > 0:
                        tp = min(valid_targets)
                        reward = tp - cur_close
                        rr = reward / risk
                        if min_rr <= rr <= max_rr: swept = True

            if swept:
                pool["consumed"] = True
                stats[side]["sweeps"] += 1
                stats[side]["avg_rr"].append(rr)
                pending_trades.append({
                    "side": side, "tp": tp, "sl": sl, "entry_bar": t
                })

        if t % 5000 == 0:
            active_pools = [p for p in active_pools if not p["consumed"]]

    return stats

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=DATA_GLOB)
    args = parser.parse_args()

    df = load_data(args.data, VWAP_WINDOW)
    highs, lows, closes = df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy()
    vwaps, open_times = df["vwap"].to_numpy(), df["open_time"].to_numpy()

    print(f"[*] Detecting swing pivots (k={K})...")
    sh, sl = detect_pivots(highs, lows, K)

    ch, cl = defaultdict(list), defaultdict(list)
    for p in sh: ch[p + K].append(highs[p])
    for p in sl: cl[p + K].append(lows[p])

    print("[*] Simulating Dynamic RR Liquidity Sweeps...")
    stats = walk_forward_dynamic_sweeps(
        highs, lows, closes, vwaps, open_times, ch, cl,
        K, MIN_PIVOTS, EPSILON_BPS, EXCURSION_BPS, MIN_RR, MAX_RR, TIMEOUT_BARS, STOP_BUFFER_BPS
    )

    print("\n" + "=" * 60)
    print("DYNAMIC LIQUIDITY SWEEP PERFORMANCE (VWAP-FILTERED)")
    print(f"Required Target R:R: {MIN_RR} to {MAX_RR}")
    print("=" * 60)
    for label, side in [("Resistance Sweeps (Short)", "high"), ("Support Sweeps (Long)", "low")]:
        s = stats[side]
        decisive = s["wins"] + s["losses"]
        wr = (s["wins"] / decisive) if decisive > 0 else 0
        avg_rr = np.mean(s["avg_rr"]) if s["avg_rr"] else 0
        print(f"\n{label}")
        print(f"  Trades Executed: {s['sweeps']}")
        print(f"  Avg Entry R:R:   {avg_rr:.2f}")
        print(f"  Wins (TP):       {s['wins']}")
        print(f"  Losses (SL):     {s['losses']}")
        print(f"  Timeouts:        {s['timeouts']}")
        print(f"  Win Rate:        {wr:.2%}")

if __name__ == "__main__":
    main()