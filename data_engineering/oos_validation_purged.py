"""
oos_validation_purged.py — Walk-forward validation with toxic asset exclusion.
"""

import glob
import json
import time
import urllib.request
import numpy as np
import polars as pl
from datetime import datetime, timezone

# ─── Config ──────────────────────────────────────────────────────────────────

DATA_GLOB       = "data_store/*_1h.parquet"
TOXIC_ASSETS    = ["LTCUSDT", "ATOMUSDT", "DOTUSDT"]

MAKER_FEE_BPS   = 2.0        # 0.02% pre-GST
GST_RATE        = 0.18       # 18% GST on fees
SLIPPAGE_BPS    = 3.0        # estimated per-leg slippage (majors ~1-2, alts ~5-10)

# Effective per-leg cost = (maker_fee * (1 + GST) + slippage) / 10000
PER_LEG_COST = (MAKER_FEE_BPS * (1 + GST_RATE) + SLIPPAGE_BPS) / 10000.0
print(f"[*] Effective per-leg cost: {PER_LEG_COST*10000:.4f} bps "
      f"(maker {MAKER_FEE_BPS} + GST + slippage {SLIPPAGE_BPS})")

CONFIGS = [
    {"name": "A (pre-committed)", "lookback": 336, "rebalance": 24, "top_n": 3},
    {"name": "B (grid winner)",   "lookback": 504, "rebalance": 48, "top_n": 5},
]

# Rolling walk-forward windows (in months)
TRAIN_MONTHS = 18
TEST_MONTHS  = 6
STEP_MONTHS  = 6


# ─── Funding Fetch (window-based) ────────────────────────────────────────────

def fetch_all_funding(symbol: str, start_ms: int, end_ms: int) -> list:
    all_events = []
    window_ms = 30 * 24 * 3600 * 1000
    cursor = start_ms

    while cursor < end_ms:
        w_end = min(cursor + window_ms, end_ms)
        url = (f"https://fapi.binance.com/fapi/v1/fundingRate"
               f"?symbol={symbol}&startTime={cursor}&endTime={w_end}&limit=1000")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode())
            if data:
                all_events.extend(data)
        except Exception as e:
            print(f"    [!] {symbol} window {cursor}->{w_end}: {e}")
        cursor = w_end
        time.sleep(0.05)   # gentle rate limiting

    return all_events


def fetch_funding_df(symbols: list, start_ms: int, end_ms: int) -> pl.DataFrame:
    print(f"\n[*] Fetching funding history [{datetime.fromtimestamp(start_ms/1000, tz=timezone.utc).date()} "
          f"-> {datetime.fromtimestamp(end_ms/1000, tz=timezone.utc).date()}]")

    frames = []
    for sym in symbols:
        events = fetch_all_funding(sym, start_ms, end_ms)
        if events:
            df = pl.DataFrame({
                "funding_time": [int(e["fundingTime"]) for e in events],
                "funding_rate": [float(e["fundingRate"]) for e in events],
                "symbol":       [sym] * len(events),
            })
            frames.append(df)
            print(f"    {sym:<12} {len(events):>6} events "
                  f"({len(events)/3/365:.2f} yrs)")
        else:
            print(f"    {sym:<12} (no data)")

    if not frames:
        return pl.DataFrame(schema={
            "funding_time": pl.Int64, "funding_rate": pl.Float64, "symbol": pl.String,
        })
    return pl.concat(frames).sort(["symbol", "funding_time"])


# ─── Funding Drag (vectorized) ───────────────────────────────────────────────

def compute_funding_drag(symbols, times, w_mat, funding_df, reb_ms):
    funding_drag = np.zeros(len(times))
    if funding_df.is_empty():
        return funding_drag

    for s_idx, sym in enumerate(symbols):
        sym_df = funding_df.filter(pl.col("symbol") == sym)
        if sym_df.is_empty():
            continue
        ev_times = sym_df["funding_time"].to_numpy()
        ev_rates = sym_df["funding_rate"].to_numpy()

        cumsum = np.concatenate([[0.0], np.cumsum(ev_rates)])
        lefts  = np.searchsorted(ev_times, times, side="left")
        rights = np.searchsorted(ev_times, times + reb_ms, side="left")
        rates_in_window = cumsum[rights] - cumsum[lefts]

        funding_drag += -w_mat[:, s_idx] * rates_in_window

    return funding_drag


# ─── Core Engine ─────────────────────────────────────────────────────────────

def compute_weights(df: pl.DataFrame, cfg: dict):
    lb, reb, top_n = cfg["lookback"], cfg["rebalance"], cfg["top_n"]

    df_cfg = df.sort(["symbol", "open_time"]).with_columns([
        (pl.col("close") / pl.col("close").shift(1) - 1.0).over("symbol").alias("hourly_ret")
    ])
    df_cfg = df_cfg.with_columns([
        pl.col("hourly_ret").rolling_std(window_size=lb).over("symbol").alias("vol_lb"),
        (pl.col("close") / pl.col("close").shift(lb) - 1.0).over("symbol").alias("ret_lb"),
    ]).with_columns([
        (pl.col("ret_lb") / pl.col("vol_lb").clip(lower_bound=0.001)).alias("mom_score")
    ])

    df_cfg = df_cfg.with_columns([
        (pl.col("close").shift(-reb) / pl.col("close") - 1.0).over("symbol").alias("fwd_ret")
    ]).drop_nulls(subset=["mom_score", "fwd_ret"])

    rebal_df = df_cfg.filter((pl.col("open_time") % (reb * 3600 * 1000)) == 0)
    rebal_df = rebal_df.with_columns([
        pl.col("symbol").count().over("open_time").alias("active_coins")
    ]).filter(pl.col("active_coins") >= (15 - len(TOXIC_ASSETS) - 2)) # Adjusted for smaller universe

    rebal_df = rebal_df.with_columns([
        pl.col("mom_score").rank(descending=True).over("open_time").alias("rank")
    ])

    w_leg = 0.5 / top_n
    rebal_df = rebal_df.with_columns([
        pl.when(pl.col("rank") <= top_n).then(pl.lit(w_leg))
          .when(pl.col("rank") > pl.col("active_coins") - top_n).then(pl.lit(-w_leg))
          .otherwise(pl.lit(0.0)).alias("target_weight")
    ])

    piv_w = rebal_df.pivot(index="open_time", on="symbol", values="target_weight").fill_null(0.0)
    piv_r = rebal_df.pivot(index="open_time", on="symbol", values="fwd_ret").fill_null(0.0)

    symbols = [c for c in piv_w.columns if c != "open_time"]
    times   = piv_w["open_time"].to_numpy()
    w_mat   = piv_w.select(symbols).to_numpy()
    r_mat   = piv_r.select(symbols).to_numpy()

    return times, w_mat, r_mat, symbols


def run_config(df, funding_df, cfg):
    reb_ms = cfg["rebalance"] * 3600 * 1000
    times, w_mat, r_mat, symbols = compute_weights(df, cfg)

    gross_returns = np.sum(w_mat * r_mat, axis=1)

    turnover = np.zeros(len(w_mat))
    turnover[0]  = np.sum(np.abs(w_mat[0]))
    turnover[1:] = np.sum(np.abs(w_mat[1:] - w_mat[:-1]), axis=1)
    fee_drag = turnover * PER_LEG_COST

    funding_drag = compute_funding_drag(symbols, times, w_mat, funding_df, reb_ms)

    net_returns = gross_returns - fee_drag + funding_drag

    periods = pl.DataFrame({
        "open_time":    times,
        "gross_ret":    gross_returns,
        "fee_drag":     fee_drag,
        "funding_drag": funding_drag,
        "net_ret":      net_returns,
    })
    return periods, symbols, w_mat, r_mat


# ─── Metrics ─────────────────────────────────────────────────────────────────

def metrics(net_returns: np.ndarray, reb_h: int) -> dict:
    if len(net_returns) < 2:
        return {"n": 0, "sharpe": 0.0, "ann_ret": 0.0, "max_dd": 0.0, "win_rate": 0.0}
    equity = np.cumprod(1.0 + net_returns)
    wm = np.maximum.accumulate(equity)
    dd = (equity / wm) - 1.0
    ppy = (24 / reb_h) * 365
    years = len(net_returns) / ppy
    mu, sd = float(np.mean(net_returns)), float(np.std(net_returns))
    sharpe = (mu / sd) * np.sqrt(ppy) if sd > 0 else 0.0
    return {
        "n":        len(net_returns),
        "sharpe":   sharpe,
        "ann_ret":  ((equity[-1] ** (1 / years)) - 1.0) * 100 if years > 0 else 0.0,
        "max_dd":   float(np.min(dd)) * 100,
        "win_rate": float(np.mean(net_returns > 0)) * 100,
    }


def print_metrics(label: str, m: dict):
    print(f"    {label:<22} n={m['n']:>5}  Sharpe={m['sharpe']:>6.2f}  "
          f"AnnRet={m['ann_ret']:>7.2f}%  MaxDD={m['max_dd']:>7.2f}%  "
          f"WR={m['win_rate']:>5.1f}%")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    files = sorted(glob.glob(DATA_GLOB))
    print(f"[*] Loading asset files and purging toxic assets: {TOXIC_ASSETS}...")

    symbols, dfs = [], []
    for f in files:
        sym = f.replace("\\", "/").split("/")[-1].replace("_1h.parquet", "")
        if sym in TOXIC_ASSETS:
            print(f"  [-] Dropped {sym} from universe.")
            continue
            
        symbols.append(sym)
        dfs.append(pl.read_parquet(f).with_columns(pl.lit(sym).alias("symbol")))

    df = pl.concat(dfs).sort(["symbol", "open_time"])
    print(f"    Total active symbols: {len(symbols)}")
    print(f"    Total bars: {len(df):,}")

    data_start = int(df["open_time"].min())
    data_end   = int(df["open_time"].max())

    funding_df = fetch_funding_df(symbols, data_start, data_end)

    total_events = len(funding_df)
    print(f"    Total funding events: {total_events:,}")

    for cfg in CONFIGS:
        print(f"\n{'=' * 80}")
        print(f"Config {cfg['name']}")
        print(f"    lookback={cfg['lookback']}h  rebalance={cfg['rebalance']}h  top_n={cfg['top_n']}")
        print(f"{'=' * 80}\n")

        periods, syms, w_mat, r_mat = run_config(df, funding_df, cfg)
        net = periods["net_ret"].to_numpy()
        reb = cfg["rebalance"]

        print("  Full sample:")
        print_metrics("FULL", metrics(net, reb))
        avg_funding = float(np.mean(periods["funding_drag"].to_numpy())) * 1e4
        avg_fee     = float(np.mean(periods["fee_drag"].to_numpy())) * 1e4
        print(f"    Avg per-period: funding={avg_funding:+.2f} bps  fee={avg_fee:.2f} bps")

        print("\n  Rolling walk-forward (train=18mo, test=6mo, step=6mo):")
        print(f"    {'Window':<28} {'Train Sharpe':>13} {'Test Sharpe':>12} {'Test AnnRet':>13} {'Test MaxDD':>11}")

        times_ms = periods["open_time"].to_numpy()
        results = []
        train_ms = TRAIN_MONTHS * 30 * 24 * 3600 * 1000
        step_ms  = STEP_MONTHS  * 30 * 24 * 3600 * 1000

        w_start = data_start
        while w_start + train_ms + step_ms <= data_end:
            train_start = w_start
            train_end   = w_start + train_ms
            test_start  = train_end
            test_end    = test_start + step_ms

            train_mask = (times_ms >= train_start) & (times_ms < train_end)
            test_mask  = (times_ms >= test_start)  & (times_ms < test_end)

            if train_mask.sum() >= 10 and test_mask.sum() >= 10:
                m_tr = metrics(net[train_mask], reb)
                m_te = metrics(net[test_mask],  reb)

                tr_lbl = (f"{datetime.fromtimestamp(train_start/1000, tz=timezone.utc).strftime('%Y-%m')}"
                          f"->{datetime.fromtimestamp(train_end/1000, tz=timezone.utc).strftime('%Y-%m')}")
                te_lbl = (f"{datetime.fromtimestamp(test_start/1000, tz=timezone.utc).strftime('%Y-%m')}"
                          f"->{datetime.fromtimestamp(test_end/1000, tz=timezone.utc).strftime('%Y-%m')}")

                print(f"    {tr_lbl:<13} | {te_lbl:<13} "
                      f"{m_tr['sharpe']:>12.2f} {m_te['sharpe']:>12.2f} "
                      f"{m_te['ann_ret']:>12.2f}% {m_te['max_dd']:>10.2f}%")
                results.append(m_te["sharpe"])

            w_start += step_ms

        if results:
            print(f"\n    Test Sharpe across windows: "
                  f"mean={np.mean(results):.2f}  median={np.median(results):.2f}  "
                  f"min={np.min(results):.2f}  max={np.max(results):.2f}")

        print(f"\n  Per-asset gross contribution (full sample):")
        contrib = {syms[i]: float(np.sum(w_mat[:, i] * r_mat[:, i]))
                   for i in range(len(syms))}
        total_abs = sum(abs(v) for v in contrib.values()) or 1.0
        for sym, c in sorted(contrib.items(), key=lambda kv: -abs(kv[1])):
            pct = 100 * c / total_abs
            bar = "█" * int(abs(pct) / 2)
            sign = "+" if c >= 0 else "-"
            print(f"    {sym:<10} {sign}{abs(c):>8.4f}  ({pct:>+6.1f}%)  {bar}")


if __name__ == "__main__":
    main()