"""
train_test_validation.py — Out-of-sample validation of cross-sectional momentum.

Tests two configs:
  A) 336h lookback / 24h rebalance / Top3   (pre-committed)
  B) 504h lookback / 48h rebalance / Top5   (grid-search winner)

Fixes funding integration from previous scripts:
  - Accumulates ALL 8h funding settlements in each rebalance window
  - Paginates Binance API to fetch full funding history (not just last 1000 events)
  - Correct sign: long pays positive funding, short receives positive funding

Reports Sharpe, Ann Return, Max DD on:
  - Full sample
  - Train (first 70%)
  - Test  (last 30%)

Plus per-asset contribution breakdown on full sample.
"""

import glob
import json
import time
import urllib.request
import numpy as np
import polars as pl

# ─── Config ──────────────────────────────────────────────────────────────────

DATA_GLOB   = "data_store/*_1h.parquet"
MAKER_FEE   = 0.0002       # 0.02% per leg (pre-GST)
TRAIN_FRAC  = 0.70         # 70% train, 30% test

CONFIGS = [
    {"name": "A (pre-committed)", "lookback": 336, "rebalance": 24, "top_n": 3},
    {"name": "B (grid winner)",   "lookback": 504, "rebalance": 48, "top_n": 5},
]

# ─── Funding Fetch (with pagination) ─────────────────────────────────────────

def fetch_all_funding(symbol: str) -> list:
    """Fetch all funding events for symbol, paginating backward."""
    all_events = []
    end_ms = None
    for _ in range(10):  # max 10 pages × 1000 = 10k events ≈ 3.3 years
        url = f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={symbol}&limit=1000"
        if end_ms is not None:
            url += f"&endTime={end_ms}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode())
        except Exception as e:
            print(f"    [!] fetch failed for {symbol}: {e}")
            break

        if not data:
            break
        all_events.extend(data)

        if len(data) < 1000:
            break
        end_ms = min(int(item["fundingTime"]) for item in data) - 1
        time.sleep(0.1)   # gentle rate limit

    return all_events


def fetch_funding_df(symbols: list) -> pl.DataFrame:
    print(f"[*] Fetching funding history for {len(symbols)} symbols...")
    frames = []
    for sym in symbols:
        events = fetch_all_funding(sym)
        if events:
            df = pl.DataFrame({
                "funding_time": [int(e["fundingTime"]) for e in events],
                "funding_rate": [float(e["fundingRate"]) for e in events],
                "symbol":       [sym] * len(events),
            })
            frames.append(df)
            print(f"    {sym}: {len(events):>6} events")
        else:
            print(f"    {sym}: (no data)")

    if not frames:
        return pl.DataFrame(schema={
            "funding_time": pl.Int64, "funding_rate": pl.Float64, "symbol": pl.String,
        })
    return pl.concat(frames).sort(["symbol", "funding_time"])


# ─── Core Engine ─────────────────────────────────────────────────────────────

def run_config(df: pl.DataFrame, funding_df: pl.DataFrame, cfg: dict):
    """
    Returns (periods_df, per_asset_contrib) where:
      periods_df       = one row per rebalance with open_time, gross_ret, fee_drag,
                         funding_drag, net_ret
      per_asset_contrib= {symbol: cumulative_contribution_to_returns}
    """
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
    ]).filter(pl.col("active_coins") >= 10)

    rebal_df = rebal_df.with_columns([
        pl.col("mom_score").rank(descending=True).over("open_time").alias("rank")
    ])

    w_leg = 0.5 / top_n
    rebal_df = rebal_df.with_columns([
        pl.when(pl.col("rank") <= top_n).then(pl.lit(w_leg))
          .when(pl.col("rank") > pl.col("active_coins") - top_n).then(pl.lit(-w_leg))
          .otherwise(pl.lit(0.0)).alias("target_weight")
    ])

    piv_weight = rebal_df.pivot(index="open_time", on="symbol", values="target_weight").fill_null(0.0)
    piv_fwd    = rebal_df.pivot(index="open_time", on="symbol", values="fwd_ret").fill_null(0.0)

    symbols = [c for c in piv_weight.columns if c != "open_time"]
    times   = piv_weight["open_time"].to_numpy()
    w_mat   = piv_weight.select(symbols).to_numpy()
    r_mat   = piv_fwd.select(symbols).to_numpy()

    gross_returns = np.sum(w_mat * r_mat, axis=1)

    # Turnover fee
    turnover = np.zeros(len(w_mat))
    turnover[0] = np.sum(np.abs(w_mat[0]))
    turnover[1:] = np.sum(np.abs(w_mat[1:] - w_mat[:-1]), axis=1)
    fee_drag = turnover * MAKER_FEE

    # Funding drag: accumulate ALL 8h settlements in each [t, t+reb] window
    funding_drag = np.zeros(len(w_mat))
    if not funding_df.is_empty():
        reb_ms = reb * 3600 * 1000
        for s_idx, sym in enumerate(symbols):
            sym_events = funding_df.filter(pl.col("symbol") == sym)
            if sym_events.is_empty():
                continue
            ev_times = sym_events["funding_time"].to_numpy()
            ev_rates = sym_events["funding_rate"].to_numpy()

            for i, t in enumerate(times):
                # Events strictly in [t, t + reb_ms)
                mask = (ev_times >= t) & (ev_times < t + reb_ms)
                if not mask.any():
                    continue
                total_rate = ev_rates[mask].sum()
                # P&L from funding = -weight * total_rate
                funding_drag[i] += -w_mat[i, s_idx] * total_rate

    net_returns = gross_returns - fee_drag + funding_drag

    # Per-asset contribution (gross only, before costs)
    contrib = {sym: float(np.sum(w_mat[:, i] * r_mat[:, i]))
               for i, sym in enumerate(symbols)}

    periods_df = pl.DataFrame({
        "open_time":    times,
        "gross_ret":    gross_returns,
        "fee_drag":     fee_drag,
        "funding_drag": funding_drag,
        "net_ret":      net_returns,
    })
    return periods_df, contrib


# ─── Metrics ─────────────────────────────────────────────────────────────────

def metrics(net_returns: np.ndarray, reb_h: int) -> dict:
    if len(net_returns) < 2:
        return {"n": 0, "sharpe": 0.0, "ann_ret": 0.0, "max_dd": 0.0,
                "win_rate": 0.0, "cum_ret": 0.0}

    equity = np.cumprod(1.0 + net_returns)
    watermark = np.maximum.accumulate(equity)
    drawdown = (equity / watermark) - 1.0

    periods_per_year = (24 / reb_h) * 365
    years = len(net_returns) / periods_per_year

    mean_r = float(np.mean(net_returns))
    std_r  = float(np.std(net_returns))
    sharpe = (mean_r / std_r) * np.sqrt(periods_per_year) if std_r > 0 else 0.0

    return {
        "n":        len(net_returns),
        "sharpe":   sharpe,
        "ann_ret":  ((equity[-1] ** (1 / years)) - 1.0) * 100 if years > 0 else 0.0,
        "max_dd":   float(np.min(drawdown)) * 100,
        "win_rate": float(np.mean(net_returns > 0)) * 100,
        "cum_ret":  (equity[-1] - 1.0) * 100,
    }


def print_metrics(label: str, m: dict):
    print(f"  {label:<22}  n={m['n']:>5}  "
          f"Sharpe={m['sharpe']:>6.2f}  "
          f"AnnRet={m['ann_ret']:>7.2f}%  "
          f"MaxDD={m['max_dd']:>7.2f}%  "
          f"WinRate={m['win_rate']:>5.1f}%")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    TOXIC_ASSETS = ["LTCUSDT", "ATOMUSDT", "DOTUSDT"]
    
    files = sorted(glob.glob(DATA_GLOB))
    print(f"[*] Loading asset files (Filtering toxic assets)...")
    
    symbols = []
    dfs = []
    for f in files:
        sym = f.replace("\\", "/").split("/")[-1].replace("_1h.parquet", "")
        
        # SKIP THE DINOSAUR COINS
        if sym in TOXIC_ASSETS:
            print(f"  [-] Dropping {sym} from universe.")
            continue
            
        symbols.append(sym)
        df = pl.read_parquet(f).with_columns(pl.lit(sym).alias("symbol"))
        dfs.append(df)
    df = pl.concat(dfs).sort(["symbol", "open_time"])
    print(f"    Total bars: {len(df):,}")

    funding_df = fetch_funding_df(symbols)

    print(f"\n{'=' * 78}")
    print("OUT-OF-SAMPLE VALIDATION")
    print(f"{'=' * 78}\n")

    for cfg in CONFIGS:
        print(f"─── Config {cfg['name']} ───")
        print(f"    lookback={cfg['lookback']}h  rebalance={cfg['rebalance']}h  top_n={cfg['top_n']}")

        periods, contrib = run_config(df, funding_df, cfg)
        n_total = len(periods)
        split_i = int(n_total * TRAIN_FRAC)

        net = periods["net_ret"].to_numpy()
        reb = cfg["rebalance"]

        print_metrics("FULL SAMPLE",  metrics(net, reb))
        print_metrics("TRAIN (first 70%)", metrics(net[:split_i], reb))
        print_metrics("TEST  (last 30%)",  metrics(net[split_i:], reb))

        # Per-asset contribution (full sample, gross P&L)
        print(f"\n  Per-asset contribution (gross, full sample):")
        sorted_c = sorted(contrib.items(), key=lambda kv: -abs(kv[1]))
        total_abs = sum(abs(v) for v in contrib.values()) or 1.0
        for sym, c in sorted_c:
            pct = 100 * c / total_abs
            bar = "█" * int(abs(pct) / 2)
            sign = "+" if c >= 0 else "-"
            print(f"    {sym:<10} {sign}{abs(c):>8.4f}  ({pct:>+6.1f}%)  {bar}")

        print()


if __name__ == "__main__":
    main()