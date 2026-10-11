"""
optimize_and_stress_test.py — Grid Search Optimization + Real Funding Rate Integration
"""

import glob
import urllib.request
import json
import numpy as np
import polars as pl
from datetime import datetime

DATA_GLOB    = "data_store/*_1h.parquet"
MAKER_FEE    = 0.0002   # 0.02% per leg

def fetch_binance_funding_rates(symbols):
    """Fetches historical funding rates from Binance Public API for the basket."""
    funding_data = {}
    for sym in symbols:
        url = f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={sym}&limit=1000"
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response:
                data = json.loads(response.read().decode())
                # Format: list of dicts with 'fundingTime' and 'fundingRate'
                rates = {(int(item['fundingTime'])): float(item['fundingRate']) for item in data}
                funding_data[sym] = rates
        except Exception as e:
            print(f"  [!] Could not fetch funding for {sym}: {e}. Using neutral baseline.")
            funding_data[sym] = {}
    return funding_data

def main():
    files = sorted(glob.glob(DATA_GLOB))
    print(f"[*] Loading {len(files)} asset files for grid search & funding stress test...")
    
    symbols = []
    dfs = []
    for f in files:
        sym = f.replace("\\", "/").split("/")[-1].replace("_1h.parquet", "")
        symbols.append(sym)
        df = pl.read_parquet(f).with_columns(pl.lit(sym).alias("symbol"))
        dfs.append(df)
        
    df = pl.concat(dfs).sort(["symbol", "open_time"])
    
    print("[*] Downloading live funding rate histories from Binance...")
    funding_history = fetch_binance_funding_rates(symbols)

    # Grid search parameter spaces
    lookbacks = [168, 336, 504]      # 7d, 14d, 21d
    rebalances = [12, 24, 48]        # 12h, 24h, 48h
    top_ns = [2, 3, 5]               # Portfolio concentration

    best_sharpe = -999
    best_params = None
    results_grid = []

    print("\n[*] Running institutional grid search across parameter space...")
    
    for lb in lookbacks:
        # Precompute volatility and momentum for this lookback
        df_opt = df.with_columns([
            (pl.col("close") / pl.col("close").shift(1) - 1.0).over("symbol").alias("hourly_ret")
        ])
        df_opt = df_opt.with_columns([
            pl.col("hourly_ret").rolling_std(window_size=lb).over("symbol").alias("vol_lb"),
            (pl.col("close") / pl.col("close").shift(lb) - 1.0).over("symbol").alias("ret_lb")
        ]).with_columns([
            (pl.col("ret_lb") / pl.col("vol_lb").clip(lower_bound=0.001)).alias("mom_score")
        ])

        for reb in rebalances:
            df_reb = df_opt.with_columns([
                (pl.col("close").shift(-reb) / pl.col("close") - 1.0).over("symbol").alias("fwd_ret")
            ]).drop_nulls(subset=["mom_score", "fwd_ret"])

            rebal_df = df_reb.filter((pl.col("open_time") % (reb * 3600 * 1000)) == 0)
            rebal_df = rebal_df.with_columns([
                pl.col("symbol").count().over("open_time").alias("active_coins")
            ]).filter(pl.col("active_coins") >= 10)

            rebal_df = rebal_df.with_columns([
                pl.col("mom_score").rank(descending=True).over("open_time").alias("rank")
            ])

            for top_n in top_ns:
                w_leg = 0.5 / top_n
                r_df = rebal_df.with_columns([
                    pl.when(pl.col("rank") <= top_n).then(pl.lit(w_leg))
                      .when(pl.col("rank") > pl.col("active_coins") - top_n).then(pl.lit(-w_leg))
                      .otherwise(pl.lit(0.0)).alias("target_weight")
                ])

                piv_weight = r_df.pivot(index="open_time", on="symbol", values="target_weight").fill_null(0.0)
                piv_fwd    = r_df.pivot(index="open_time", on="symbol", values="fwd_ret").fill_null(0.0)

                active_syms = [c for c in piv_weight.columns if c != "open_time"]
                w_mat = piv_weight.select(active_syms).to_numpy()
                r_mat = piv_fwd.select(active_syms).to_numpy()
                times = piv_weight["open_time"].to_numpy()

                gross_returns = np.sum(w_mat * r_mat, axis=1)

                # Turnover & Maker fees
                turnover = np.zeros(len(w_mat))
                turnover[0] = np.sum(np.abs(w_mat[0]))
                turnover[1:] = np.sum(np.abs(w_mat[1:] - w_mat[:-1]), axis=1)
                fee_drag = turnover * MAKER_FEE

                # Funding rate cashflow integration (8h settlement intervals)
                # If short (weight < 0), we receive funding when rate > 0. If long (weight > 0), we pay when rate > 0.
                funding_drag = np.zeros(len(w_mat))
                for i, t in enumerate(times):
                    # Match closest funding timestamp (every 8 hours)
                    period_funding = 0.0
                    for s_idx, sym in enumerate(active_syms):
                        weight = w_mat[i, s_idx]
                        if weight == 0: continue
                        sym_rates = funding_history.get(sym, {})
                        # Find closest funding rate timestamp
                        if sym_rates:
                            closest_time = min(sym_rates.keys(), key=lambda x: abs(x - t))
                            if abs(closest_time - t) <= 4 * 3600 * 1000: # within 4 hours
                                rate = sym_rates[closest_time]
                                # Long pays funding if rate > 0. Short receives funding if rate > 0.
                                # P&L impact on position = -1 * weight * rate (since weight is positive for long, negative for short)
                                period_funding += (-weight * rate)
                    funding_drag[i] = period_funding

                net_returns = gross_returns - fee_drag + funding_drag

                # Performance metrics
                equity = np.cumprod(1.0 + net_returns)
                watermark = np.maximum.accumulate(equity)
                drawdown = (equity / watermark) - 1.0
                
                total_periods = len(net_returns)
                periods_per_year = (24 / reb) * 365
                years = total_periods / periods_per_year
                if years <= 0: continue

                ann_ret = ((equity[-1] ** (1 / years)) - 1.0) * 100
                max_dd = np.min(drawdown) * 100
                mean_r = np.mean(net_returns)
                std_r = np.std(net_returns)
                sharpe = (mean_r / std_r) * np.sqrt(periods_per_year) if std_r > 0 else 0

                results_grid.append({
                    "lookback": lb, "rebalance": reb, "top_n": top_n,
                    "sharpe": sharpe, "ann_ret": ann_ret, "max_dd": max_dd
                })

                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_params = (lb, reb, top_n)

    print("\n" + "=" * 65)
    print("GRID SEARCH OPTIMIZATION & FUNDING RATE STRESS TEST RESULTS")
    print("=" * 65)
    print(f"  Best Parameters Found:")
    print(f"    -> Lookback Window: {best_params[0]} hours")
    print(f"    -> Rebalance Rate:  {best_params[1]} hours")
    print(f"    -> Portfolio Size:  Top/Bottom {best_params[2]}")
    print(f"  Optimized Sharpe Ratio:    {best_sharpe:.2f}")
    print("=" * 65)

    # Print top 3 configurations
    sorted_results = sorted(results_grid, key=lambda x: x["sharpe"], reverse=True)[:3]
    for idx, res in enumerate(sorted_results, 1):
        print(f"  Rank {idx}: Lookback={res['lookback']}h | Rebalance={res['rebalance']}h | TopN={res['top_n']} "
              f"| Sharpe={res['sharpe']:.2f} | Ann.Ret={res['ann_ret']:.2f}% | MaxDD={res['max_dd']:.2f}%")
    print("=" * 65)

if __name__ == "__main__":
    main()