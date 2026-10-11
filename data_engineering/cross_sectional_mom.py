"""
cross_sectional_mom.py — Institutional Delta-Neutral Relative Strength (Macro Trend)
"""

import glob
import numpy as np
import polars as pl

DATA_GLOB    = "data_store/*_1h.parquet"
LOOKBACK_H   = 336      # 14-day trailing momentum (Captures true narrative rotation)
REBALANCE_H  = 24       # Rebalance daily
TOP_N        = 3        # Long top 3, Short bottom 3
MAKER_FEE    = 0.0002   # 0.02% Maker fee per leg on Delta Exchange

def main():
    files = sorted(glob.glob(DATA_GLOB))
    print(f"[*] Loading {len(files)} asset files...")
    
    dfs = []
    for f in files:
        sym = f.replace("\\", "/").split("/")[-1].replace("_1h.parquet", "")
        df = pl.read_parquet(f).with_columns(pl.lit(sym).alias("symbol"))
        dfs.append(df)
        
    df = pl.concat(dfs).sort(["symbol", "open_time"])
    
    # 1. Hourly Volatility & Trailing Return
    df = df.with_columns([
        (pl.col("close") / pl.col("close").shift(1) - 1.0).over("symbol").alias("hourly_ret")
    ])
    
    df = df.with_columns([
        pl.col("hourly_ret").rolling_std(window_size=LOOKBACK_H).over("symbol").alias("vol_lookback"),
        (pl.col("close") / pl.col("close").shift(LOOKBACK_H) - 1.0).over("symbol").alias("ret_lookback")
    ])
    
    # 2. Volatility-Adjusted Z-Score
    df = df.with_columns([
        (pl.col("ret_lookback") / pl.col("vol_lookback").clip(lower_bound=0.001)).alias("mom_score")
    ])
    
    # 3. Forward Period Return
    df = df.with_columns([
        (pl.col("close").shift(-REBALANCE_H) / pl.col("close") - 1.0).over("symbol").alias("fwd_ret")
    ]).drop_nulls(subset=["mom_score", "fwd_ret"])
    
    # 4. Extract Rebalance Timestamps
    rebal_df = df.filter((pl.col("open_time") % (REBALANCE_H * 3600 * 1000)) == 0)
    
    rebal_df = rebal_df.with_columns([
        pl.col("symbol").count().over("open_time").alias("active_coins")
    ]).filter(pl.col("active_coins") >= 10)
    
    rebal_df = rebal_df.with_columns([
        pl.col("mom_score").rank(descending=True).over("open_time").alias("rank")
    ])
    
    # Assign target weights
    w_leg = 0.5 / TOP_N
    rebal_df = rebal_df.with_columns([
        pl.when(pl.col("rank") <= TOP_N).then(pl.lit(w_leg))
          .when(pl.col("rank") > pl.col("active_coins") - TOP_N).then(pl.lit(-w_leg))
          .otherwise(pl.lit(0.0)).alias("target_weight")
    ])
    
    # 5. Compute Real Portfolio Turnover and Return (Fixed deprecation warnings)
    piv_weight = rebal_df.pivot(index="open_time", on="symbol", values="target_weight").fill_null(0.0)
    piv_fwd    = rebal_df.pivot(index="open_time", on="symbol", values="fwd_ret").fill_null(0.0)
    
    symbols = [c for c in piv_weight.columns if c != "open_time"]
    w_mat = piv_weight.select(symbols).to_numpy()
    r_mat = piv_fwd.select(symbols).to_numpy()
    
    # Gross returns per period
    gross_returns = np.sum(w_mat * r_mat, axis=1)
    
    # Compute turnover
    turnover = np.zeros(len(w_mat))
    turnover[0] = np.sum(np.abs(w_mat[0]))
    turnover[1:] = np.sum(np.abs(w_mat[1:] - w_mat[:-1]), axis=1)
    
    # Real friction
    fee_drag = turnover * MAKER_FEE
    net_returns = gross_returns - fee_drag
    
    # 6. Performance Metrics
    equity = np.cumprod(1.0 + net_returns)
    watermark = np.maximum.accumulate(equity)
    drawdown = (equity / watermark) - 1.0
    
    total_periods = len(net_returns)
    periods_per_year = (24 / REBALANCE_H) * 365
    years = total_periods / periods_per_year
    
    cum_net_ret = (equity[-1] - 1.0) * 100
    ann_net_ret = ((equity[-1] ** (1 / years)) - 1.0) * 100 if years > 0 else 0
    win_rate = (np.sum(net_returns > 0) / total_periods) * 100
    max_dd = np.min(drawdown) * 100
    
    mean_r = np.mean(net_returns)
    std_r = np.std(net_returns)
    sharpe = (mean_r / std_r) * np.sqrt(periods_per_year) if std_r > 0 else 0
    avg_turnover = np.mean(turnover[1:]) * 100 
    
    print("\n" + "=" * 60)
    print("CROSS-SECTIONAL MOMENTUM (MACRO HORIZON)")
    print(f"Lookback: {LOOKBACK_H}h | Rebalance: {REBALANCE_H}h | Top/Bottom {TOP_N}")
    print("=" * 60)
    print(f"  Total Rebalance Periods: {total_periods:,}")
    print(f"  Average Turnover/Period: {avg_turnover:.1f}%")
    print(f"  Win Rate ({REBALANCE_H}h periods):   {win_rate:.2f}%")
    print(f"  Max Drawdown:            {max_dd:.2f}%")
    print(f"  Sharpe Ratio:            {sharpe:.2f}")
    print(f"  Cumulative Net Return:   {cum_net_ret:.2f}%")
    print(f"  Annualized Net Return:   {ann_net_ret:.2f}%")
    print("=" * 60)

if __name__ == "__main__":
    main()