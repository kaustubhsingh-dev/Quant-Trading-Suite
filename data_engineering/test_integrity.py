import polars as pl, glob
files = sorted(glob.glob("data_store/klines/BTCUSDT_1m/*.parquet"))
df = pl.concat([pl.read_parquet(f) for f in files]).sort("open_time")
print(f"Months: {len(files)}")
print(f"Rows:   {len(df):,}")
print(f"Range:  {df['time'].min()} → {df['time'].max()}")
gaps = df.select(pl.col("open_time").diff().alias("d")).filter(
    (pl.col("d") != 60_000) & pl.col("d").is_not_null()
)
print(f"Gaps:   {len(gaps)}")