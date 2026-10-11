import glob
import polars as pl

DATA_GLOB = "data_store/klines/BTCUSDT_1m/*.parquet"
OUT_FILE = "data_store/BTCUSDT_15m.parquet"

def main():
    files = sorted(glob.glob(DATA_GLOB))
    print(f"Loading {len(files)} 1m files...")
    df = pl.concat([pl.read_parquet(f) for f in files]).sort("open_time")
    
    df = df.with_columns(pl.from_epoch("open_time", time_unit="ms").alias("datetime"))
    
    df_15m = (
        df.group_by_dynamic("datetime", every="15m")
        .agg([
            pl.col("open_time").first(),
            pl.col("open").first(),
            pl.col("high").max(),
            pl.col("low").min(),
            pl.col("close").last(),
            pl.col("volume").sum(),
        ])
    ).drop("datetime")
    
    df_15m.write_parquet(OUT_FILE)
    print(f"Resampled to {len(df_15m):,} 15m bars. Saved to {OUT_FILE}")

if __name__ == "__main__":
    main()