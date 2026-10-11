import os
import zipfile
import urllib.request
import polars as pl

SYMBOLS = ["ETHUSDT", "SOLUSDT", "BNBUSDT", "DOGEUSDT"]
MONTHS = [
    "2023-10", "2023-11", "2023-12",
    "2024-01", "2024-02", "2024-03",
    "2024-04", "2024-05", "2024-06",
    "2024-07", "2024-08", "2024-09",
]
BASE_URL = "https://data.binance.vision/data/futures/um/monthly/klines"
SAVE_DIR = "data_store/klines_1h"

os.makedirs(SAVE_DIR, exist_ok=True)

def csv_has_header(path: str) -> bool:
    """Detect if the Binance CSV has a header row."""
    with open(path, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()
        if not first_line:
            return False
        first_field = first_line.split(",")[0].strip()
        return not first_field.lstrip("-").isdigit()

for sym in SYMBOLS:
    print(f"\n[*] Fetching 1h klines for {sym}...")
    dfs = []
    for m in MONTHS:
        fname = f"{sym}-1h-{m}.zip"
        url = f"{BASE_URL}/{sym}/1h/{fname}"
        local_zip = os.path.join(SAVE_DIR, fname)
        try:
            urllib.request.urlretrieve(url, local_zip)
            with zipfile.ZipFile(local_zip, "r") as z:
                csv_name = z.namelist()[0]
                z.extract(csv_name, SAVE_DIR)
            csv_path = os.path.join(SAVE_DIR, csv_name)
            
            # Dynamically read header status
            has_h = csv_has_header(csv_path)
            df_temp = pl.read_csv(csv_path, has_header=has_h)
            
            # Extract by column index to ignore changing header names
            cols = df_temp.columns
            sub = df_temp.select([
                pl.col(cols[0]).cast(pl.Int64).alias("open_time"),
                pl.col(cols[1]).cast(pl.Float64).alias("open"),
                pl.col(cols[2]).cast(pl.Float64).alias("high"),
                pl.col(cols[3]).cast(pl.Float64).alias("low"),
                pl.col(cols[4]).cast(pl.Float64).alias("close"),
                pl.col(cols[5]).cast(pl.Float64).alias("volume"),
            ])
            
            dfs.append(sub)
            os.remove(csv_path)
            os.remove(local_zip)
        except Exception as e:
            print(f"  [!] Failed {fname}: {e}")
            if os.path.exists(local_zip):
                os.remove(local_zip)
                
    if dfs:
        full_sym = pl.concat(dfs).sort("open_time")
        out_path = f"data_store/{sym}_1h.parquet"
        full_sym.write_parquet(out_path)
        print(f"[+] Saved {len(full_sym):,} bars to {out_path}")