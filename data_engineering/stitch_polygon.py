import os
import zipfile
import urllib.request
import polars as pl
from datetime import datetime
from dateutil.relativedelta import relativedelta

BASE_URL = "https://data.binance.vision/data/futures/um/monthly/klines"
SAVE_DIR = "data_store"

def generate_months(start_year=2024, start_month=9, duration_months=22):
    """Generate months from Sept 2024 to mid 2026"""
    months = []
    current_date = datetime(start_year, start_month, 1)
    for _ in range(duration_months):
        months.append(current_date.strftime("%Y-%m"))
        current_date += relativedelta(months=1)
    return months

def csv_has_header(path: str) -> bool:
    with open(path, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()
        if not first_line:
            return False
        first_field = first_line.split(",")[0].strip()
        return not first_field.lstrip("-").isdigit()

def main():
    months = generate_months()
    sym = "POLUSDT"
    print(f"[*] Fetching 1h klines for {sym} (Post-Migration)...")
    dfs = []
    
    for m in months:
        fname = f"{sym}-1h-{m}.zip"
        url = f"{BASE_URL}/{sym}/1h/{fname}"
        local_zip = os.path.join(SAVE_DIR, fname)
        
        try:
            urllib.request.urlretrieve(url, local_zip)
            with zipfile.ZipFile(local_zip, "r") as z:
                csv_name = z.namelist()[0]
                z.extract(csv_name, SAVE_DIR)
            csv_path = os.path.join(SAVE_DIR, csv_name)
            
            has_h = csv_has_header(csv_path)
            df_temp = pl.read_csv(csv_path, has_header=has_h, ignore_errors=True)
            
            cols = df_temp.columns
            sub = df_temp.select([
                pl.col(cols[0]).cast(pl.Int64).alias("open_time"),
                pl.col(cols[1]).cast(pl.Float64).alias("open"),
                pl.col(cols[2]).cast(pl.Float64).alias("high"),
                pl.col(cols[3]).cast(pl.Float64).alias("low"),
                pl.col(cols[4]).cast(pl.Float64).alias("close"),
                pl.col(cols[5]).cast(pl.Float64).alias("volume"),
            ]).drop_nulls()
            
            dfs.append(sub)
            os.remove(csv_path)
            os.remove(local_zip)
        except Exception:
            if os.path.exists(local_zip):
                os.remove(local_zip)
                
    if not dfs:
        print("[!] No POL data found. Skipping.")
        return

    pol_df = pl.concat(dfs).sort("open_time").unique(subset=["open_time"])
    print(f"[+] Downloaded {len(pol_df):,} POL bars.")
    
    matic_path = os.path.join(SAVE_DIR, "MATICUSDT_1h.parquet")
    if os.path.exists(matic_path):
        print("[*] Stitching MATIC and POL into a single synthetic asset...")
        matic_df = pl.read_parquet(matic_path)
        
        stitched_df = pl.concat([matic_df, pol_df]).sort("open_time").unique(subset=["open_time"], keep="last")
        
        out_path = os.path.join(SAVE_DIR, "POL_SYNTHETIC_1h.parquet")
        stitched_df.write_parquet(out_path)
        
        os.remove(matic_path)
        print(f"[+] Successfully saved {len(stitched_df):,} stitched bars to {out_path}")
        print("[+] Deleted old MATICUSDT file to keep universe at exactly 15 assets.")
    else:
        print("[!] MATICUSDT file not found. Could not stitch.")

if __name__ == "__main__":
    main()