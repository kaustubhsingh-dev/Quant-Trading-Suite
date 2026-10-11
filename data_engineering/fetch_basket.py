import os
import zipfile
import urllib.request
import polars as pl
from datetime import datetime
from dateutil.relativedelta import relativedelta

# Top 15 Highly Liquid Binance/Delta Perpetuals
SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", 
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT", 
    "MATICUSDT", "LTCUSDT", "BCHUSDT", "NEARUSDT", "ATOMUSDT"
]

BASE_URL = "https://data.binance.vision/data/futures/um/monthly/klines"
SAVE_DIR = "data_store"

def generate_months(start_year=2023, start_month=1, duration_months=42):
    """Generates a list of YYYY-MM strings for the last 3.5 years."""
    months = []
    current_date = datetime(start_year, start_month, 1)
    for _ in range(duration_months):
        months.append(current_date.strftime("%Y-%m"))
        current_date += relativedelta(months=1)
    return months

def csv_has_header(path: str) -> bool:
    """Detect if the Binance CSV has a header row."""
    with open(path, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()
        if not first_line:
            return False
        first_field = first_line.split(",")[0].strip()
        return not first_field.lstrip("-").isdigit()

def main():
    months = generate_months()
    print(f"[*] Fetching data from {months[0]} to {months[-1]}")
    
    for sym in SYMBOLS:
        print(f"\n[*] Fetching 1h klines for {sym}...")
        dfs = []
        for m in months:
            fname = f"{sym}-1h-{m}.zip"
            url = f"{BASE_URL}/{sym}/1h/{fname}"
            local_zip = os.path.join(SAVE_DIR, fname)
            
            try:
                # Suppress HTTP errors for future months that don't exist yet
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
                # Clean up failed zips silently (e.g. 404 Not Found for current unclosed month)
                if os.path.exists(local_zip):
                    os.remove(local_zip)
                    
        if dfs:
            full_sym = pl.concat(dfs).sort("open_time").unique(subset=["open_time"])
            out_path = f"{SAVE_DIR}/{sym}_1h.parquet"
            full_sym.write_parquet(out_path)
            print(f"[+] Saved {len(full_sym):,} bars to {out_path}")
        else:
            print(f"[!] No data found for {sym}. Check symbol name.")

if __name__ == "__main__":
    main()