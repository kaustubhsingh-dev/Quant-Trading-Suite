"""
cvd_engine.py — Streaming Order Flow Feature Extraction.

Loads valid structural touches, downloads daily aggTrades on the fly,
extracts Cumulative Volume Delta (CVD) metrics, and cleans up disk space.

Usage:
    python cvd_engine.py
"""

import os
import time
import zipfile
import urllib.request
import numpy as np
import polars as pl

TOUCHES_PATH = "data_store/valid_touches.parquet"
CVD_OUTPUT   = "data_store/touches_with_cvd.parquet"
TEMP_DIR     = "data_store/temp_agg"
BASE_URL     = "https://data.binance.vision/data/futures/um/daily/aggTrades/BTCUSDT"

os.makedirs(TEMP_DIR, exist_ok=True)


def csv_has_header(path: str) -> bool:
    """Binance aggTrades files have a header row from ~2023-07 onward."""
    with open(path, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()
        if not first_line:
            return False
        first_field = first_line.split(",")[0].strip()
        return not first_field.lstrip("-").isdigit()


def get_daily_aggtrades(date_str: str) -> pl.DataFrame:
    """Downloads, parses, and deletes the raw zip to save disk space."""
    filename = f"BTCUSDT-aggTrades-{date_str}.zip"
    url = f"{BASE_URL}/{filename}"
    zip_path = os.path.join(TEMP_DIR, filename)

    try:
        urllib.request.urlretrieve(url, zip_path)
    except Exception as e:
        print(f"  [!] Download failed for {date_str}: {e}")
        if os.path.exists(zip_path):
            os.remove(zip_path)
        return pl.DataFrame()

    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            csv_name = z.namelist()[0]
            z.extract(csv_name, TEMP_DIR)
        extracted_csv = os.path.join(TEMP_DIR, csv_name)

        has_header = csv_has_header(extracted_csv)

        # is_buyer_maker = True  -> market SELL (taker sell)
        # is_buyer_maker = False -> market BUY  (taker buy)
        df = (
            pl.read_csv(
                extracted_csv,
                has_header=has_header,
                new_columns=["id", "price", "qty", "f_id", "l_id", "time", "is_buyer_maker"],
            )
            .select([
                pl.col("time").cast(pl.Int64),
                pl.col("qty").cast(pl.Float64),
                pl.col("is_buyer_maker").cast(pl.Boolean),
            ])
            .with_columns(
                pl.when(pl.col("is_buyer_maker"))
                  .then(-pl.col("qty"))
                  .otherwise(pl.col("qty"))
                  .alias("delta")
            )
            .sort("time")
        )

        os.remove(extracted_csv)
        os.remove(zip_path)
        return df

    except Exception as e:
        print(f"  [!] Parse failed for {date_str}: {e}")
        for p in (zip_path,):
            if os.path.exists(p):
                os.remove(p)
        return pl.DataFrame()


def main():
    if not os.path.exists(TOUCHES_PATH):
        raise FileNotFoundError(
            f"Missing {TOUCHES_PATH}. Run structure_engine.py first."
        )

    print(f"[*] Loading touches from {TOUCHES_PATH}")
    touches = pl.read_parquet(TOUCHES_PATH)

    touches = touches.with_columns(
        pl.from_epoch("touch_time", time_unit="ms")
          .dt.strftime("%Y-%m-%d")
          .alias("date_str")
    )

    unique_dates = touches["date_str"].unique().sort().to_list()
    print(f"[*] {len(touches):,} touches across {len(unique_dates)} days.\n")

    enriched_touches = []
    t0 = time.perf_counter()

    for i, date_str in enumerate(unique_dates):
        daily_touches = touches.filter(pl.col("date_str") == date_str)
        print(f"[{i+1}/{len(unique_dates)}] {date_str} | {len(daily_touches)} touches")

        tape = get_daily_aggtrades(date_str)
        if tape.is_empty():
            continue

        tape_times = tape["time"].to_numpy()
        tape_delta = tape["delta"].to_numpy()
        tape_vol   = tape["qty"].to_numpy()

        for row in daily_touches.iter_rows(named=True):
            tt = row["touch_time"]

            t_pre_start = tt - (5 * 60 * 1000)   # 5 minutes before
            t_post_end  = tt + (1 * 60 * 1000)   # 1 minute after

            idx_pre  = np.searchsorted(tape_times, t_pre_start)
            idx_tt   = np.searchsorted(tape_times, tt)
            idx_post = np.searchsorted(tape_times, t_post_end)

            if idx_tt > idx_pre:
                row["vol_pre_5m"] = float(np.sum(tape_vol[idx_pre:idx_tt]))
                row["cvd_pre_5m"] = float(np.sum(tape_delta[idx_pre:idx_tt]))
            else:
                row["vol_pre_5m"] = 0.0
                row["cvd_pre_5m"] = 0.0

            if idx_post > idx_tt:
                row["vol_post_1m"] = float(np.sum(tape_vol[idx_tt:idx_post]))
                row["cvd_post_1m"] = float(np.sum(tape_delta[idx_tt:idx_post]))
            else:
                row["vol_post_1m"] = 0.0
                row["cvd_post_1m"] = 0.0

            enriched_touches.append(row)

        # free memory between days
        del tape, tape_times, tape_delta, tape_vol

    elapsed = (time.perf_counter() - t0) / 60.0

    if not enriched_touches:
        print("\n[!] No touches enriched. Nothing to write.")
        return

    out_df = pl.DataFrame(enriched_touches).drop("date_str")
    out_df.write_parquet(CVD_OUTPUT)

    print(f"\n[+] Enriched {len(out_df):,} touches in {elapsed:.1f} minutes.")
    print(f"[+] Saved to {CVD_OUTPUT}")


if __name__ == "__main__":
    main()