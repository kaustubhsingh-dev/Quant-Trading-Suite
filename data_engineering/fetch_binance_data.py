"""
fetch_binance_data.py — Direct ingestion pipeline from data.binance.vision.

Downloads official Binance Futures (UM) historical data:
  - Klines:    Monthly zips (fast — 12 files per year instead of 365)
  - aggTrades: Daily zips (only when needed for specific sweep windows)

Features:
  - Retry with exponential backoff
  - 404 handling (distinguishes "no data" from "network failure")
  - Disk space guard
  - User-Agent (prevents sporadic 403s from S3)
  - Idempotent — skips if parquet already exists

Usage:
    python fetch_binance_data.py --mode klines --symbol BTCUSDT --interval 1m --months 12
    python fetch_binance_data.py --mode aggtrades --symbol BTCUSDT --date 2024-05-01
    python fetch_binance_data.py --mode aggtrades --symbol BTCUSDT --start 2024-05-01 --days 7
"""

import os
import sys
import time
import shutil
import argparse
import zipfile
import urllib.request
import urllib.error
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

# ─── Config ──────────────────────────────────────────────────────────────────

DATA_DIR = "data_store"
BASE_URL = "https://data.binance.vision/data/futures/um"
MIN_FREE_GB = 5.0

USER_AGENT = "Mozilla/5.0 (compatible; quant-data-fetcher/1.0)"

KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "count", "taker_buy_volume",
    "taker_buy_quote_volume", "ignore",
]

AGGTRADE_COLUMNS = [
    "agg_trade_id", "price", "quantity", "first_trade_id",
    "last_trade_id", "transact_time", "is_buyer_maker",
]

def csv_has_header(path: str) -> bool:
    """Return True if the first line of a CSV is a header.
    
    Binance kline CSVs started including a header row around 2023-07.
    Pre-2023 files do not. Detect by checking whether the first
    column of the first line is numeric.
    """
    with open(path, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()
        if not first_line:
            return False
        first_field = first_line.split(",")[0].strip()
        return not first_field.lstrip("-").isdigit()

# ─── Utilities ───────────────────────────────────────────────────────────────

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def check_disk_space(min_gb: float = MIN_FREE_GB):
    free_gb = shutil.disk_usage(DATA_DIR if os.path.exists(DATA_DIR) else ".").free / 1e9
    if free_gb < min_gb:
        raise SystemExit(f"[FATAL] Only {free_gb:.1f} GB free — need {min_gb} GB. Aborting.")


def setup_user_agent():
    """Install a global URL opener with a User-Agent header."""
    opener = urllib.request.build_opener()
    opener.addheaders = [("User-Agent", USER_AGENT)]
    urllib.request.install_opener(opener)


def download_with_retry(url: str, dest: str, max_retries: int = 3) -> str:
    """
    Returns one of:
      "ok"      — download succeeded
      "missing" — 404, no data for this period (not a failure)
      "error"   — network/other failure after retries
    """
    for attempt in range(max_retries):
        try:
            urllib.request.urlretrieve(url, dest)
            return "ok"
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return "missing"
            if attempt == max_retries - 1:
                print(f"  [-] HTTP {e.code} after {max_retries} attempts: {url}")
                return "error"
            wait = 2 ** attempt
            print(f"  [!] HTTP {e.code}, retry {attempt+1} in {wait}s")
            time.sleep(wait)
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"  [-] Failed after {max_retries} attempts: {e}")
                return "error"
            wait = 2 ** attempt
            print(f"  [!] {type(e).__name__}, retry {attempt+1} in {wait}s")
            time.sleep(wait)
    return "error"


def extract_single_csv(zip_path: str, extract_to: str) -> str:
    """Extract the first CSV from a zip and return its path."""
    with zipfile.ZipFile(zip_path, "r") as z:
        csv_name = z.namelist()[0]
        z.extract(csv_name, extract_to)
    return os.path.join(extract_to, csv_name)


# ─── Kline Downloader (Monthly) ──────────────────────────────────────────────

def download_monthly_klines(symbol: str, interval: str, month_str: str) -> str:
    """
    month_str = 'YYYY-MM'. Downloads one month of klines.
    Example URL: .../monthly/klines/BTCUSDT/1m/BTCUSDT-1m-2024-05.zip
    """
    out_dir = f"{DATA_DIR}/klines/{symbol}_{interval}"
    ensure_dir(out_dir)

    filename = f"{symbol}-{interval}-{month_str}.zip"
    url = f"{BASE_URL}/monthly/klines/{symbol}/{interval}/{filename}"
    zip_path = os.path.join(DATA_DIR, filename)
    parquet_path = f"{out_dir}/{symbol}_{interval}_{month_str}.parquet"

    if os.path.exists(parquet_path):
        print(f"  [=] Already have {parquet_path}")
        return parquet_path

    status = download_with_retry(url, zip_path)

    if status == "missing":
        print(f"  [~] No data for {month_str} (404)")
        return ""

    if status == "error":
        return ""

    try:
        extracted_csv = extract_single_csv(zip_path, DATA_DIR)

        has_header = csv_has_header(extracted_csv)
        df = (
            pl.read_csv(extracted_csv, has_header=has_header, new_columns=KLINE_COLUMNS)
            .select([
                pl.col("open_time").cast(pl.Int64),
                pl.col("open").cast(pl.Float64),
                pl.col("high").cast(pl.Float64),
                pl.col("low").cast(pl.Float64),
                pl.col("close").cast(pl.Float64),
                pl.col("volume").cast(pl.Float64),
                pl.col("taker_buy_volume").cast(pl.Float64),
            ])
            .with_columns(
                pl.from_epoch(pl.col("open_time"), time_unit="ms").alias("time")
            )
        )

        df.write_parquet(parquet_path)
        os.remove(extracted_csv)
        os.remove(zip_path)
        print(f"  [+] Saved {parquet_path} ({len(df):,} rows)")
        return parquet_path

    except Exception as e:
        print(f"  [-] Parse failed for {month_str}: {e}")
        for p in (locals().get("extracted_csv"), zip_path):
            if p and os.path.exists(p):
                os.remove(p)
        return ""


def download_klines_range(symbol: str, interval: str, start_month: str, months: int):
    """Download N consecutive months starting at start_month (YYYY-MM)."""
    y, m = map(int, start_month.split("-"))
    for i in range(months):
        cur = datetime(y + (m - 1 + i) // 12, ((m - 1 + i) % 12) + 1, 1)
        month_str = cur.strftime("%Y-%m")
        print(f"[{i+1}/{months}] {symbol} {interval} {month_str}")
        download_monthly_klines(symbol, interval, month_str)
        check_disk_space()


# ─── aggTrades Downloader (Daily) ────────────────────────────────────────────

def download_daily_aggtrades(symbol: str, date_str: str) -> str:
    """
    date_str = 'YYYY-MM-DD'. Downloads one day of aggTrades.
    Example URL: .../daily/aggTrades/BTCUSDT/BTCUSDT-aggTrades-2024-05-01.zip
    """
    out_dir = f"{DATA_DIR}/aggTrades/{symbol}"
    ensure_dir(out_dir)

    filename = f"{symbol}-aggTrades-{date_str}.zip"
    url = f"{BASE_URL}/daily/aggTrades/{symbol}/{filename}"
    zip_path = os.path.join(DATA_DIR, filename)
    parquet_path = f"{out_dir}/{symbol}_aggTrades_{date_str}.parquet"

    if os.path.exists(parquet_path):
        print(f"  [=] Already have {parquet_path}")
        return parquet_path

    status = download_with_retry(url, zip_path)

    if status == "missing":
        print(f"  [~] No aggTrades for {date_str} (404)")
        return ""

    if status == "error":
        return ""

    try:
        extracted_csv = extract_single_csv(zip_path, DATA_DIR)

        has_header = csv_has_header(extracted_csv)
        df = (
            pl.read_csv(extracted_csv, has_header=has_header, new_columns=AGGTRADE_COLUMNS)
            .select([
                pl.col("agg_trade_id").cast(pl.Int64),
                pl.col("transact_time").cast(pl.Int64),
                pl.col("price").cast(pl.Float64),
                pl.col("quantity").cast(pl.Float64),
                pl.col("is_buyer_maker").cast(pl.Boolean),
            ])
            .with_columns(
                pl.from_epoch(pl.col("transact_time"), time_unit="ms").alias("time")
            )
        )

        df.write_parquet(parquet_path)
        os.remove(extracted_csv)
        os.remove(zip_path)
        print(f"  [+] Saved {parquet_path} ({len(df):,} rows)")
        return parquet_path

    except Exception as e:
        print(f"  [-] Parse failed for {date_str}: {e}")
        for p in (locals().get("extracted_csv"), zip_path):
            if p and os.path.exists(p):
                os.remove(p)
        return ""


def download_aggtrades_range(symbol: str, start_date: str, days: int):
    """Download N consecutive days starting at start_date (YYYY-MM-DD)."""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    for i in range(days):
        cur = (start + timedelta(days=i)).strftime("%Y-%m-%d")
        print(f"[{i+1}/{days}] {symbol} aggTrades {cur}")
        download_daily_aggtrades(symbol, cur)
        check_disk_space()


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Binance Futures historical data fetcher")
    parser.add_argument("--mode", choices=["klines", "aggtrades"], required=True)
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--interval", default="1m", help="Kline interval (klines mode only)")

    # Klines options
    parser.add_argument("--start-month", default=None,
                        help="YYYY-MM for klines mode")
    parser.add_argument("--months", type=int, default=12,
                        help="Number of months to download (klines mode)")

    # aggTrades options
    parser.add_argument("--date", default=None,
                        help="Single date YYYY-MM-DD (aggtrades mode)")
    parser.add_argument("--start", default=None,
                        help="Start date YYYY-MM-DD for a range (aggtrades mode)")
    parser.add_argument("--days", type=int, default=1,
                        help="Number of days starting from --start (aggtrades mode)")

    args = parser.parse_args()

    ensure_dir(DATA_DIR)
    setup_user_agent()
    check_disk_space()

    if args.mode == "klines":
        if not args.start_month:
            raise SystemExit("--start-month required for klines mode")
        print(f"=== DOWNLOADING {args.symbol} {args.interval} KLINES ===")
        download_klines_range(args.symbol, args.interval, args.start_month, args.months)

    elif args.mode == "aggtrades":
        if args.date:
            print(f"=== DOWNLOADING {args.symbol} aggTrades for {args.date} ===")
            download_daily_aggtrades(args.symbol, args.date)
        elif args.start:
            print(f"=== DOWNLOADING {args.symbol} aggTrades, {args.days} days from {args.start} ===")
            download_aggtrades_range(args.symbol, args.start, args.days)
        else:
            raise SystemExit("Provide --date or --start for aggtrades mode")

    print("\nDone.")


if __name__ == "__main__":
    main()