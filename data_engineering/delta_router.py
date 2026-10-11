"""
delta_router.py — Delta Exchange India cross-sectional momentum router.

Config: 504h lookback / 48h rebalance / Top 5 long + Bottom 5 short / 0.5x leverage.
Phase:  DRY_RUN=True (paper). Logs intended orders, no POST sent.
"""

import os
import csv
import json
import time
import hmac
import hashlib
import requests
import polars as pl
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ─── Config ────────────────────────────────────────────────────────────────

BASE_URL      = "https://api.india.delta.exchange"
API_KEY       = os.getenv("DELTA_API_KEY",    "YOUR_API_KEY")
API_SECRET    = os.getenv("DELTA_API_SECRET", "YOUR_API_SECRET")

LOOKBACK_H    = 504
TOP_N         = 5
REBALANCE_H   = 48
LEVERAGE      = 0.5
PAPER_CAPITAL = 200000.0
DRY_RUN       = True

STATE_FILE    = Path("state/router_state.json")
LOG_FILE      = Path("logs/paper_trades.csv")
FUNDING_LOG   = Path("logs/funding_rates.csv")
ORDER_LOG     = Path("logs/order_responses.csv")

REBALANCE_TOL_S = 600

UNIVERSE = [
    "BTCUSD", "ETHUSD", "BNBUSD", "SOLUSD", "XRPUSD",
    "DOGEUSD", "ADAUSD", "AVAXUSD", "LINKUSD", "DOTUSD",
    "POLUSD", "LTCUSD", "BCHUSD", "NEARUSD",
]


# ─── Auth ──────────────────────────────────────────────────────────────────

def _sign(method: str, ts: str, path: str, query: str = "", payload: str = "") -> str:
    data = method + ts + path + query + payload
    return hmac.new(API_SECRET.encode(), data.encode(), hashlib.sha256).hexdigest()


def auth_headers(method: str, path: str, query: str = "", payload: str = "") -> dict:
    ts = str(int(time.time()))
    return {
        "api-key":      API_KEY,
        "timestamp":    ts,
        "signature":    _sign(method, ts, path, query, payload),
        "User-Agent":   "quant-daemon/1.0",
        "Content-Type": "application/json",
    }


def api_get(path: str, query: str = "", signed: bool = False) -> dict:
    url = BASE_URL + path + (("?" + query) if query else "")
    headers = auth_headers("GET", path, query) if signed else {"User-Agent": "quant-daemon/1.0"}
    r = requests.get(url, headers=headers, timeout=15)
    r.raise_for_status()
    return r.json()


# ─── State ─────────────────────────────────────────────────────────────────

def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return {"last_rebalance_ts": 0, "peak_equity": 0.0, "current_equity": 0.0}


def save_state(s: dict):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(s, indent=2))


# ─── Account ───────────────────────────────────────────────────────────────

def fetch_wallet_balance() -> float:
    data = api_get("/v2/wallet/balances", signed=True)
    for a in data.get("result", []):
        if a.get("asset_symbol") == "USDT":
            return float(a.get("available_balance", 0.0) or 0.0)
    return 0.0


def fetch_open_positions() -> dict:
    """
    Delta's /v2/positions endpoint requires product_id or underlying_asset_symbol.
    We loop through the universe and query each symbol individually.

    In DRY_RUN, return empty positions — no real capital is deployed, and the
    paper log only records intended orders.
    """
    if DRY_RUN:
        print("[i] PAPER MODE: assuming zero current positions.")
        return {}

    positions: dict = {}
    failures = 0

    for sym in UNIVERSE:
        # BTCUSD -> BTC, ETHUSD -> ETH, etc.
        underlying = sym.replace("USD", "")
        path = "/v2/positions"
        query = f"underlying_asset_symbol={underlying}"

        try:
            data = api_get(path, query=query, signed=True)
        except Exception as e:
            print(f"    [!] positions {sym}: {e}")
            failures += 1
            continue

        for p in data.get("result", []):
            p_sym = p.get("product_symbol")
            size  = float(p.get("size", 0) or 0)
            if p_sym == sym and size != 0:
                positions[sym] = positions.get(sym, 0.0) + size

    # If EVERY query failed in live mode, abort to prevent duplicate orders
    if failures == len(UNIVERSE):
        raise RuntimeError("All position queries failed in LIVE mode — aborting.")

    return positions


def cancel_all_open_orders() -> bool:
    print("[*] Cancelling all open orders...")
    path = "/v2/orders"
    try:
        headers = auth_headers("DELETE", path)
        res = requests.delete(BASE_URL + path, headers=headers, timeout=15)
        if res.status_code >= 400:
            print(f"    [!] Cancel failed: HTTP {res.status_code} {res.text[:200]}")
            return False
        print("    [+] Open orders cleared.")
        return True
    except Exception as e:
        print(f"    [!] Cancel exception: {e}")
        return False


# ─── Order Submission ──────────────────────────────────────────────────────

def submit_post_only_order(symbol: str, action: str, size: int,
                           price: float, product_id: int) -> Optional[dict]:
    if size < 1:
        print(f"    [!] {symbol}: size {size} < 1, skipping.")
        return None
    if product_id == 0:
        print(f"    [!] {symbol}: product_id unknown, skipping.")
        return None

    path = "/v2/orders"
    payload = {
        "product_id":  product_id,
        "order_type":  "limit_order",
        "size":        size,
        "side":        action.strip().capitalize(),
        "limit_price": f"{price}",
        "post_only":   True,
    }
    payload_str = json.dumps(payload)

    try:
        headers = auth_headers("POST", path, payload=payload_str)
        res = requests.post(BASE_URL + path, headers=headers, data=payload_str, timeout=15)
        if res.status_code >= 400:
            print(f"    [!] {symbol} HTTP {res.status_code}: {res.text[:200]}")
            log_order_response(symbol, action, size, price, res.status_code, res.text)
            return None
        body = res.json()
        order_id = (body.get("result") or {}).get("id")
        print(f"    [+] {action.strip()} {size} {symbol} @ {price} → order_id={order_id}")
        log_order_response(symbol, action, size, price, 200, json.dumps(body)[:500])
        return body
    except Exception as e:
        print(f"    [!] {symbol} submission error: {e}")
        log_order_response(symbol, action, size, price, -1, str(e)[:200])
        return None


# ─── Market Data ───────────────────────────────────────────────────────────

def fetch_bulk_tickers() -> dict:
    data = api_get("/v2/tickers")
    out = {}
    for t in data.get("result", []):
        sym = t.get("symbol")
        if sym:
            out[sym] = t
    return out


def fetch_candles(symbol: str, limit: int) -> pl.DataFrame:
    """
    Delta's /v2/history/candles requires start/end in unix seconds, not limit.
    Resolution is '1h', not '60m'.
    Fetches the last `limit` hours ending at the top of the current hour.
    """
    path = "/v2/history/candles"
    now_s = int(time.time())
    end_s = (now_s // 3600) * 3600              # top of current hour
    start_s = end_s - limit * 3600              # limit hours before that

    query = f"symbol={symbol}&resolution=1h&start={start_s}&end={end_s}"

    try:
        data = api_get(path, query=query)
    except Exception as e:
        print(f"    [!] candles {symbol}: {e}")
        return pl.DataFrame()

    rows = data.get("result", [])
    if not rows:
        return pl.DataFrame()

    df = pl.DataFrame(rows).select([
        (pl.col("time").cast(pl.Int64) * 1000).alias("open_time"),
        pl.col("close").cast(pl.Float64),
    ]).sort("open_time")

    # Drop the currently-forming bar (open_time >= now's hour)
    current_hour_ms = (int(time.time() * 1000) // 3_600_000) * 3_600_000
    df = df.filter(pl.col("open_time") < current_hour_ms)

    return df


def round_to_tick(price: float, tick: float) -> float:
    if tick <= 0:
        return price
    return round(price / tick) * tick


def safe_int_size(size: float, min_sz: float) -> Optional[int]:
    s = int(round(abs(size)))
    if s < 1 or s < int(min_sz):
        return None
    return s


def extract_spec(ticker: dict) -> dict:
    try:
        pid = int(ticker.get("product_id") or 0)
    except (TypeError, ValueError):
        pid = 0
    try:
        cv = float(ticker.get("contract_value") or 0)
    except (TypeError, ValueError):
        cv = 0.0
    try:
        tick = float(ticker.get("tick_size") or 0.0001)
    except (TypeError, ValueError):
        tick = 0.0001

    quotes = ticker.get("quotes") or {}
    try:
        best_bid = float(quotes.get("best_bid") or 0)
    except (TypeError, ValueError):
        best_bid = 0.0
    try:
        best_ask = float(quotes.get("best_ask") or 0)
    except (TypeError, ValueError):
        best_ask = 0.0

    try:
        mark = float(ticker.get("mark_price") or 0)
    except (TypeError, ValueError):
        mark = 0.0

    try:
        funding = float(ticker.get("funding_rate") or 0)
    except (TypeError, ValueError):
        funding = 0.0

    return {
        "product_id":     pid,
        "contract_value": cv,
        "tick_size":      tick,
        "best_bid":       best_bid,
        "best_ask":       best_ask,
        "mark_price":     mark,
        "funding_rate":   funding,
        "lot_size":       1.0,
        "min_size":       1.0,
    }


def choose_post_only_price(action: str, best_bid: float, best_ask: float,
                           mark: float, tick: float) -> float:
    if action.strip().upper() == "BUY":
        raw = best_bid if best_bid > 0 else mark
    else:
        raw = best_ask if best_ask > 0 else mark
    return round_to_tick(raw, tick)


# ─── Signal ────────────────────────────────────────────────────────────────

def compute_targets(tickers: dict) -> dict:
    print(f"[*] Fetching {len(UNIVERSE)} symbols of hourly history...")
    frames = []
    for sym in UNIVERSE:
        if sym not in tickers:
            print(f"    [skip] {sym}: not in Delta tickers universe")
            continue
        df = fetch_candles(sym, limit=LOOKBACK_H + 10)
        if df.is_empty() or len(df) < LOOKBACK_H + 5:
            print(f"    [skip] {sym}: not enough data ({len(df)} bars)")
            continue
        frames.append(df.with_columns(pl.lit(sym).alias("symbol")))
        time.sleep(0.05)

    if not frames:
        raise RuntimeError("No usable market data fetched.")

    full = pl.concat(frames).sort(["symbol", "open_time"])

    calc = (
        full.with_columns(
            (pl.col("close") / pl.col("close").shift(1) - 1.0)
              .over("symbol").alias("hourly_ret")
        )
        .with_columns([
            pl.col("hourly_ret").rolling_std(window_size=LOOKBACK_H)
              .over("symbol").alias("vol_lb"),
            (pl.col("close") / pl.col("close").shift(LOOKBACK_H) - 1.0)
              .over("symbol").alias("ret_lb"),
        ])
        .with_columns([
            (pl.col("ret_lb") / pl.col("vol_lb").clip(lower_bound=0.001))
              .alias("mom_score")
        ])
        .drop_nulls(subset=["mom_score"])
    )

    latest_time = calc["open_time"].max()
    latest = calc.filter(pl.col("open_time") == latest_time)

    active_coins = len(latest)
    if active_coins < TOP_N * 2 + 1:
        raise RuntimeError(f"Universe too small: {active_coins} symbols at {latest_time}")

    ranked = latest.with_columns(
        pl.col("mom_score").rank(descending=True).alias("rank")
    )

    w_leg = 0.5 / TOP_N
    ranked = ranked.with_columns([
        pl.when(pl.col("rank") <= TOP_N).then(pl.lit(w_leg))
          .when(pl.col("rank") > active_coins - TOP_N).then(pl.lit(-w_leg))
          .otherwise(pl.lit(0.0)).alias("target_weight")
    ])

    targets: dict = {}
    for row in ranked.iter_rows(named=True):
        sym = row["symbol"]
        ticker = tickers.get(sym, {}) or {}
        mark = 0.0
        try:
            mark = float(ticker.get("mark_price") or 0)
        except (TypeError, ValueError):
            mark = 0.0
        if mark <= 0:
            mark = float(row["close"])
        targets[sym] = (float(row["target_weight"]), mark)
    return targets


# ─── Logging ───────────────────────────────────────────────────────────────

def init_logs():
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    if not LOG_FILE.exists():
        with open(LOG_FILE, "w", newline="") as f:
            csv.writer(f).writerow([
                "timestamp_utc", "symbol", "side", "target_weight",
                "limit_price", "intended_contracts", "intended_notional_usd",
                "current_contracts", "order_delta_contracts", "dry_run",
            ])

    if not FUNDING_LOG.exists():
        with open(FUNDING_LOG, "w", newline="") as f:
            csv.writer(f).writerow(["timestamp_utc", "symbol", "funding_rate"])

    if not ORDER_LOG.exists():
        with open(ORDER_LOG, "w", newline="") as f:
            csv.writer(f).writerow([
                "timestamp_utc", "symbol", "side", "size",
                "limit_price", "http_status", "response",
            ])


def log_trade(ts, sym, side, weight, price, intended, notional, current, delta):
    with open(LOG_FILE, "a", newline="") as f:
        csv.writer(f).writerow([
            ts, sym, side, f"{weight:.6f}", f"{price:.6f}",
            f"{intended:.6f}", f"{notional:.2f}",
            f"{current:.6f}", f"{delta:.6f}", DRY_RUN,
        ])


def log_funding(ts, sym, rate):
    with open(FUNDING_LOG, "a", newline="") as f:
        csv.writer(f).writerow([ts, sym, f"{rate:.8f}"])


def log_order_response(sym, action, size, price, status, response):
    with open(ORDER_LOG, "a", newline="") as f:
        csv.writer(f).writerow([
            datetime.now(timezone.utc).isoformat(),
            sym, action.strip(), size, f"{price:.6f}",
            status, response,
        ])


# ─── Rebalance Window ──────────────────────────────────────────────────────

def should_rebalance(state: dict) -> bool:
    now = int(time.time())
    last = int(state.get("last_rebalance_ts", 0))
    min_gap = REBALANCE_H * 3600 - REBALANCE_TOL_S
    if now - last >= min_gap:
        return True
    remaining = min_gap - (now - last)
    print(f"[i] Next rebalance in {remaining/3600:.2f} hours.")
    return False


# ─── Main ──────────────────────────────────────────────────────────────────

def main():
    if API_KEY.startswith("YOUR_"):
        print("[!] API keys not set. Export DELTA_API_KEY and DELTA_API_SECRET.")
        return

    print("=" * 74)
    print(f"DELTA INDIA ROUTER — {'DRY RUN' if DRY_RUN else '*** LIVE ***'}")
    print(f"Config: lookback={LOOKBACK_H}h  rebalance={REBALANCE_H}h  "
          f"top_n={TOP_N}  lev={LEVERAGE}x")
    print("=" * 74)

    init_logs()
    state = load_state()
    ts_now = datetime.now(timezone.utc).isoformat()

    if not should_rebalance(state):
        return

    try:
        balance = None
        try:
            balance = fetch_wallet_balance()
            print(f"[+] Live USDT balance: {balance:.2f}")
        except Exception as e:
            print(f"[!] Balance fetch failed: {e}")

        if DRY_RUN:
            if balance is None or balance < 50:
                print(f"[i] PAPER MODE: using notional capital ${PAPER_CAPITAL:.2f}")
                balance = PAPER_CAPITAL
        else:
            if balance is None:
                print("[!] Cannot fetch balance in live mode. Aborting.")
                return
            if balance < 50:
                print("[!] Balance too low (<$50). Aborting.")
                return

        # Compute running peak/DD for the circuit breaker
        peak = max(state.get("peak_equity", balance), balance)
        dd = (balance - peak) / peak if peak > 0 else 0.0
        state["peak_equity"] = peak
        state["current_equity"] = balance

        if dd < -0.25:
            print(f"[!] CIRCUIT BREAKER: drawdown {dd*100:.1f}% < -25%. Halting.")
            save_state(state)
            return

        print(f"[i] Equity: {balance:.2f} | Peak: {peak:.2f} | DD: {dd*100:+.1f}%")

        if not DRY_RUN:
            if not cancel_all_open_orders():
                print("[!] Cancel-all failed. Aborting.")
                return
            time.sleep(1)

        current_positions = fetch_open_positions()
        tickers = fetch_bulk_tickers()
        targets = compute_targets(tickers)

        print(f"\n[*] Target portfolio (gross notional per side: "
              f"{balance*LEVERAGE:.2f} USD):")
        print(f"{'symbol':<10} {'weight':>8} {'price':>12} "
              f"{'current':>10} {'delta':>10} {'action':>6}")

        n_orders = 0
        for sym in UNIVERSE:
            if sym not in targets:
                continue
            ticker = tickers.get(sym, {}) or {}
            spec = extract_spec(ticker)

            weight, mark = targets[sym]
            log_funding(ts_now, sym, spec["funding_rate"])

            product_id   = spec["product_id"]
            contract_val = spec["contract_value"]
            tick_sz      = spec["tick_size"]
            lot          = spec["lot_size"]
            min_sz       = spec["min_size"]

            if contract_val <= 0 or mark <= 0:
                print(f"    [skip] {sym}: bad spec mark={mark} cv={contract_val}")
                continue

            target_usd = balance * LEVERAGE * weight
            target_contracts = target_usd / (mark * contract_val)
            target_contracts = round(target_contracts / lot) * lot

            current = current_positions.get(sym, 0.0)
            delta   = target_contracts - current
            delta_notional = abs(delta) * mark * contract_val

            if delta_notional < 5.0:
                continue

            action = "BUY" if delta > 0 else "SELL"
            valid_price = choose_post_only_price(
                action,
                best_bid=spec["best_bid"],
                best_ask=spec["best_ask"],
                mark=mark,
                tick=tick_sz,
            )

            size_int = safe_int_size(delta, min_sz)
            if size_int is None:
                print(f"    [skip] {sym}: size {delta:.4f} below min {min_sz}")
                continue

            print(f"{sym:<10} {weight:>+8.4f} {valid_price:>12.4f} "
                  f"{current:>10.4f} {delta:>+10.4f} {action:>6}")

            log_trade(ts_now, sym, action, weight, valid_price,
                      target_contracts, target_usd, current, delta)

            if DRY_RUN:
                print(f"    [paper] would POST-ONLY {action} {size_int} "
                      f"{sym} @ {valid_price:.2f}")
            else:
                submit_post_only_order(sym, action, size_int, valid_price, product_id)
                time.sleep(0.2)

            n_orders += 1

        if n_orders == 0:
            print("  (no trades needed this cycle)")

        state["last_rebalance_ts"] = int(time.time())
        save_state(state)
        print(f"\n[+] Rebalance logged ({n_orders} orders). State saved.")

    except Exception as e:
        print(f"[!] System error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()