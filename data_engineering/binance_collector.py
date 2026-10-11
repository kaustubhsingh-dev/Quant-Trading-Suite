"""
binance_collector.py — Record BTC/USDT Futures aggTrade + depth20 to HDF5
Usage: python binance_collector.py --symbol btcusdt --output ./data

Streams:
  - <symbol>@aggTrade        : tick-by-tick aggregated trades
  - <symbol>@depth20@100ms   : top-20 bid/ask snapshot every 100ms

Storage: NumPy structured dtypes are used as the HDF5 table description
directly, guaranteeing byte-for-byte layout match between write and read.
"""

import asyncio
import json
import time
import signal
import argparse
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple
import sys
import numpy as np
import tables as tb
import websockets
from websockets.exceptions import ConnectionClosed, InvalidStatusCode

# ─── Configuration ───────────────────────────────────────────────────────────

WS_BASE              = "wss://fstream.binancefuture.com/stream"
FLUSH_INTERVAL       = 1000
FLUSH_INTERVAL_DEPTH = 4000
FLUSH_TIMEOUT        = 5.0
DISK_FLUSH_SECONDS   = 60.0
FILE_ROTATE_SECONDS  = 3600
QUEUE_MAXSIZE        = 200_000
DEPTH_LEVELS         = 20

TRADE_DTYPE = np.dtype([
    ("event_time",     "i8"),
    ("trade_time",     "i8"),
    ("recv_time_ns",   "i8"),
    ("agg_trade_id",   "i8"),
    ("price",          "f8"),
    ("quantity",       "f8"),
    ("is_buyer_mm",    "?"),
    ("first_trade_id", "i8"),
    ("last_trade_id",  "i8"),
])

DEPTH_DTYPE = np.dtype([
    ("event_time",   "i8"),
    ("txn_time",     "i8"),
    ("recv_time_ns", "i8"),
    ("msg_seq",      "i8"),
    ("side",         "i1"),
    ("level",        "i1"),
    ("price",        "f8"),
    ("quantity",     "f8"),
])

# ─── HDF5 Writer ────────────────────────────────────────────────────────────

class HDF5Writer:
    def __init__(self, output_dir: Path, symbol: str):
        self.output_dir = output_dir / symbol / "aggtrade"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.symbol = symbol
        self.file = None
        self.trade_table = None
        self.depth_table = None
        self.file_created_at = 0.0

    def _open_new_file(self):
        if self.file:
            self.trade_table.flush()
            self.depth_table.flush()
            self.file.close()

        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H")
        path = self.output_dir / f"{ts}.h5"
        self.file = tb.open_file(str(path), mode="a")

        if "/trades" not in self.file:
            self.trade_table = self.file.create_table(
                "/", "trades", TRADE_DTYPE,
                f"{self.symbol.upper()} aggTrade tick data",
                filters=tb.Filters(complevel=5, complib="blosc"),
                expectedrows=5_000_000,
            )
        else:
            self.trade_table = self.file.root.trades

        if "/depth" not in self.file:
            self.depth_table = self.file.create_table(
                "/", "depth", DEPTH_DTYPE,
                f"{self.symbol.upper()} top-{DEPTH_LEVELS} depth snapshots",
                filters=tb.Filters(complevel=5, complib="blosc"),
                expectedrows=20_000_000,
            )
        else:
            self.depth_table = self.file.root.depth

        self.file_created_at = time.time()
        print(f"[HDF5] Opened {path} | "
              f"trade_row={TRADE_DTYPE.itemsize}B depth_row={DEPTH_DTYPE.itemsize}B")

    def _maybe_rotate(self):
        if self.file is None or (time.time() - self.file_created_at) > FILE_ROTATE_SECONDS:
            self._open_new_file()

    def append_trades(self, batch: np.ndarray):
        self._maybe_rotate()
        self.trade_table.append(batch)

    def append_depth(self, batch: np.ndarray):
        self._maybe_rotate()
        self.depth_table.append(batch)

    def disk_flush(self):
        if self.file:
            self.trade_table.flush()
            self.depth_table.flush()

    def close(self):
        if self.file:
            self.trade_table.flush()
            self.depth_table.flush()
            self.file.close()
            print("[HDF5] Closed.")

# ─── Worker-thread helpers ──────────────────────────────────────────────────

def _worker_append_trades(hdf5, chunk):
    arr = np.array(chunk, dtype=TRADE_DTYPE)
    hdf5.append_trades(arr)

def _worker_append_depth(hdf5, chunk):
    arr = np.array(chunk, dtype=DEPTH_DTYPE)
    hdf5.append_depth(arr)

def _worker_disk_flush(hdf5):
    hdf5.disk_flush()

def _worker_close(hdf5):
    hdf5.close()

# ─── WebSocket Consumer + Router ────────────────────────────────────────────

async def consume(symbol: str,
                  trade_q: asyncio.Queue,
                  depth_q: asyncio.Queue,
                  stop_event: asyncio.Event):
    url = (f"{WS_BASE}?streams="
           f"{symbol.lower()}@aggTrade/"
           f"{symbol.lower()}@depth{DEPTH_LEVELS}@100ms")
    backoff = 1.0

    while not stop_event.is_set():
        try:
            print(f"[WS] Connecting to {url} ...")
            async with websockets.connect(
                url,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=5,
                max_size=2**22,
            ) as ws:
                print("[WS] Connected.")
                backoff = 1.0

                # First message (bounded wait)
                try:
                    first = await asyncio.wait_for(ws.recv(), timeout=15)
                    print(f"[WS] First message received ({len(first)} bytes)")
                    await _route(first, trade_q, depth_q)
                except asyncio.TimeoutError:
                    print("[WS] No data within 15s, reconnecting ...")
                    continue

                # Main loop — wakes up every 1s to check stop_event
                while not stop_event.is_set():
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
                    except asyncio.TimeoutError:
                        continue
                    await _route(raw, trade_q, depth_q)

                print("[WS] Stop requested, closing connection ...")

        except (ConnectionClosed, InvalidStatusCode, OSError) as e:
            if stop_event.is_set():
                break
            print(f"[WS] Disconnected: {e}. Reconnecting in {backoff:.1f}s ...")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30.0)


async def _route(raw,
                 trade_q: asyncio.Queue,
                 depth_q: asyncio.Queue):
    wrapper = json.loads(raw)
    stream  = wrapper.get("stream", "")
    data    = wrapper.get("data", wrapper)
    recv_ns = time.time_ns()

    if stream.endswith("@aggTrade"):
        await trade_q.put((
            data["E"], data["T"], recv_ns, data["a"],
            float(data["p"]), float(data["q"]),
            bool(data["m"]), data["f"], data["l"],   # ← cast to bool explicitly
        ))

    elif "depth" in stream:
        et, tt = data["E"], data["T"]
        seq = getattr(_route, "msg_seq", 0) + 1
        _route.msg_seq = seq
        rows = []
        for lvl, (p, q) in enumerate(data["b"]):
            rows.append((et, tt, recv_ns, seq, 0, lvl, float(p), float(q)))
        for lvl, (p, q) in enumerate(data["a"]):
            rows.append((et, tt, recv_ns, seq, 1, lvl, float(p), float(q)))
        if rows:
            await depth_q.put(rows)

# ─── Writer Tasks ───────────────────────────────────────────────────────────

async def trade_writer(queue: asyncio.Queue,
                       hdf5: HDF5Writer,
                       stop_event: asyncio.Event,
                       executor: ThreadPoolExecutor,
                       loop: asyncio.AbstractEventLoop):
    buffer: List[Tuple] = []
    total = 0
    last_batch = time.time()
    last_disk  = time.time()
    slow = 0

    while not stop_event.is_set() or not queue.empty():
        try:
            buffer.append(await asyncio.wait_for(queue.get(), timeout=FLUSH_TIMEOUT))
        except asyncio.TimeoutError:
            pass

        now = time.time()
        if buffer and (len(buffer) >= FLUSH_INTERVAL or (now - last_batch) > FLUSH_TIMEOUT):
            chunk, buffer = buffer, []
            t0 = time.perf_counter()
            await loop.run_in_executor(executor, _worker_append_trades, hdf5, chunk)
            dt = (time.perf_counter() - t0) * 1000.0
            if dt > 20:
                slow += 1
                print(f"[WARN] trades: {dt:.1f}ms for {len(chunk)} rows")
            total += len(chunk)
            last_batch = now
            print(f"[TRADES] Total: {total:,} | batch {len(chunk)} in {dt:.1f}ms")

        if now - last_disk >= DISK_FLUSH_SECONDS:
            await loop.run_in_executor(executor, _worker_disk_flush, hdf5)
            last_disk = now

    if buffer:
        chunk, buffer = buffer, []
        await loop.run_in_executor(executor, _worker_append_trades, hdf5, chunk)
        total += len(chunk)
    print(f"\n[TRADES] Done. Total: {total:,} | slow flushes: {slow}")


async def depth_writer(queue: asyncio.Queue,
                       hdf5: HDF5Writer,
                       stop_event: asyncio.Event,
                       executor: ThreadPoolExecutor,
                       loop: asyncio.AbstractEventLoop):
    buffer: List[Tuple] = []
    total = 0
    msgs  = 0
    last_batch = time.time()
    last_disk  = time.time()
    slow = 0

    while not stop_event.is_set() or not queue.empty():
        try:
            rows = await asyncio.wait_for(queue.get(), timeout=FLUSH_TIMEOUT)
            buffer.extend(rows)
            msgs += 1
        except asyncio.TimeoutError:
            pass

        now = time.time()
        if buffer and (len(buffer) >= FLUSH_INTERVAL_DEPTH or (now - last_batch) > FLUSH_TIMEOUT):
            chunk, buffer = buffer, []
            t0 = time.perf_counter()
            await loop.run_in_executor(executor, _worker_append_depth, hdf5, chunk)
            dt = (time.perf_counter() - t0) * 1000.0
            if dt > 20:
                slow += 1
                print(f"[WARN] depth:  {dt:.1f}ms for {len(chunk)} rows")
            total += len(chunk)
            last_batch = now
            print(f"[DEPTH]  Total: {total:,} rows | {msgs:,} msgs | "
                  f"batch {len(chunk)} in {dt:.1f}ms")

        if now - last_disk >= DISK_FLUSH_SECONDS:
            await loop.run_in_executor(executor, _worker_disk_flush, hdf5)
            last_disk = now

    if buffer:
        chunk, buffer = buffer, []
        await loop.run_in_executor(executor, _worker_append_depth, hdf5, chunk)
        total += len(chunk)
    print(f"\n[DEPTH]  Done. Total: {total:,} rows | {msgs:,} msgs | slow: {slow}")

# ─── Main ───────────────────────────────────────────────────────────────────

async def main(args):
    trade_q    = asyncio.Queue(maxsize=QUEUE_MAXSIZE)
    depth_q    = asyncio.Queue(maxsize=QUEUE_MAXSIZE)
    stop_event = asyncio.Event()
    hdf5       = HDF5Writer(Path(args.output), args.symbol)

    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hdf5-writer")
    loop     = asyncio.get_running_loop()

    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, stop_event.set)
    except NotImplementedError:
        def _sig_handler(signum, frame):
            loop.call_soon_threadsafe(stop_event.set)
        signal.signal(signal.SIGINT, _sig_handler)
        try:
            signal.signal(signal.SIGTERM, _sig_handler)
        except (ValueError, OSError):
            pass

    session_start = time.time()
    print(f"[SESSION] Started at {datetime.now(timezone.utc).isoformat()}")
    print(f"Recording {args.symbol.upper()} to {args.output}")
    print(f"Streams: aggTrade + depth{DEPTH_LEVELS}@100ms")
    print(f"Batch flush: trades {FLUSH_INTERVAL} rows | depth {FLUSH_INTERVAL_DEPTH} rows | "
          f"idle {FLUSH_TIMEOUT}s")
    print(f"Disk fsync:  every {DISK_FLUSH_SECONDS:.0f}s")
    print(f"File rotate: every {FILE_ROTATE_SECONDS}s\n")

    try:
        await asyncio.gather(
            consume(args.symbol, trade_q, depth_q, stop_event),
            trade_writer(trade_q, hdf5, stop_event, executor, loop),
            depth_writer(depth_q, hdf5, stop_event, executor, loop),
        )
    finally:
        print("[MAIN] Shutting down ...")
        try:
            await asyncio.wait_for(
                loop.run_in_executor(executor, _worker_close, hdf5),
                timeout=15.0,
            )
        except asyncio.TimeoutError:
            print("[MAIN] WARNING: HDF5 close timed out; file may be dirty")
        executor.shutdown(wait=False)
        dur = (time.time() - session_start) / 60.0
        print(f"[SESSION] Ended at {datetime.now(timezone.utc).isoformat()} "
              f"(duration {dur:.1f} min)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="btcusdt")
    parser.add_argument("--output", default="./data")
    args = parser.parse_args()

    exit_code = 0
    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        print("\n[MAIN] Interrupted by user.")
    except Exception:
        import traceback
        traceback.print_exc()
        exit_code = 1
    finally:
        import sys, os
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exit_code)