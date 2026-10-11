"""inspect.py — Validate the latest recorded HDF5 file."""
import os
import numpy as np
import pandas as pd
import tables as tb

DATA_DIR = "data/btcusdt/aggtrade"

# Auto-discover the most recent file
files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".h5"))
if not files:
    raise SystemExit(f"No .h5 files in {DATA_DIR}")

latest = files[-1]
path = os.path.join(DATA_DIR, latest)
print(f"Reading: {path}\n")

with tb.open_file(path) as h:
    t = h.root.trades
    p = h.root.depth

    print("=== SCHEMA ===")
    print(f"trade itemsize: {t.dtype.itemsize}  (expect 65)")
    print(f"depth itemsize: {p.dtype.itemsize}  (expect 50)")

    print("\n=== TRADES ===")
    print(f"Rows: {len(t)}")
    if len(t) > 0:
        first = t[0]
        lat_ms = (first["recv_time_ns"] // 1_000_000) - first["event_time"]
        print(f"First trade price:    {first['price']}")
        print(f"First trade qty:      {first['quantity']}")
        print(f"First trade is_buyer_mm: {first['is_buyer_mm']}")
        print(f"First trade latency:  {lat_ms} ms")
        prices = t.cols.price[:]
        print(f"Price min/max:        {prices.min():.2f} / {prices.max():.2f}")
        buy_pct = (~t.cols.is_buyer_mm[:]).mean()
        print(f"Buy aggressor %:      {buy_pct:.2%}")

    print("\n=== DEPTH ===")
    print(f"Rows: {len(p)}")
    if len(p) > 0:
        recv_arr = p.cols.recv_time_ns[:]
        msgs = len(np.unique(recv_arr))
        print(f"Messages: {msgs}")
        print(f"Rows/message (avg):   {len(p)/msgs:.2f}  (expect 40.00)")
        sides = p.cols.side[:]
        print(f"Side counts: {dict(zip(*np.unique(sides, return_counts=True)))}")

        # Reconstruct first snapshot
        print("\nFirst snapshot, top 3 per side:")
        snap = pd.DataFrame({
            "side":  p.cols.side[:40],
            "level": p.cols.level[:40],
            "price": p.cols.price[:40],
            "qty":   p.cols.quantity[:40],
        })
        snap = snap[snap.level < 3].sort_values(["side", "level"])
        print(snap.to_string(index=False))

        best_bid = snap[(snap.side == 0) & (snap.level == 0)].price.values[0]
        best_ask = snap[(snap.side == 1) & (snap.level == 0)].price.values[0]
        print(f"\nBest bid: {best_bid}")
        print(f"Best ask: {best_ask}")
        print(f"Spread:   {best_ask - best_bid:.2f}")