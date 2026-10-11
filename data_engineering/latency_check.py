"""latency_check.py — Empirical network latency profiler for Binance Futures.

Usage:
    python latency_check.py
"""

import time
import requests
import websocket
import statistics
import numpy as np

REST_PING_URL = "https://fapi.binance.com/fapi/v1/ping"
WS_STREAM_URL = "wss://fstream.binancefuture.com/ws"

def measure_rest_latency(n=20):
    print(f"\n[1/2] Measuring Binance Futures REST RTT ({n} iterations)...")
    rtts = []
    
    for i in range(n):
        try:
            start = time.perf_counter()
            response = requests.get(REST_PING_URL, timeout=5)
            end = time.perf_counter()
            
            if response.status_code == 200:
                rtt_ms = (end - start) * 1000.0
                rtts.append(rtt_ms)
                print(f"  Iteration {i+1:02d}: {rtt_ms:>6.2f} ms")
            else:
                print(f"  Iteration {i+1:02d}: HTTP {response.status_code}")
        except Exception as e:
            print(f"  Iteration {i+1:02d}: FAILED ({e})")
        
        time.sleep(0.5) # Prevent rate-limiting
        
    if not rtts:
        print("  REST latency test failed completely.")
        return None
        
    print(f"  → Min:  {min(rtts):.2f} ms")
    print(f"  → Mean: {statistics.mean(rtts):.2f} ms")
    print(f"  → P95:  {np.percentile(rtts, 95):.2f} ms")
    print(f"  → Max:  {max(rtts):.2f} ms")
    return rtts

def measure_ws_latency():
    print(f"\n[2/2] Measuring WebSocket Connection Handshake Latency...")
    try:
        start = time.perf_counter()
        ws = websocket.create_connection(WS_STREAM_URL, timeout=5)
        end = time.perf_counter()
        
        handshake_ms = (end - start) * 1000.0
        print(f"  → WS Handshake RTT: {handshake_ms:.2f} ms")
        ws.close()
        return handshake_ms
    except Exception as e:
        print(f"  → WebSocket connection failed: {e}")
        return None

if __name__ == "__main__":
    print("=== BINANCE FUTURES NETWORK PROFILER ===")
    rest_rtts = measure_rest_latency(n=15)
    ws_rtt = measure_ws_latency()
    
    if rest_rtts:
        mean_rtt = statistics.mean(rest_rtts)
        one_way_latency = mean_rtt / 2.0
        print("\n=== SUMMARY FOR SIMULATOR CONFIG ===")
        print(f"  Empirical Round-Trip Time (RTT): {mean_rtt:.2f} ms")
        print(f"  Estimated One-Way Cancel Latency: {one_way_latency:.2f} ms")
        print("======================================")