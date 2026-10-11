# Quantitative Trading Suite

An end-to-end data engineering and quantitative research framework for crypto and traditional markets.

## Repository Structure
- **`data_engineering/`**: Asynchronous WebSocket collectors (writing to HDF5), data auditing, and historical data fetchers.
- **`research/`**: Empirical tests for structural market inefficiencies, including CVD analysis, Order Book Imbalance (OBI), Variance Risk Premium (VRP), and Cointegration.

## Key Features
- **Data Pipeline:** Streams live aggTrade and depth20 data from Binance Futures into HDF5 files with automatic file rotation, disk flushing, and gap auditing.
- **Research Engine:** Vectorized backtesting using `polars` and `numpy`. Includes rigorous zero-lookahead validation, transaction cost modeling, and out-of-sample walk-forward testing.
- **Execution:** Delta Exchange router with paper-trading mode, latency profiling, and TWAP execution.

## How to Run
1. Set up your environment variables in a `.env` file (see `.env.example`).
2. Install dependencies: `pip install -r requirements.txt`
3. Run the data collector: `python data_engineering/binance_collector.py`
4. Run a research script: `python research/oos_validation.py`