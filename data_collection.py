import yfinance as yf
import pandas as pd
import time

def get_stock_data(ticker, period='max', interval='1d',retries=2, delay=2):
    attempt = 0
    while attempt <= retries:
        try:
            print(f"[INFO] Attempting to fetch data for {ticker} (Attempt {attempt + 1})...")
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period, interval=interval)

            if hist.empty:
                raise ValueError(f"No data found for ticker: {ticker}")

            print(f"[SUCCESS] Data fetched for {ticker} ({len(hist)} records).")
            return hist

        except Exception as e:
            print(f"[ERROR] {e}")
            attempt += 1
            if attempt > retries:
                raise ValueError(f"Failed to fetch data for {ticker} after {retries} retries: {e}")
            print(f"[INFO] Retrying in {delay} seconds...")
            time.sleep(delay)

# Example usage
if __name__ == "__main__":
    ticker = 'GOLD'
    try:
        data = get_stock_data(ticker, period='max', interval='1d')
        print(data.tail(10))
    except ValueError as err:
        print(f"[FATAL] {err}")


