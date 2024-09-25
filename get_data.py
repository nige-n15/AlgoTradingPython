import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta


def fetch_data(symbol, start_date, end_date, interval='1d'):
    ticker = yf.Ticker(symbol)
    data = ticker.history(start=start_date, end=end_date, interval=interval)
    return data


def main():
    symbol = "BTC-USD"  # Example: Bitcoin to USD
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)  # Last year of data

    data = fetch_data(symbol, start_date, end_date)
    print(data.head())

    # Save to CSV
    data.to_csv(f"{symbol}_data.csv")


if __name__ == "__main__":
    main()