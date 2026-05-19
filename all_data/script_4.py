import requests
import pandas as pd
import time
import os
from datetime import datetime, timedelta

symbols = [
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "SOLUSDT",
    "XRPUSDT",
    "ADAUSDT",
    "DOGEUSDT",
    "AVAXUSDT",
]

start_date = datetime(2020, 1, 1)
end_date = datetime.now()

os.makedirs("all_data", exist_ok=True)

def fetch_symbol(symbol):

    all_data = []

    url = "https://api.binance.com/api/v3/klines"

    current_time = start_date

    while current_time < end_date:

        params = {
            "symbol": symbol,
            "interval": "4h",
            "startTime": int(current_time.timestamp() * 1000),
            "limit": 1000
        }

        response = requests.get(url, params=params)
        data = response.json()

        if isinstance(data, dict) and "code" in data:
            print(f"Error for {symbol}: {data}")
            break

        if len(data) == 0:
            break

        for row in data:
            all_data.append({
                "date": pd.to_datetime(row[0], unit="ms"),
                "open": float(row[1]),
                "high": float(row[2]),
                "low": float(row[3]),
                "close": float(row[4]),
                "volume": float(row[5]),
                "quote_volume": float(row[7]),
                "num_trades": int(row[8]),
                "taker_buy_volume": float(row[9]),
                "taker_buy_quote": float(row[10]),
            })

        last_candle_time = pd.to_datetime(data[-1][0], unit="ms")

        print(f"[{symbol}] fetched until {last_candle_time}")

        current_time = last_candle_time + timedelta(hours=4)

        time.sleep(0.5)

    df = pd.DataFrame(all_data)

    df = df.drop_duplicates(subset=["date"])

    df = df.sort_values("date")

    return df


for symbol in symbols:

    print(f"\nFetching {symbol}...")

    df = fetch_symbol(symbol)

    name = symbol.replace("USDT", "").lower()

    filename = f"all_data/{name}_4h_binance.csv"

    df.to_csv(filename, index=False)

    print(f"Saved {len(df)} rows to {filename}")