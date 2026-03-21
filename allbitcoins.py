import requests
import pandas as pd
import time
from datetime import datetime, timedelta

symbols = [
    "BTCUSDT",   # Bitcoin
    "ETHUSDT",   # Ethereum
]

end_date = datetime.now()
start_date = end_date - timedelta(days=6 * 365)


def fetch_symbol(symbol):
    all_data = []
    current_start = start_date

    while current_start < end_date:
        current_end = min(current_start + timedelta(days=365), end_date)

        from_ts = int(current_start.timestamp()) * 1000
        to_ts   = int(current_end.timestamp()) * 1000

        url = "https://api.binance.com/api/v3/klines"
        params = {
            "symbol":    symbol,
            "interval":  "1d",
            "startTime": from_ts,
            "endTime":   to_ts,
            "limit":     1000
        }

        response = requests.get(url, params=params)
        data = response.json()

        if isinstance(data, dict) and "code" in data:
            print(f"[{symbol}] Error: {data}")
            return None

        for row in data:
            all_data.append({
                "date":         pd.to_datetime(row[0], unit="ms").date(),
                "open":         float(row[1]),
                "high":         float(row[2]),
                "low":          float(row[3]),
                "close":        float(row[4]),
                "volume":       float(row[5]),
                "quote_volume": float(row[7])
            })

        print(f"  [{symbol}] {current_start.date()} → {current_end.date()}, rows: {len(data)}")
        current_start = current_end
        time.sleep(0.5)

    return pd.DataFrame(all_data)   # raw, no cleaning


for symbol in symbols:
    print(f"\nFetching {symbol}...")
    df = fetch_symbol(symbol)

    if df is not None:
        name = symbol.replace("USDT", "").lower()
        filename = f"{name}_5years_binance.csv"
        df.to_csv(filename, index=False)
        print(f"  Saved {len(df)} rows to {filename}")
    else:
        print(f"  Skipped {symbol} due to error.")