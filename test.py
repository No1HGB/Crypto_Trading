from fetch import fetch_data
from preprocess import cal_values
import pandas as pd

symbol = "BTCUSDT"
interval = "1h"

df: pd.DataFrame = fetch_data(symbol=symbol, interval=interval, numbers=400)
df = cal_values(df)
print(df["close"])
