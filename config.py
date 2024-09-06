import os
from dotenv import load_dotenv

load_dotenv()

key = os.getenv("KEY")
secret = os.getenv("SECRET")

symbols = ["BTCUSDT", "ETHUSDT"]
leverage = 5
interval = "15m"  # 1h,4h,1d
ratio = 20  # margin ratio per balance (%)
tp = 1  # take profit ratio for price (%)
sl = 1  # stop loss ratio for price (%)
tp_s = 0.5
sl_s = 0.5
