import os
from dotenv import load_dotenv

load_dotenv()

key = os.getenv("KEY")
secret = os.getenv("SECRET")

symbols = ["BTCUSDT", "ETHUSDT"]
leverage = 5
interval = "1h"  # 1h,4h,1d
ratio = 20  # margin ratio per balance (%)

# tp/sl atr
tp_btc_atr = 2.3
sl_btc_atr = 2
tp_eth_atr = 1.51
sl_eth_atr = 1.3

# tp/sl percent
tp_btc = 1.5
sl_btc = 1.5
tp_eth = 1.35
sl_eth = 0.7
