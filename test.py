from preprocess import cal_values, make_data, x_data
from fetch import fetch_data

symbol = "BTCUSDT"
interval = "15m"
data_num = 127

df = fetch_data(symbol, interval, data_num)
end_timestamp = df.iloc[-1]["close_time"]
df = cal_values(df)
X_data = x_data(df, symbol, False)
print(X_data.shape)
