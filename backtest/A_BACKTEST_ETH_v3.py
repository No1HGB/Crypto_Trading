import pandas as pd
from fetch import fetch_data

from preprocess import cal_values, x_data_backtest_v3
from backtest_logic import trend_long, trend_short
import joblib

# 초기값 설정
symbol = "ETHUSDT"
interval = "1h"


initial_capital = 1000
capital = initial_capital
margin = 0
leverage = 5
position = -1
position_cnt = 0
entry_price = 0
take_profit_price = 0
stop_loss_price = 0
model_dir = f"../train/models/gb_classifier_BTCUSDT_v3.pkl"

prob_baseline = 0.7

model = joblib.load(model_dir)

# 익절, 손절 조건 설정
sl_atr = 1.5
tp_atr = 1.5

# 백테스트 결과를 저장할 변수 초기화
win_count = 0
loss_count = 0

df: pd.DataFrame = fetch_data(symbol=symbol, interval=interval, numbers=400)
df = cal_values(df)
print(df.shape)


# 백테스트 실행
for i in range(48, len(df)):
    if capital <= 0:
        break

    X_data = x_data_backtest_v3(df, symbol, i)
    pred = model.predict(X_data)
    prob_lst = model.predict_proba(X_data)
    prob = float(max(prob_lst[0]))
    t_long = trend_long(df, i)
    t_short = trend_short(df, i)

    if position == 1:
        current_price = df.at[i, "close"]
        position_cnt += 1

        if pred == 1 and prob >= prob_baseline and position_cnt >= 4:
            position_cnt = 1

        if stop_loss_price >= df.at[i, "low"]:
            loss = margin * leverage * abs(stop_loss_price - entry_price) / entry_price

            capital -= loss
            loss_count += 1
            margin = 0
            position = -1
            position_cnt = 0

        elif df.at[i, "high"] >= take_profit_price:
            profit = (
                margin * leverage * abs(take_profit_price - entry_price) / entry_price
            )

            capital += profit
            win_count += 1
            margin = 0
            position = -1
            position_cnt = 0

        elif position_cnt == 4 and not (pred == 1 and prob >= prob_baseline):
            profit_loss = (
                margin * leverage * (current_price - entry_price) / entry_price
            )

            if profit_loss > 0:
                capital += profit_loss
                win_count += 1
                margin = 0
                position = -1
                position_cnt = 0

            else:
                capital += profit_loss
                loss_count += 1
                margin = 0
                position = -1
                position_cnt = 0

    elif position == 0:
        current_price = df.at[i, "close"]
        position_cnt += 1

        if pred == 0 and prob >= prob_baseline and position_cnt >= 4:
            position_cnt = 1

        if df.at[i, "high"] >= stop_loss_price:
            loss = margin * leverage * abs(stop_loss_price - entry_price) / entry_price

            capital -= loss
            loss_count += 1
            margin = 0
            position = -1
            position_cnt = 0

        elif take_profit_price >= df.at[i, "low"]:
            profit = (
                margin * leverage * abs(take_profit_price - entry_price) / entry_price
            )

            capital += profit
            win_count += 1
            margin = 0
            position = -1
            position_cnt = 0

        elif position_cnt == 4 and not (pred == 0 and prob >= prob_baseline):
            profit_loss = (
                margin * leverage * (current_price - entry_price) / entry_price
            )

            if profit_loss > 0:
                capital += profit_loss
                win_count += 1
                margin = 0
                position = -1
                position_cnt = 0

            else:
                capital += profit_loss
                loss_count += 1
                margin = 0
                position = -1
                position_cnt = 0

    if position == -1:  # 포지션이 없다면
        if pred == 1 and prob >= prob_baseline and not trend_short(df, i):
            position = 1
            margin = capital / 5
            capital -= margin * leverage * (0.07 / 100)
            entry_price = df.at[i, "close"]
            ATR = df.at[i, "ATR"]
            position_cnt = 1

            # 손절가 설정
            stop_loss_price = entry_price - sl_atr * ATR
            # 익절가 설정
            take_profit_price = entry_price + tp_atr * ATR

        elif pred == 0 and prob >= prob_baseline and not trend_long(df, i):
            position = 0
            margin = capital / 5
            capital -= margin * leverage * (0.07 / 100)
            entry_price = df.at[i, "close"]
            ATR = df.at[i, "ATR"]
            position_cnt = 1

            # 손절가 설정
            stop_loss_price = entry_price + sl_atr * ATR
            # 익절가 설정
            take_profit_price = entry_price - tp_atr * ATR


# 백테스트 결과 계산
total_trades = win_count + loss_count
win_rate = (win_count / total_trades) * 100 if total_trades > 0 else 0
final_capital = capital

# 결과 출력
print(f"Total Trades: {total_trades}")
print(f"Wins: {win_count}")
print(f"Losses: {loss_count}")
print(f"Win Rate: {win_rate:.2f}%")
print(f"Final Capital: {final_capital:.2f}")
