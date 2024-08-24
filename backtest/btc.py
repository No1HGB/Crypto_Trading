import pandas as pd
import joblib

from fetch import fetch_data
from calculate import cal_values, backtest_X_data


# 초기값 설정
initial_capital = 1000
capital = initial_capital
margin = 0
leverage = 5
position = 0  # 포지션: 0 - 없음, 1 - 매도, 2 - 매수
entry_price = 0
take_profit_price = 0
stop_loss_price = 0

# 익절, 손절 조건 설정
take_profit_ratio = 0.02
stop_loss_ratio = 0.015

# 백테스트 결과를 저장할 변수 초기화
win_count = 0
loss_count = 0

df: pd.DataFrame = fetch_data(symbol="BTCUSDT", interval="1h", numbers=720)
df = cal_values(df)
X_data = backtest_X_data(df)

# 모델
model_dir = f"model/gb_classifier_btc.pkl"
model = joblib.load(model_dir)

# 백테스트 실행
for i in range(len(df)):
    if capital <= 0:
        break
    pred = model.predict([X_data[i]])

    if position == 2:
        current_price = df.at[i, "close"]

        if stop_loss_price >= df.at[i, "low"] and stop_loss_price <= df.at[i, "high"]:
            loss = margin * leverage * stop_loss_ratio

            capital -= loss
            loss_count += 1
            margin = 0
            position = 0

        elif pred == 1:
            profit_or_loss = (
                margin * leverage * (current_price - entry_price) / entry_price
            )
            if profit_or_loss > 0:
                capital += profit_or_loss
                win_count += 1
                margin = 0
                position = 0
            else:
                capital += profit_or_loss
                loss_count += 1
                margin = 0
                position = 0

    elif position == 1:
        current_price = df.at[i, "close"]

        if stop_loss_price >= df.at[i, "low"] and stop_loss_price <= df.at[i, "high"]:
            loss = margin * leverage * stop_loss_ratio

            capital -= loss
            loss_count += 1
            margin = 0
            position = 0

        elif pred == 2:
            profit_or_loss = (
                margin * leverage * (entry_price - current_price) / entry_price
            )
            if profit_or_loss > 0:
                capital += profit_or_loss
                win_count += 1
                margin = 0
                position = 0
            else:
                capital += profit_or_loss
                loss_count += 1
                margin = 0
                position = 0

    if position == 0:  # 포지션이 없다면
        if pred == 2:
            position = 2
            margin = capital / 5
            capital -= margin * leverage * (0.1 / 100)
            entry_price = df.at[i, "close"]

            # 손절가 설정
            stop_loss_price = entry_price * (1 - stop_loss_ratio)

        elif pred == 1:
            position = 1
            margin = capital / 5
            capital -= margin * leverage * (0.1 / 100)
            entry_price = df.at[i, "close"]

            # 손절가 설정
            stop_loss_price = entry_price * (1 + stop_loss_ratio)


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
