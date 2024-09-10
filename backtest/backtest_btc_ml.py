import pandas as pd
from fetch import fetch_data
from preprocess import cal_values, x_data_backtest
import joblib

# 초기값 설정
symbol = "BTCUSDT"
initial_capital = 1000
capital = initial_capital
margin = 0
leverage = 5
position = 0
position_cnt = 0
entry_price = 0
take_profit_price = 0
stop_loss_price = 0
model_dir = f"../train/models/gb_classifier_{symbol}.pkl"

cnt_criteria = 5

model = joblib.load(model_dir)

# 익절, 손절 조건 설정
take_profit_ratio = 0.015
stop_loss_ratio = 0.015

# 백테스트 결과를 저장할 변수 초기화
win_count = 0
loss_count = 0

df: pd.DataFrame = fetch_data(symbol="BTCUSDT", interval="1h", numbers=770)
df = cal_values(df)
print(df.shape)

# 백테스트 실행
for i in range(17, len(df)):
    if capital <= 0:
        break

    if (
        df.at[i - 1, "ha_open"] > df.at[i - 1, "ha_close"]
        and df.at[i, "ha_open"] < df.at[i, "ha_close"]
        # and (df.at[i, "open"] < df.at[i, "ma20"])
    ):
        X_data = x_data_backtest(df, symbol, i)
        pred = model.predict(X_data)
        pred += 1
        pred_human = 2
        prob_lst = model.predict_proba(X_data)
        prob = max(prob_lst[0])
    elif (
        df.at[i - 1, "ha_open"] < df.at[i - 1, "ha_close"]
        and df.at[i, "ha_open"] > df.at[i, "ha_close"]
        # and (df.at[i, "open"] > df.at[i, "ma20"])
    ):
        X_data = x_data_backtest(df, symbol, i)
        pred = model.predict(X_data)
        pred += 1
        pred_human = 1
        prob_lst = model.predict_proba(X_data)
        prob = max(prob_lst[0])
    else:
        pred = 0
        pred_human = 0
        prob = 0

    if position == 2:
        current_price = df.at[i, "close"]
        position_cnt += 1

        if stop_loss_price >= df.at[i, "low"] and stop_loss_price <= df.at[i, "high"]:
            loss = margin * leverage * stop_loss_ratio

            capital -= loss
            loss_count += 1
            margin = 0
            position = 0
            position_cnt = 0

        # elif (
        #     take_profit_price >= df.at[i, "low"]
        #     and take_profit_price <= df.at[i, "high"]
        # ):
        #     profit = (
        #         margin * leverage * abs(take_profit_price - entry_price) / entry_price
        #     )
        #
        #     capital += profit
        #     win_count += 1
        #     margin = 0
        #     position = 0

        elif position_cnt == cnt_criteria:  # or (pred != 2 and pred_human != 2)
            profit_loss = (
                margin * leverage * (current_price - entry_price) / entry_price
            )

            if profit_loss > 0:
                capital += profit_loss
                win_count += 1
                margin = 0
                position = 0
                position_cnt = 0
            else:
                capital += profit_loss
                loss_count += 1
                margin = 0
                position = 0
                position_cnt = 0

    elif position == 1:
        current_price = df.at[i, "close"]
        position_cnt += 1

        if stop_loss_price >= df.at[i, "low"] and stop_loss_price <= df.at[i, "high"]:
            loss = margin * leverage * stop_loss_ratio

            capital -= loss
            loss_count += 1
            margin = 0
            position = 0
            position_cnt = 0

        # elif (
        #     take_profit_price >= df.at[i, "low"]
        #     and take_profit_price <= df.at[i, "high"]
        # ):
        #     profit = (
        #         margin * leverage * abs(take_profit_price - entry_price) / entry_price
        #     )
        #
        #     capital += profit
        #     win_count += 1
        #     margin = 0
        #     position = 0

        elif position_cnt == cnt_criteria:  # or (pred != 1 and pred_human != 1):
            profit_loss = (
                margin * leverage * (entry_price - current_price) / entry_price
            )

            if profit_loss > 0:
                capital += profit_loss
                win_count += 1
                margin = 0
                position = 0
                position_cnt = 0
            else:
                capital += profit_loss
                loss_count += 1
                margin = 0
                position = 0
                position_cnt = 0

    if position == 0:  # 포지션이 없다면
        if pred == 2 and pred_human == 2 and prob >= 0.9:
            position = 2
            margin = capital / 5
            capital -= margin * leverage * (0.04 / 100)
            entry_price = df.at[i, "close"]
            position_cnt = 1

            # 손절가 설정
            stop_loss_price = entry_price * (1 - stop_loss_ratio)
            # 익절가 설정
            # take_profit_price = entry_price * (1 + take_profit_ratio)

        elif pred == 1 and pred_human == 1 and prob >= 0.9:
            position = 1
            margin = capital / 5
            capital -= margin * leverage * (0.04 / 100)
            entry_price = df.at[i, "close"]
            position_cnt = 1

            # 손절가 설정
            stop_loss_price = entry_price * (1 + stop_loss_ratio)
            # 익절가 설정
            # take_profit_price = entry_price * (1 - take_profit_ratio)


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
