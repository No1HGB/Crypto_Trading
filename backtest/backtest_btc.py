import pandas as pd
from fetch import fetch_data
from preprocess import cal_values
from backtest_logic import bb_long, bb_short, trend_long, trend_short

# 초기값 설정
initial_capital = 1000
capital = initial_capital
margin = 0
leverage = 5
position = 0  # 포지션: 0 - 없음, 1 - 매수, -1 - 매도
entry_price = 0
take_profit_price = 0
stop_loss_price = 0

# 익절, 손절 조건 설정
tp_atr = 1.5
sl_atr = 1.5

# 백테스트 결과를 저장할 변수 초기화
win_count = 0
loss_count = 0

df: pd.DataFrame = fetch_data(symbol="BTCUSDT", interval="1h", numbers=700)
df = cal_values(df)
print(df.shape)

# 백테스트 실행
for i in range(3, len(df)):
    if capital <= 0:
        break

    b_long = bb_long(df, i)
    b_short = bb_short(df, i)
    t_long = trend_long(df, i)
    t_short = trend_short(df, i)

    if position == 1:
        current_price = df.at[i, "close"]

        if df.at[i, "high"] >= stop_loss_price >= df.at[i, "low"]:
            loss = margin * leverage * abs(stop_loss_price - entry_price) / entry_price

            capital -= loss
            loss_count += 1
            margin = 0
            position = 0

        elif df.at[i, "high"] >= take_profit_price >= df.at[i, "low"]:
            profit = (
                margin * leverage * abs(take_profit_price - entry_price) / entry_price
            )

            capital += profit
            win_count += 1
            margin = 0
            position = 0

        elif b_short:
            profit_loss = (
                margin * leverage * (current_price - entry_price) / entry_price
            )

            if profit_loss > 0:
                capital += profit_loss
                win_count += 1
                margin = 0
                position = 0
            else:
                capital += profit_loss
                loss_count += 1
                margin = 0
                position = 0

    elif position == -1:
        current_price = df.at[i, "close"]

        if df.at[i, "high"] >= stop_loss_price >= df.at[i, "low"]:
            loss = margin * leverage * abs(stop_loss_price - entry_price) / entry_price

            capital -= loss
            loss_count += 1
            margin = 0
            position = 0

        elif df.at[i, "high"] >= take_profit_price >= df.at[i, "low"]:
            profit = (
                margin * leverage * abs(take_profit_price - entry_price) / entry_price
            )

            capital += profit
            win_count += 1
            margin = 0
            position = 0

        elif b_long:
            profit_loss = (
                margin * leverage * (entry_price - current_price) / entry_price
            )

            if profit_loss > 0:
                capital += profit_loss
                win_count += 1
                margin = 0
                position = 0
            else:
                capital += profit_loss
                loss_count += 1
                margin = 0
                position = 0

    if position == 0:  # 포지션이 없다면
        if (b_long and not t_short) or t_long:
            position = 1
            margin = capital / 5
            capital -= margin * leverage * (0.07 / 100)
            entry_price = df.at[i, "close"]
            ATR = df.at[i, "ATR"]

            # 손절가 설정
            stop_loss_price = entry_price - sl_atr * ATR
            # 익절가 설정
            take_profit_price = entry_price + tp_atr * ATR

        elif (b_short and not t_long) or t_short:
            position = -1
            margin = capital / 5
            capital -= margin * leverage * (0.07 / 100)
            entry_price = df.at[i, "close"]
            ATR = df.at[i, "ATR"]

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
