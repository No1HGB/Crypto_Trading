import pandas as pd
from fetch import fetch_data
from backtest_preprocess import cal_value

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
take_profit_ratio = 0.0045
stop_loss_ratio = 0.003

# 백테스트 결과를 저장할 변수 초기화
win_count = 0
loss_count = 0

df: pd.DataFrame = fetch_data(symbol="BTCUSDT", interval="5m", numbers=3000)
df = cal_value(df)
print(df.shape)

# 백테스트 실행
for i in range(100, len(df)):
    if capital <= 0:
        break

    is_buy = False
    is_sell = False

    if pd.notna(df.at[i - 1, "buy_signal"]):
        if (
            df.at[i - 1, "buy_signal"]
            and df.at[i, "close"] > df.at[i, "open"]
            and df.at[i, "rsi"] >= 50
        ):
            is_buy = True
    elif pd.notna(df.at[i - 1, "sell_signal"]):
        if (
            df.at[i - 1, "sell_signal"]
            and df.at[i, "close"] < df.at[i, "open"]
            and df.at[i, "rsi"] <= 50
        ):
            is_sell = True

    if position == 1:
        current_price = df.at[i, "close"]

        if df.at[i, "high"] >= stop_loss_price >= df.at[i, "low"]:
            loss = margin * leverage * stop_loss_ratio

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

        elif is_sell:
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
            loss = margin * leverage * stop_loss_ratio

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

        elif is_buy:
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
        if is_buy:
            position = 1
            margin = capital / 5
            capital -= margin * leverage * (0.04 / 100)
            entry_price = df.at[i, "close"]

            # 손익비 설정
            min_val = df.iloc[i - 12 : i][["open", "close"]].min().min()
            stop_loss_ratio = abs((df.at[i, "close"] - min_val) / min_val)
            take_profit_ratio = stop_loss_ratio * 1.5

            # 손절가 설정
            stop_loss_price = entry_price * (1 - stop_loss_ratio)
            # 익절가 설정
            take_profit_price = entry_price * (1 + take_profit_ratio)

        elif is_sell:
            position = -1
            margin = capital / 5
            capital -= margin * leverage * (0.04 / 100)
            entry_price = df.at[i, "close"]

            # 손익비 설정
            max_val = df.iloc[i - 12 : i][["open", "close"]].max().max()
            stop_loss_ratio = abs((df.at[i, "close"] - max_val) / max_val)
            take_profit_ratio = stop_loss_ratio * 1.5

            # 손절가 설정
            stop_loss_price = entry_price * (1 + stop_loss_ratio)
            # 익절가 설정
            take_profit_price = entry_price * (1 - take_profit_ratio)


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
