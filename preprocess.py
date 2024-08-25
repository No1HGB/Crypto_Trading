import numpy as np
import pandas as pd


def cal_values(df: pd.DataFrame) -> pd.DataFrame:
    df["MA10"] = df["close"].rolling(window=10).mean()
    df["MA20"] = df["close"].rolling(window=20).mean()
    df["MA50"] = df["close"].rolling(window=50).mean()
    df["MA200"] = df["close"].rolling(window=200).mean()
    df["EMA10"] = df["close"].ewm(alpha=(2 / 11), adjust=False).mean()
    df["EMA20"] = df["close"].ewm(alpha=(2 / 21), adjust=False).mean()
    df["EMA50"] = df["close"].ewm(alpha=(2 / 51), adjust=False).mean()
    df["EMA200"] = df["close"].ewm(alpha=(2 / 201), adjust=False).mean()
    df["volume_MA50"] = df["volume"].rolling(window=50).mean()

    # 하이킨아시
    df["ha_close"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
    df["ha_open"] = 0.0
    df.at[0, "ha_open"] = df.iloc[0]["open"]
    df["ha_high"] = 0.0
    df["ha_low"] = 0.0
    for i in range(1, len(df)):
        df.at[i, "ha_open"] = (df.at[i - 1, "ha_open"] + df.at[i - 1, "ha_close"]) / 2
        df.at[i, "ha_high"] = max(
            df.at[i, "high"], df.at[i, "ha_open"], df.at[i, "ha_close"]
        )
        df.at[i, "ha_low"] = min(
            df.at[i, "low"], df.at[i, "ha_open"], df.at[i, "ha_close"]
        )
    df.at[0, "ha_high"] = df.at[0, "high"]
    df.at[0, "ha_low"] = df.at[0, "low"]

    # null값 행 및 0값 행 제거
    df.dropna(axis=0, inplace=True, how="any")
    df = df[(df["open"] > 0) & (df["close"] > 0) & (df["volume"] > 0)].copy()

    # 필요한 값 계산
    df["delta"] = df["close"] / df["open"]
    df["up_delta"] = df["high"] / df[["open", "close"]].max(axis=1)
    df["down_delta"] = df["low"] / df[["open", "close"]].min(axis=1)
    df["d10"] = df["close"] / df["MA10"]
    df["d20"] = df["close"] / df["MA20"]
    df["d50"] = df["close"] / df["MA50"]
    df["d200"] = df["close"] / df["MA200"]
    df["e_d10"] = df["close"] / df["EMA10"]
    df["e_d20"] = df["close"] / df["EMA20"]
    df["e_d50"] = df["close"] / df["EMA50"]
    df["e_d200"] = df["close"] / df["EMA200"]
    df["volume_delta"] = df["volume"] / df["volume_MA50"]
    df["volume_ratio"] = df["volume"] / df["volume"].shift(1)
    df["ha_delta"] = df["ha_close"] / df["ha_open"]
    df["ha_up_delta"] = df["ha_high"] / df[["ha_open", "ha_close"]].max(axis=1)
    df["ha_down_delta"] = df["ha_low"] / df[["ha_open", "ha_close"]].min(axis=1)

    df.drop(
        [
            "volume",
            "MA10",
            "MA20",
            "MA50",
            "MA200",
            "EMA10",
            "EMA20",
            "EMA50",
            "EMA200",
            "volume_MA50",
        ],
        axis=1,
        inplace=True,
    )
    df.dropna(axis=0, inplace=True, how="any")
    df.reset_index(drop=True, inplace=True)

    return df


def make_data(df: pd.DataFrame, window: int, sl: float, tp_min: float = 0.2):
    columns = [
        "open_time",
        "close_time",
        "delta",
        "up_delta",
        "down_delta",
        "d10",
        "d20",
        "d50",
        "d200",
        "e_d10",
        "e_d20",
        "e_d50",
        "e_d200",
        "volume_delta",
        "volume_ratio",
        "ha_delta",
        "ha_up_delta",
        "ha_down_delta",
    ]
    x_data = []
    y_data = []

    selected_df = df[columns]
    for i in range(df.shape[0] - window):
        entry = df.iloc[i]["close"]
        close = df.iloc[i + window]["close"]
        window_delta = (close - entry) / entry * 100
        low = df.iloc[i + 1 : i + window]["low"].min()
        high = df.iloc[i + 1 : i + window]["high"].max()

        long_sl = entry * (1 - sl / 100)
        short_sl = entry * (1 + sl / 100)

        # long
        if tp_min <= abs(window_delta) and window_delta > 0 and long_sl < low:
            y_data.append(2)
        # short
        elif tp_min <= abs(window_delta) and window_delta < 0 and short_sl > high:
            y_data.append(1)
        else:
            y_data.append(0)

        x = selected_df.iloc[i].values
        x_data.append(x)

    real_x_data = []
    real_x = selected_df.iloc[-1].values
    real_x_data.append(real_x)

    return np.array(x_data), np.array(y_data), np.array(real_x_data)


def backtest_X_data(df: pd.DataFrame):
    columns = [
        "open_time",
        "close_time",
        "delta",
        "up_delta",
        "down_delta",
        "d10",
        "d20",
        "d50",
        "d200",
        "e_d10",
        "e_d20",
        "e_d50",
        "e_d200",
        "volume_delta",
        "volume_ratio",
        "ha_delta",
        "ha_up_delta",
        "ha_down_delta",
    ]
    x_data = []

    selected_df = df[columns]
    for i in range(df.shape[0]):
        x = selected_df.iloc[i].values
        x_data.append(x)

    return np.array(x_data)


def make_spot_data(df: pd.DataFrame, window: int):
    columns = [
        "open_time",
        "close_time",
        "delta",
        "up_delta",
        "down_delta",
        "d10",
        "d20",
        "d50",
        "d200",
        "e_d10",
        "e_d20",
        "e_d50",
        "e_d200",
        "volume_delta",
        "volume_ratio",
        "ha_delta",
        "ha_up_delta",
        "ha_down_delta",
    ]
    x_data = []
    y_data = []

    selected_df = df[columns]
    for i in range(df.shape[0] - window):
        open = df.iloc[i + 1]["open"]
        close = df.iloc[i + window]["close"]
        if close > open:
            y_data.append(1)

        else:
            y_data.append(0)

        x = selected_df.iloc[i].values
        x_data.append(x)

    real_x_data = []
    real_x = selected_df.iloc[-1].values
    real_x_data.append(real_x)

    return np.array(x_data), np.array(y_data), np.array(real_x_data)


def x_data(last_row: pd.DataFrame):
    columns = [
        "open_time",
        "close_time",
        "delta",
        "up_delta",
        "down_delta",
        "d10",
        "d20",
        "d50",
        "d200",
        "e_d10",
        "e_d20",
        "e_d50",
        "e_d200",
        "volume_delta",
        "volume_ratio",
        "ha_delta",
        "ha_up_delta",
        "ha_down_delta",
    ]
    selected_data = last_row[columns]

    return np.array([selected_data.values])
