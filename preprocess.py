import numpy as np
import pandas as pd


def cal_values(df: pd.DataFrame) -> pd.DataFrame:
    df["ma4"] = df["close"].rolling(window=4).mean()
    df["std4"] = df["close"].rolling(window=4).std()
    df["upper_bb4"] = df["ma4"] + 4 * df["std4"]
    df["lower_bb4"] = df["ma4"] - 4 * df["std4"]
    df["ma20"] = df["close"].rolling(window=20).mean()
    df["std20"] = df["close"].rolling(window=20).std()
    df["upper_bb"] = df["ma20"] + 2 * df["std20"]
    df["lower_bb"] = df["ma20"] - 2 * df["std20"]
    df["volume_ma50"] = df["volume"].rolling(window=50).mean()

    df["ma10"] = df["close"].rolling(window=10).mean()
    df["ma50"] = df["close"].rolling(window=50).mean()
    df["ma200"] = df["close"].rolling(window=200).mean()
    df["ema10"] = df["close"].ewm(alpha=2 / 11, adjust=False).mean()
    df["ema20"] = df["close"].ewm(alpha=2 / 21, adjust=False).mean()
    df["ema50"] = df["close"].ewm(alpha=2 / 51, adjust=False).mean()
    df["ema200"] = df["close"].ewm(alpha=2 / 201, adjust=False).mean()

    # 필요한 값 계산
    df["delta"] = df["close"] / df["open"]
    df["up_delta"] = df["high"] / df[["open", "close"]].max(axis=1)
    df["down_delta"] = df["low"] / df[["open", "close"]].min(axis=1)

    df["d4"] = df["close"] / df["ma4"]
    df["dup4"] = df["close"] / df["upper_bb4"]
    df["dlow4"] = df["close"] / df["lower_bb4"]

    df["d20"] = df["close"] / df["ma20"]
    df["dup"] = df["close"] / df["upper_bb"]
    df["dlow"] = df["close"] / df["lower_bb"]

    df["volume_delta"] = df["volume"] / df["volume_ma50"]

    df["d10"] = df["close"] / df["ma10"]
    df["d50"] = df["close"] / df["ma50"]
    df["d200"] = df["close"] / df["ma200"]
    df["ed10"] = df["close"] / df["ema10"]
    df["ed20"] = df["close"] / df["ema20"]
    df["ed50"] = df["close"] / df["ema50"]
    df["ed200"] = df["close"] / df["ema200"]

    df.drop(
        ["volume", "std4", "std20"],
        axis=1,
        inplace=True,
    )
    df.dropna(axis=0, inplace=True, how="any")
    df.reset_index(drop=True, inplace=True)

    return df


def x_data(df: pd.DataFrame, symbol: str):
    if symbol == "BTCUSDT":
        days = 24
    else:
        days = 24

    X_data = []
    use_cols = [
        "delta",
        "up_delta",
        "down_delta",
        "d20",
        "dup",
        "dlow",
        "volume_delta",
        "d10",
        "d50",
        "d200",
        "ed10",
        "ed20",
        "ed50",
        "ed200",
    ]

    X_vector = df.iloc[-days:][use_cols].values.flatten()
    X_data.append(X_vector)
    X_data = np.array(X_data)

    return X_data


def make_data(df, symbol, n=6):
    X_data = []
    y_data = []

    if symbol == "BTCUSDT":
        days = 24
    else:
        days = 24

    for i in range(days, len(df) - n):
        use_cols = [
            "delta",
            "up_delta",
            "down_delta",
            "d20",
            "dup",
            "dlow",
            "volume_delta",
            "d10",
            "d50",
            "d200",
            "ed10",
            "ed20",
            "ed50",
            "ed200",
        ]

        X_vector = df.iloc[i - days : i][use_cols].values.flatten()
        X_data.append(X_vector)

        if df.iloc[i]["close"] < df.iloc[i + 6]["close"]:
            y_data.append(1)
        else:
            y_data.append(0)

    X_data = np.array(X_data)
    y_data = np.array(y_data)

    return X_data, y_data


def x_data_backtest(df: pd.DataFrame, symbol: str, i):
    if symbol == "BTCUSDT":
        days = 24
    else:
        days = 24

    X_data = []
    use_cols = [
        "delta",
        "up_delta",
        "down_delta",
        "d20",
        "dup",
        "dlow",
        "volume_delta",
        "d10",
        "d50",
        "d200",
        "ed10",
        "ed20",
        "ed50",
        "ed200",
    ]

    # i는 24부터
    X_vector = df.iloc[i - days : i][use_cols].values.flatten()
    X_data.append(X_vector)
    X_data = np.array(X_data)

    return X_data
