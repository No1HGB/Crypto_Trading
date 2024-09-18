import numpy as np
import pandas as pd


def cal_values(df: pd.DataFrame) -> pd.DataFrame:
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

    # ATR 계산
    df["previous_close"] = df["close"].shift(1)
    df["high_low"] = df["high"] - df["low"]
    df["high_pc"] = np.abs(df["high"] - df["previous_close"])
    df["low_pc"] = np.abs(df["low"] - df["previous_close"])
    df["TR"] = df[["high_low", "high_pc", "low_pc"]].max(axis=1)
    df["ATR"] = df["TR"].rolling(window=14).mean()

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

    df.dropna(axis=0, inplace=True, how="any")
    df.reset_index(drop=True, inplace=True)

    return df


def x_data(df: pd.DataFrame, symbol: str):
    if symbol == "BTCUSDT":
        days = 48
    else:
        days = 48

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


"""
모델 훈련 및 백테스팅
"""


def make_data(df, symbol):
    X_data = []
    y_data = []

    if symbol == "BTCUSDT":
        days = 48
        n = 2
    else:
        days = 48
        n = 2

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

        if (
            df.iloc[i + 1]["ha_open"] < df.iloc[i + 1]["ha_close"]
            and df.iloc[i + 2]["ha_open"] < df.iloc[i + 2]["ha_close"]
        ):
            y_data.append(2)
        elif (
            df.iloc[i + 1]["ha_open"] > df.iloc[i + 1]["ha_close"]
            and df.iloc[i + 2]["ha_open"] > df.iloc[i + 2]["ha_close"]
        ):
            y_data.append(1)
        else:
            y_data.append(0)

    X_data = np.array(X_data)
    y_data = np.array(y_data)

    return X_data, y_data


def x_data_backtest(df: pd.DataFrame, symbol: str, i):
    if symbol == "BTCUSDT":
        days = 48
    else:
        days = 48

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


def make_data_v2(df, symbol):
    X_data = []
    y_data = []

    if symbol == "BTCUSDT":
        days = 48
        n = 3
    else:
        days = 48
        n = 3

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

        if (
            df.iloc[i]["ha_open"] > df.iloc[i]["ha_close"]
            and df.iloc[i + 1]["ha_open"] < df.iloc[i + 1]["ha_close"]
            and df.iloc[i + 2]["ha_open"] < df.iloc[i + 2]["ha_close"]
            and df.iloc[i + 3]["ha_open"] < df.iloc[i + 3]["ha_close"]
        ):
            y_data.append(1)
        elif (
            df.iloc[i]["ha_open"] < df.iloc[i]["ha_close"]
            and df.iloc[i + 1]["ha_open"] > df.iloc[i + 1]["ha_close"]
            and df.iloc[i + 2]["ha_open"] > df.iloc[i + 2]["ha_close"]
            and df.iloc[i + 3]["ha_open"] > df.iloc[i + 3]["ha_close"]
        ):
            y_data.append(0)
        else:
            y_data.append(-1)

    X_data = np.array(X_data)
    y_data = np.array(y_data)

    slice_indices = y_data != -1
    X_data = X_data[slice_indices]
    y_data = y_data[slice_indices]

    return X_data, y_data


def x_data_backtest_v2(df: pd.DataFrame, symbol: str, i):
    if symbol == "BTCUSDT":
        days = 48
    else:
        days = 48

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
