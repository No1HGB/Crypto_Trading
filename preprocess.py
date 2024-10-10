import numpy as np
import pandas as pd


def cal_values(df: pd.DataFrame) -> pd.DataFrame:
    df["ma20"] = df["close"].rolling(window=20).mean()
    df["std20"] = df["close"].rolling(window=20).std()
    df["upper_bb"] = df["ma20"] + 2 * df["std20"]
    df["lower_bb"] = df["ma20"] - 2 * df["std20"]
    df["volume_ma"] = df["volume"].rolling(window=50).mean()

    df["ema10"] = df["close"].ewm(alpha=2 / 11, adjust=False).mean()
    df["ema20"] = df["close"].ewm(alpha=2 / 21, adjust=False).mean()
    df["ema50"] = df["close"].ewm(alpha=2 / 51, adjust=False).mean()
    df["ema100"] = df["close"].ewm(alpha=2 / 101, adjust=False).mean()
    df["ema200"] = df["close"].ewm(alpha=2 / 201, adjust=False).mean()

    # 필요한 값 계산
    df["delta"] = df["close"] / df["open"]
    df["up_delta"] = df["high"] / df[["open", "close"]].max(axis=1)
    df["down_delta"] = df["low"] / df[["open", "close"]].min(axis=1)

    df["d20"] = df["close"] / df["ma20"]
    df["dup"] = df["close"] / df["upper_bb"]
    df["dlow"] = df["close"] / df["lower_bb"]

    df["volume_delta"] = df["volume"] / df["volume_ma"]

    df["ed10"] = df["close"] / df["ema10"]
    df["ed20"] = df["close"] / df["ema20"]
    df["ed50"] = df["close"] / df["ema50"]
    df["ed100"] = df["close"] / df["ema100"]
    df["ed200"] = df["close"] / df["ema200"]

    # 피봇 포인트
    df["prev_high"] = df["high"].shift(1)
    df["prev_low"] = df["low"].shift(1)
    df["prev_close"] = df["close"].shift(1)
    df["pivot"] = (df["prev_high"] + df["prev_low"] + df["prev_close"]) / 3
    df["R1"] = (2 * df["pivot"]) - df["prev_low"]
    df["S1"] = (2 * df["pivot"]) - df["prev_high"]
    df["R2"] = df["pivot"] + (df["prev_high"] - df["prev_low"])
    df["S2"] = df["pivot"] - (df["prev_high"] - df["prev_low"])
    df["R3"] = df["pivot"] + 2 * (df["prev_high"] - df["prev_low"])
    df["S3"] = df["pivot"] - 2 * (df["prev_high"] - df["prev_low"])

    df["pivot_delta"] = df["pivot"] / df["close"]
    df["R1_delta"] = df["close"] / df["R1"]
    df["S1_delta"] = df["close"] / df["S1"]
    df["R2_delta"] = df["close"] / df["R2"]
    df["S2_delta"] = df["close"] / df["S2"]
    df["R3_delta"] = df["close"] / df["R3"]
    df["S3_delta"] = df["close"] / df["S3"]

    # Peak 계산
    df["rolling_max_high"] = (
        df["high"].rolling(window=100, min_periods=1).max().shift(1)
    )
    df["rolling_min_low"] = df["low"].rolling(window=100, min_periods=1).min().shift(1)
    df["swing_high"] = df["high"][(df["high"] > df["rolling_max_high"])]
    df["swing_low"] = df["low"][(df["low"] < df["rolling_min_low"])]
    df["resistance"] = df["swing_high"].ffill()
    df["support"] = df["swing_low"].ffill()
    df.drop(
        ["rolling_max_high", "rolling_min_low", "swing_high", "swing_low"],
        axis=1,
        inplace=True,
    )

    df["resistance_delta"] = df["close"] / df["resistance"]
    df["support_delta"] = df["close"] / df["support"]

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

    df["ha_delta"] = df["ha_close"] / df["ha_open"]
    df["ha_up_delta"] = df["ha_high"] / df[["ha_open", "ha_close"]].max(axis=1)
    df["ha_down_delta"] = df["ha_low"] / df[["ha_open", "ha_close"]].min(axis=1)

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
        "volume_delta",
        "ed10",
        "ed20",
        "ed50",
        "ed100",
        "ed200",
        "dup",
        "dlow",
        "PH",
        "dPHV",
        "PL",
        "dPLV",
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
        n = 2
    else:
        days = 48
        n = 2

    for i in range(days, len(df) - n):
        use_cols = [
            "delta",
            "up_delta",
            "down_delta",
            "volume_delta",
            "peak",
            "ed10",
            "ed20",
            "ed50",
            "ed100",
            "dup",
            "dlow",
        ]

        X_vector = df.iloc[i - days : i][use_cols].values.flatten()
        X_data.append(X_vector)

        if (
            df.iloc[i]["ha_open"] < df.iloc[i]["ha_close"]
            and df.iloc[i + 1]["ha_open"] < df.iloc[i + 1]["ha_close"]
        ):
            y_data.append(2)
        elif (
            df.iloc[i]["ha_open"] > df.iloc[i]["ha_close"]
            and df.iloc[i + 1]["ha_open"] > df.iloc[i + 1]["ha_close"]
        ):
            y_data.append(1)
        else:
            y_data.append(0)

    X_data = np.array(X_data)
    y_data = np.array(y_data)

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
        "volume_delta",
        "peak",
        "ed10",
        "ed20",
        "ed50",
        "ed100",
        "dup",
        "dlow",
    ]

    # i는 24부터
    X_vector = df.iloc[i - days : i][use_cols].values.flatten()
    X_data.append(X_vector)
    X_data = np.array(X_data)

    return X_data


def make_data_v3(df, symbol):
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
            "volume_delta",
            "ed10",
            "ed20",
            "ed50",
            "ed100",
            "ed200",
            "d20",
            "dup",
            "dlow",
            # "PH",
            # "dPHV",
            # "PL",
            # "dPLV",
        ]

        X_vector = df.iloc[i - days : i][use_cols].values.flatten()
        X_data.append(X_vector)

        if df.iloc[i]["open"] < df.iloc[i + 2]["close"]:
            y_data.append(1)
        elif df.iloc[i]["open"] > df.iloc[i + 2]["close"]:
            y_data.append(0)
        else:
            y_data.append(-1)

    X_data = np.array(X_data)
    y_data = np.array(y_data)

    slice_indices = y_data != -1
    X_data = X_data[slice_indices]
    y_data = y_data[slice_indices]

    return X_data, y_data


def x_data_backtest_v3(df: pd.DataFrame, symbol: str, i):
    if symbol == "BTCUSDT":
        days = 48
    else:
        days = 48

    X_data = []
    use_cols = [
        "delta",
        "up_delta",
        "down_delta",
        "volume_delta",
        "ed10",
        "ed20",
        "ed50",
        "ed100",
        "ed200",
        "dup",
        "dlow",
        "PH",
        "dPHV",
        "PL",
        "dPLV",
    ]

    X_vector = df.iloc[i - days : i][use_cols].values.flatten()
    X_data.append(X_vector)
    X_data = np.array(X_data)

    return X_data


def make_data_v4(df, symbol):
    X_data = []
    y_data = []

    if symbol == "BTCUSDT":
        days = 48
    else:
        days = 48

    for i in range(days, len(df)):
        use_cols = [
            "delta",
            "up_delta",
            "down_delta",
            "volume_delta",
            "ed10",
            "ed20",
            "ed50",
            "ed100",
            "ed200",
            "d20",
            "dup",
            "dlow",
            "pivot_delta",
            "R1_delta",
            "S1_delta",
            "R2_delta",
            "S2_delta",
            "R3_delta",
            "S3_delta",
            "resistance_delta",
            "support_delta",
            "ha_delta",
            "ha_up_delta",
            "ha_down_delta",
        ]

        base_df = df.iloc[i - days : i]
        X_vector = base_df[use_cols].values.flatten()
        X_data.append(X_vector)

        entry = base_df.iloc[-1]["close"]
        ATR = base_df.iloc[-1]["ATR"]
        long_price = entry + 1.5 * ATR
        short_price = entry - 1.5 * ATR

        future_df = df.iloc[i:]
        target = None

        for index, row in future_df.iterrows():
            if row["high"] > long_price:
                target = 1
                break
            elif row["low"] < short_price:
                target = 0
                break

        if target is not None:
            y_data.append(target)
        else:
            y_data.append(-1)

    X_data = np.array(X_data)
    y_data = np.array(y_data)

    slice_indices = y_data != -1
    X_data = X_data[slice_indices]
    y_data = y_data[slice_indices]

    # 결과 반환
    return X_data, y_data


def x_data_backtest_v4(df: pd.DataFrame, symbol: str, i):
    if symbol == "BTCUSDT":
        days = 48
    else:
        days = 48

    X_data = []
    use_cols = [
        "delta",
        "up_delta",
        "down_delta",
        "volume_delta",
        "ed10",
        "ed20",
        "ed50",
        "ed100",
        "ed200",
        "d20",
        "dup",
        "dlow",
        "pivot_delta",
        "R1_delta",
        "S1_delta",
        "R2_delta",
        "S2_delta",
        "R3_delta",
        "S3_delta",
        "resistance_delta",
        "support_delta",
        "ha_delta",
        "ha_up_delta",
        "ha_down_delta",
    ]

    X_vector = df.iloc[i - days : i][use_cols].values.flatten()
    X_data.append(X_vector)
    X_data = np.array(X_data)

    return X_data
