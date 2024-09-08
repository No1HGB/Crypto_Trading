import numpy as np
import pandas as pd


def cal_values(df: pd.DataFrame) -> pd.DataFrame:
    df["ma20"] = df["close"].rolling(window=20).mean()
    df["std20"] = df["close"].rolling(window=20).std()
    df["upper_bb"] = df["ma20"] + 2 * df["std20"]
    df["lower_bb"] = df["ma20"] - 2 * df["std20"]
    df["volume_ma50"] = df["volume"].rolling(window=50).mean()
    df["volume_delta"] = df["volume"] / df["volume_ma50"]

    df["ma4"] = df["close"].rolling(window=4).mean()
    df["std4"] = df["close"].rolling(window=4).std()
    df["upper_bb4"] = df["ma20"] + 4 * df["std4"]
    df["lower_bb4"] = df["ma20"] - 4 * df["std4"]

    # 필요한 값 계산
    df["delta"] = df["close"] / df["open"]
    df["up_delta"] = df["high"] / df[["open", "close"]].max(axis=1)
    df["down_delta"] = df["low"] / df[["open", "close"]].min(axis=1)
    df["d20"] = df["close"] / df["ma20"]
    df["dup"] = df["close"] / df["upper_bb"]
    df["dlow"] = df["close"] / df["lower_bb"]

    df["d4"] = df["close"] / df["ma4"]
    df["dup4"] = df["close"] / df["upper_bb4"]
    df["dlow4"] = df["close"] / df["lower_bb4"]

    df.drop(
        ["volume", "ma20", "std20", "volume_ma50", "upper_bb", "lower_bb"],
        axis=1,
        inplace=True,
    )
    df.dropna(axis=0, inplace=True, how="any")
    df.reset_index(drop=True, inplace=True)

    return df


def make_data(df, symbol, n=1):
    X_data = []
    y_data = []
    days = 1
    ratio = 0.1
    if symbol == "BTCUSDT":
        days = 13
        ratio = 1
    elif symbol == "ETHUSDT":
        days = 25
        ratio = 1

    for i in range(days, len(df) - n):
        use_cols = [
            "delta",
            # "up_delta",
            # "down_delta",
            "d20",
            "dup",
            "dlow",
            "d4",
            "dup4",
            "dlow4",
            "volume_delta",
        ]
        if df.loc[i, "volume_delta"] >= 1:

            X_vector = df.iloc[i - days : i][use_cols].values.flatten()
            X_data.append(X_vector)

            # y_data 구성
            entry = df.loc[i, "close"]
            up = down = None

            # 다음 행부터 검사
            for j in range(i + 1, len(df)):
                future_high = df.loc[j, "high"]
                future_low = df.loc[j, "low"]

                up = (future_high - entry) / entry * 100
                down = (future_low - entry) / entry * 100

                # 검사 조건
                if up >= ratio:
                    y_data.append(1)
                    break
                elif down <= -ratio:
                    y_data.append(0)
                    break

            # 만약 검사가 완료되지 않았다면 X_data에서 마지막 벡터 제거
            if len(y_data) < len(X_data):
                X_data = X_data[: len(y_data)]

    X_data = np.array(X_data)
    y_data = np.array(y_data)

    return X_data, y_data


def make_data_small(df, symbol, n=1):
    X_data = []
    y_data = []
    days = 1
    ratio = 0.1
    if symbol == "BTCUSDT":
        days = 13
        ratio = 0.5
    elif symbol == "ETHUSDT":
        days = 25
        ratio = 0.5

    for i in range(days, len(df) - n):
        use_cols = [
            "delta",
            # "up_delta",
            # "down_delta",
            "d20",
            "dup",
            "dlow",
            "d4",
            "dup4",
            "dlow4",
        ]
        if df.loc[i, "volume_delta"] < 1:

            X_vector = df.iloc[i - days : i][use_cols].values.flatten()
            X_data.append(X_vector)

            # y_data 구성
            entry = df.loc[i, "close"]
            up = down = None

            # 다음 행부터 검사
            for j in range(i + 1, len(df)):
                future_high = df.loc[j, "high"]
                future_low = df.loc[j, "low"]

                up = (future_high - entry) / entry * 100
                down = (future_low - entry) / entry * 100

                # 검사 조건
                if up >= ratio:
                    y_data.append(1)
                    break
                elif down <= -ratio:
                    y_data.append(0)
                    break

            # 만약 검사가 완료되지 않았다면 X_data에서 마지막 벡터 제거
            if len(y_data) < len(X_data):
                X_data = X_data[: len(y_data)]

    X_data = np.array(X_data)
    y_data = np.array(y_data)

    return X_data, y_data


def x_data(df: pd.DataFrame, symbol: str, is_small: bool):
    days = 1
    if symbol == "BTCUSDT" and not is_small:
        days = 13
    elif symbol == "BTCUSDT" and is_small:
        days = 13
    elif symbol == "ETHUSDT" and not is_small:
        days = 25
    elif symbol == "ETHUSDT" and is_small:
        days = 25

    X_data = []
    use_cols = [
        "delta",
        # "up_delta",
        # "down_delta",
        "d20",
        "dup",
        "dlow",
        "d4",
        "dup4",
        "dlow4",
        "volume_delta",
    ]
    use_cols_small = [
        "delta",
        # "up_delta",
        # "down_delta",
        "d20",
        "dup",
        "dlow",
        "d4",
        "dup4",
        "dlow4",
    ]
    if not is_small:
        X_vector = df.iloc[-days:][use_cols].values.flatten()
    else:
        X_vector = df.iloc[-days:][use_cols_small].values.flatten()
    X_data.append(X_vector)
    X_data = np.array(X_data)

    return X_data
