import numpy as np
import pandas as pd
from collections import deque


def clean_deque(i, k, deq, df, key, isHigh):
    if deq and deq[0] == i - k:
        deq.popleft()
    if isHigh:
        while deq and df.iloc[i][key] > df.iloc[deq[-1]][key]:
            deq.pop()
    else:
        while deq and df.iloc[i][key] < df.iloc[deq[-1]][key]:
            deq.pop()


def pivotPoints(pivot=None, data=None):
    """
    This function calculates the pivot points based on the pivot lenght.
    These can be HH, LH , HL, LL values based on the adjacent pivots
    which occur based on the length of the pivot.
    """

    data["PH"] = False
    data["PHV"] = np.nan
    data["PL"] = False
    data["PLV"] = np.nan
    keyHigh = "high"
    keyLow = "low"
    win_size = pivot + 1
    deqHigh = deque()
    deqLow = deque()
    max_idx = 0
    min_idx = 0
    i = 0
    j = pivot
    pivot_low = None
    pivot_high = None
    for index, row in data.iterrows():
        if i < win_size:
            clean_deque(i, win_size, deqHigh, data, keyHigh, True)
            clean_deque(i, win_size, deqLow, data, keyLow, False)
            deqHigh.append(i)
            deqLow.append(i)
            if data.iloc[i][keyHigh] > data.iloc[max_idx][keyHigh]:
                max_idx = i
            if data.iloc[i][keyLow] < data.iloc[min_idx][keyLow]:
                min_idx = i
            if i == win_size - 1:
                if data.iloc[max_idx][keyHigh] == data.iloc[j][keyHigh]:
                    data.at[data.index[j], "PH"] = True
                    pivot_high = data.iloc[j][keyHigh]
                if data.iloc[min_idx][keyLow] == data.iloc[j][keyLow]:
                    data.at[data.index[j], "PL"] = True
                    pivot_low = data.iloc[j][keyLow]
        if i >= win_size:
            j += 1
            clean_deque(i, win_size, deqHigh, data, keyHigh, True)
            clean_deque(i, win_size, deqLow, data, keyLow, False)
            deqHigh.append(i)
            deqLow.append(i)
            pivot_val = data.iloc[deqHigh[0]][keyHigh]
            if pivot_val == data.iloc[j][keyHigh]:
                data.at[data.index[j], "PH"] = True
                pivot_high = data.iloc[j][keyHigh]
            if data.iloc[deqLow[0]][keyLow] == data.iloc[j][keyLow]:
                data.at[data.index[j], "PL"] = True
                pivot_low = data.iloc[j][keyLow]

        data.at[data.index[j], "PHV"] = pivot_high
        data.at[data.index[j], "PLV"] = pivot_low
        i = i + 1

    return data


def cal_values(df: pd.DataFrame) -> pd.DataFrame:
    df["ma20"] = df["close"].rolling(window=20).mean()
    df["std20"] = df["close"].rolling(window=20).std()
    df["upper_bb"] = df["ma20"] + 2 * df["std20"]
    df["lower_bb"] = df["ma20"] - 2 * df["std20"]
    df["volume_ma50"] = df["volume"].rolling(window=50).mean()

    df["ma10"] = df["close"].rolling(window=10).mean()
    df["ma50"] = df["close"].rolling(window=50).mean()
    df["ma100"] = df["close"].rolling(window=100).mean()
    df["ma200"] = df["close"].rolling(window=200).mean()
    df["ema10"] = df["close"].ewm(alpha=2 / 11, adjust=False).mean()
    df["ema20"] = df["close"].ewm(alpha=2 / 21, adjust=False).mean()
    df["ema50"] = df["close"].ewm(alpha=2 / 51, adjust=False).mean()
    df["ema100"] = df["close"].ewm(alpha=2 / 101, adjust=False).mean()
    df["ema200"] = df["close"].ewm(alpha=2 / 201, adjust=False).mean()

    df["avg_price"] = (df["open"] + df["close"]) / 2

    pivots = pivotPoints(pivot=2, data=df)
    df["PH"] = pivots["PH"]
    df["PHV"] = pivots["PHV"]
    df["PL"] = pivots["PL"]
    df["PLV"] = pivots["PLV"]

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
    df["d100"] = df["close"] / df["ma100"]
    df["d200"] = df["close"] / df["ma200"]
    df["ed10"] = df["close"] / df["ema10"]
    df["ed20"] = df["close"] / df["ema20"]
    df["ed50"] = df["close"] / df["ema50"]
    df["ed100"] = df["close"] / df["ema100"]
    df["ed200"] = df["close"] / df["ema200"]

    df["dPHV"] = df["PHV"] / df["close"]
    df["dPLV"] = df["PLV"] / df["close"]

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

    # 하이킨아시 필요한 값
    df["ha_delta"] = df["ha_close"] / df["ha_open"]
    df["ha_up_delta"] = df["ha_high"] / df[["ha_open", "ha_close"]].max(axis=1)
    df["ha_down_delta"] = df["ha_low"] / df[["ha_open", "ha_close"]].min(axis=1)

    # 가격 극대값과 극소값 찾기
    price_max_peaks = (
        (df["avg_price"] > df["avg_price"].shift(1))
        & (df["avg_price"] > df["avg_price"].shift(-1))
        & (df["avg_price"] > df["avg_price"].shift(2))
        & (df["avg_price"] > df["avg_price"].shift(-2))
    )

    price_min_troughs = (
        (df["avg_price"] < df["avg_price"].shift(1))
        & (df["avg_price"] < df["avg_price"].shift(-1))
        & (df["avg_price"] < df["avg_price"].shift(2))
        & (df["avg_price"] < df["avg_price"].shift(-2))
    )

    df["peak"] = np.where(price_max_peaks, 1, np.where(price_min_troughs, 2, 0))

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
        days = 672
    else:
        days = 672

    for i in range(days, len(df)):
        use_cols = [
            "delta",
            "up_delta",
            "down_delta",
            "volume_delta",
            # "ed10",
            # "ed20",
            # "ed50",
        ]

        base_df = df.iloc[i - days : i]
        X_vector = base_df[use_cols].values.flatten()
        X_data.append(X_vector)

        entry = base_df.iloc[-1]["close"]
        ATR = base_df.iloc[-1]["ATR"]
        long_price = entry + 2 * ATR
        short_price = entry - 2 * ATR

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

    # # 0인 경우와 (1, 2인 경우) 따로 분리
    # X_data_0 = X_data[y_data == 0]
    # y_data_0 = y_data[y_data == 0]
    # X_data_non_0 = X_data[y_data != 0]
    # y_data_non_0 = y_data[y_data != 0]
    #
    # # non-zero 클래스 데이터 개수 계산
    # n_non_zero = len(y_data_non_0)
    #
    # # 0 클래스 데이터 개수를 1,2 클래스 데이터의 절반으로 설정
    # n_samples = int(n_non_zero / 2)
    #
    # # 0 클래스 데이터를 언더샘플링
    # X_data_0_resampled, y_data_0_resampled = resample(
    #     X_data_0, y_data_0, n_samples=n_samples, random_state=42
    # )
    #
    # # 샘플링된 데이터와 원래 (1, 2) 데이터를 다시 합침
    # X_data_resampled = np.concatenate([X_data_0_resampled, X_data_non_0])
    # y_data_resampled = np.concatenate([y_data_0_resampled, y_data_non_0])


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
        # "ed10",
        # "ed20",
        # "ed50",
    ]

    X_vector = df.iloc[i - days : i][use_cols].values.flatten()
    X_data.append(X_vector)
    X_data = np.array(X_data)

    return X_data
