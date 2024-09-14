import pandas as pd


def entry_long(df: pd.DataFrame, i) -> bool:
    second_down_delta = df.at[i - 1, "lower_bb"] - df.at[i - 1, "lower_bb4"]
    last_down_delta = df.at[i, "lower_bb"] - df.at[i, "lower_bb4"]

    if (
        df.at[i, "close"] > df.at[i, "open"]
        and df.at[i - 1, "close"] < df.at[i - 1, "open"]
        and df.at[i, "volume_delta"] > 1
    ):
        return (
            second_down_delta > 0
            and last_down_delta > 0
            and df.at[i, "close"] > df.at[i, "lower_bb"]
            and df.at[i, "close"] < df.at[i, "ma20"]
        )

    return False


def close_long(df: pd.DataFrame, i) -> bool:
    second_up_delta = df.at[i - 1, "upper_bb4"] - df.at[i - 1, "upper_bb"]
    last_up_delta = df.at[i, "upper_bb4"] - df.at[i, "upper_bb"]

    if last_up_delta > 0 and second_up_delta < 0:
        return df.at[i, "close"] > df.at[i, "ma20"]

    return False


def entry_short(df: pd.DataFrame, i) -> bool:
    second_up_delta = df.at[i - 1, "upper_bb4"] - df.at[i - 1, "upper_bb"]
    last_up_delta = df.at[i, "upper_bb4"] - df.at[i, "upper_bb"]

    if (
        df.at[i, "close"] < df.at[i, "open"]
        and df.at[i - 1, "close"] > df.at[i - 1, "open"]
        and df.at[i, "volume_delta"] > 1
    ):
        return (
            second_up_delta > 0
            and last_up_delta > 0
            and df.at[i, "close"] < df.at[i, "upper_bb"]
            and df.at[i, "close"] > df.at[i, "ma20"]
        )

    return False


def close_short(df: pd.DataFrame, i) -> bool:
    second_down_delta = df.at[i - 1, "lower_bb"] - df.at[i - 1, "lower_bb4"]
    last_down_delta = df.at[i, "lower_bb"] - df.at[i, "lower_bb4"]

    if last_down_delta > 0 and second_down_delta < 0:
        return df.at[i, "close"] < df.at[i, "ma20"]

    return False


def ha_long(df: pd.DataFrame, i) -> bool:
    if df.at[i, "volume_delta"] > 1:
        return df.at[i, "avg_price"] > df.at[i - 1, "avg_price"]
    return False


def ha_short(df: pd.DataFrame, i) -> bool:
    if df.at[i, "volume_delta"] > 1:
        return df.at[i, "avg_price"] < df.at[i - 1, "avg_price"]
    return False


def bb_long(df: pd.DataFrame, i) -> bool:

    if (
        df.at[i, "close"] > df.at[i, "open"]
        and df.at[i - 1, "close"] < df.at[i - 1, "open"]
    ):
        return (
            df.at[i, "close"] > df.at[i, "lower_bb"] > df.at[i, "low"]
            and df.at[i - 1, "open"] > df.at[i, "lower_bb"] > df.at[i - 1, "low"]
        )
    return False


def bb_short(df: pd.DataFrame, i) -> bool:

    if (
        df.at[i, "close"] < df.at[i, "open"]
        and df.at[i - 1, "close"] > df.at[i - 1, "open"]
    ):
        return (
            df.at[i, "high"] > df.at[i, "upper_bb"] > df.at[i, "close"]
            and df.at[i - 1, "high"] > df.at[i, "upper_bb"] > df.at[i - 1, "open"]
        )
    return False


def trend_long(df: pd.DataFrame, i) -> bool:

    up_max = df.iloc[i - 7 : i][["open", "close"]].max().max()

    if df.at[i, "close"] > df.at[i, "open"]:
        return (
            df.at[i, "close"] > up_max
            and df.at[i, "ema10"] > df.at[i, "ema20"] > df.at[i, "ema50"]
        )

    return False


def trend_short(df: pd.DataFrame, i) -> bool:

    down_min = df.iloc[i - 7 : i][["open", "close"]].min().min()

    if df.at[i, "close"] < df.at[i, "open"]:
        return (
            df.at[i, "close"] < down_min
            and df.at[i, "ema10"] < df.at[i, "ema20"] < df.at[i, "ema50"]
        )

    return False
