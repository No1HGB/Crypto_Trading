import pandas as pd


def trend_long(df: pd.DataFrame) -> bool:
    last_row = df.iloc[-1]
    up_max = df.iloc[-7:-1][["open", "close"]].max().max()

    if last_row["close"] > last_row["open"]:
        return (
            last_row["close"] > up_max
            and last_row["ema10"] > last_row["ema20"] > last_row["ema50"]
        )

    return False


def trend_short(df: pd.DataFrame) -> bool:
    last_row = df.iloc[-1]
    down_min = df.iloc[7:-1][["open", "close"]].min().min()

    if last_row["close"] < last_row["open"]:
        return (
            last_row["close"] < down_min
            and last_row["ema10"] < last_row["ema20"] < last_row["ema50"]
        )

    return False
