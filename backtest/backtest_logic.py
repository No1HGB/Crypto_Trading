import pandas as pd


def bb_long(df: pd.DataFrame, i) -> bool:
    return df.at[i, "low"] <= df.at[i, "lower_bb"]


def bb_short(df: pd.DataFrame, i) -> bool:
    return df.at[i, "high"] >= df.at[i, "upper_bb"]


def trend_long(df: pd.DataFrame, i) -> bool:

    def is_trend(df: pd.DataFrame, t) -> bool:
        return (
            df.at[t, "ema10"] > df.at[t, "ema20"] > df.at[t, "ema50"]
            and df.at[t, "ma10"] > df.at[t, "ma20"] > df.at[t, "ma50"]
        ) and df.at[t, "close"] > df.at[t, "upper_bb"]

    return is_trend(df, i) or is_trend(df, i - 1) or is_trend(df, i - 2)


def trend_short(df: pd.DataFrame, i) -> bool:

    def is_trend(df: pd.DataFrame, t) -> bool:
        return (
            df.at[t, "ema10"] < df.at[t, "ema20"] < df.at[t, "ema50"]
            and df.at[t, "ma10"] < df.at[t, "ma20"] < df.at[t, "ma50"]
        ) and df.at[t, "close"] < df.at[t, "lower_bb"]

    return is_trend(df, i) or is_trend(df, i - 1) or is_trend(df, i - 2)
