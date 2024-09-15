import pandas as pd


def trend_long(df: pd.DataFrame) -> bool:
    last_row = df.iloc[-1]
    second_last_row = df.iloc[-2]
    third_last_row = df.iloc[-3]

    def is_trend(row: pd.DataFrame) -> bool:
        return (
            row["close"] > row["upper_bb"]
            and row["ema10"] > row["ema20"] > row["ema50"]
            and row["ma10"] > row["ma20"] > row["ma50"]
        )

    return is_trend(last_row) or is_trend(second_last_row) or is_trend(third_last_row)


def trend_short(df: pd.DataFrame) -> bool:
    last_row = df.iloc[-1]
    second_last_row = df.iloc[-2]
    third_last_row = df.iloc[-3]

    def is_trend(row: pd.DataFrame) -> bool:
        return (
            row["close"] < row["lower_bb"]
            and row["ema10"] < row["ema20"] < row["ema50"]
            and row["ma10"] < row["ma20"] < row["ma50"]
        )

    return is_trend(last_row) or is_trend(second_last_row) or is_trend(third_last_row)
