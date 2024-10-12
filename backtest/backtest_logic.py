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


def simple_trend_long(df: pd.DataFrame, i) -> bool:
    return (
        df.at[i, "ema10"] > df.at[i, "ema20"] > df.at[i, "ema50"]
        or df.at[i, "ma10"] > df.at[i, "ma20"] > df.at[i, "ma50"]
    )


def simple_trend_short(df: pd.DataFrame, i) -> bool:
    return (
        df.at[i, "ema10"] < df.at[i, "ema20"] < df.at[i, "ema50"]
        or df.at[i, "ma10"] < df.at[i, "ma20"] < df.at[i, "ma50"]
    )


def ha_trend_long(df: pd.DataFrame, i, v_coff) -> bool:

    return (
        df.at[i, "ema10"] > df.at[i, "ema20"] > df.at[i, "ema50"]
        and df.at[i, "volume"] >= df.at[i, "volume_ma"] * v_coff
        and df.at[i, "ha_close"] > df.at[i, "ha_open"]
        and (
            df.at[i - 1, "ha_close"] < df.at[i - 1, "ha_open"]
            or df.at[i - 2, "ha_close"] < df.at[i - 2, "ha_open"]
        )
    )


def ha_trend_short(df: pd.DataFrame, i, v_coff) -> bool:

    return (
        df.at[i, "ema10"] < df.at[i, "ema20"] < df.at[i, "ema50"]
        and df.at[i, "volume"] >= df.at[i, "volume_ma"] * v_coff
        and df.at[i, "ha_close"] < df.at[i, "ha_open"]
        and (
            df.at[i - 1, "ha_close"] > df.at[i - 1, "ha_open"]
            or df.at[i - 2, "ha_close"] > df.at[i - 2, "ha_open"]
        )
    )


def ha_long(df: pd.DataFrame, i, v_coff) -> bool:
    return (
        df.at[i, "ha_close"] > df.at[i, "ha_open"]
        and df.at[i, "volume"] >= df.at[i, "volume_ma"] * v_coff
        and df.at[i, "open"] < max(df.at[i, "ema20"], df.at[i, "ema50"])
        and df.at[i, "close"] > df.at[i, "open"]
        and df.at[i, "avg_price"] > df.at[i - 1, "avg_price"]
    )


def ha_short(df: pd.DataFrame, i, v_coff) -> bool:
    return (
        df.at[i, "ha_close"] < df.at[i, "ha_open"]
        and df.at[i, "volume"] >= df.at[i, "volume_ma"] * v_coff
        and df.at[i, "open"] > min(df.at[i, "ema20"], df.at[i, "ema50"])
        and df.at[i, "close"] < df.at[i, "open"]
        and df.at[i, "avg_price"] < df.at[i - 1, "avg_price"]
    )
