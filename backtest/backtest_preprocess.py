import pandas as pd

from backtest_signal import (
    alma,
    rsi,
    chande_momentum_oscillator,
    gaussian_moving_average,
)


def cal_value(df: pd.DataFrame) -> pd.DataFrame:
    smoothing = 72
    lookback = 15
    volatility_period = 5
    std = 5

    df["pchange"] = df["close"].diff(smoothing) / df["close"] * 100
    df["avpchange"] = alma(df["pchange"], length=lookback, sigma=std)
    df["rsi"] = rsi(df["close"], length=14)
    df["rsiL"] = df["rsi"] > df["rsi"].shift(1)
    df["rsiS"] = df["rsi"] < df["rsi"].shift(1)
    df["chandeMO"] = chande_momentum_oscillator(df["close"], length=9)
    df["cL"] = df["chandeMO"] > df["chandeMO"].shift(1)
    df["cS"] = df["chandeMO"] < df["chandeMO"].shift(1)
    df["gma"] = gaussian_moving_average(
        df["avpchange"], length=14, volatility_period=volatility_period
    )
    df["buy_signal"] = (df["avpchange"] > df["gma"]) & (
        df["avpchange"].shift(1) <= df["gma"].shift(1)
    )
    df["sell_signal"] = (df["avpchange"] < df["gma"]) & (
        df["avpchange"].shift(1) >= df["gma"].shift(1)
    )

    return df
