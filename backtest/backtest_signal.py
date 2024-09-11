import numpy as np
import pandas as pd


# ALMA Smoothing
def alma(series, length=25, sigma=7, offset=0.85):
    m = offset * (length - 1)
    s = length / sigma
    weights = np.exp(-np.square(np.arange(length) - m) / (2 * s * s))
    weights /= np.sum(weights)
    return np.convolve(series, weights, mode="same")


def rsi(series, length=14):
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=length).mean()
    avg_loss = pd.Series(loss).rolling(window=length).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def chande_momentum_oscillator(series, length=9):
    momentum = series.diff(1)
    pos_mom = np.where(momentum > 0, momentum, 0)
    neg_mom = np.where(momentum < 0, -momentum, 0)
    sum_pos = pd.Series(pos_mom).rolling(window=length).sum()
    sum_neg = pd.Series(neg_mom).rolling(window=length).sum()
    return 100 * (sum_pos - sum_neg) / (sum_pos + sum_neg)


def gaussian_moving_average(series, length=14, volatility_period=20, adaptive=True):
    if adaptive:
        sigma = series.rolling(volatility_period).std()
    else:
        sigma = 1.0
    weights = [
        np.exp(-(((i - (length - 1)) / (2 * sigma.iloc[-1])) ** 2) / 2)
        for i in range(length)
    ]
    weights /= np.sum(weights)

    gma = np.convolve(series, weights, mode="same")
    return pd.Series(gma).ewm(span=7).mean()
