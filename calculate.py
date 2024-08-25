import config


def cal_stop_price(entryPrice, side, symbol):
    stop_ratio = config.stop_ratio

    if side == "BUY":
        stopPrice = entryPrice * (1 - stop_ratio / 100)

    elif side == "SELL":
        stopPrice = entryPrice * (1 + stop_ratio / 100)

    if symbol == "BTCUSDT":
        stopPrice = round(stopPrice, 1)
    elif symbol == "ETHUSDT":
        stopPrice = round(stopPrice, 2)

    return stopPrice
