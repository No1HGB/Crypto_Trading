import config


def cal_stop_price(entryPrice, side, symbol):
    if symbol == "BTCUSDT":
        stop_ratio = config.sl_btc
    else:
        stop_ratio = config.sl_eth

    if side == "BUY":
        stopPrice = entryPrice * (1 - stop_ratio / 100)
    else:
        stopPrice = entryPrice * (1 + stop_ratio / 100)

    if symbol == "BTCUSDT":
        stopPrice = round(stopPrice, 1)
    else:
        stopPrice = round(stopPrice, 2)

    return stopPrice


def cal_profit_price(entryPrice, side, symbol):
    if symbol == "BTCUSDT":
        profit_ratio = config.tp_btc
    else:
        profit_ratio = config.tp_eth

    if side == "BUY":
        stopPrice = entryPrice * (1 + profit_ratio / 100)
    else:
        stopPrice = entryPrice * (1 - profit_ratio / 100)

    if symbol == "BTCUSDT":
        stopPrice = round(stopPrice, 1)
    else:
        stopPrice = round(stopPrice, 2)

    return stopPrice
