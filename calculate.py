import config


def cal_stop_price(entryPrice, side, symbol, is_small):
    if not is_small:
        stop_ratio = config.sl
    else:
        stop_ratio = config.sl_s

    if side == "BUY":
        stopPrice = entryPrice * (1 - stop_ratio / 100)

    else:
        stopPrice = entryPrice * (1 + stop_ratio / 100)

    if symbol == "BTCUSDT":
        stopPrice = round(stopPrice, 1)

    else:
        stopPrice = round(stopPrice, 2)

    return stopPrice


def cal_profit_price(entryPrice, side, symbol, is_small):
    if not is_small:
        profit_ratio = config.tp
    else:
        profit_ratio = config.tp_s

    if side == "BUY":
        stopPrice = entryPrice * (1 + profit_ratio / 100)

    else:
        stopPrice = entryPrice * (1 - profit_ratio / 100)

    if symbol == "BTCUSDT":
        stopPrice = round(stopPrice, 1)
    else:
        stopPrice = round(stopPrice, 2)

    return stopPrice
