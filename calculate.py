import config


# def cal_stop_price(entryPrice, side, symbol):
#     if symbol == "BTCUSDT":
#         stop_ratio = config.sl_btc
#     else:
#         stop_ratio = config.sl_eth
#
#     if side == "BUY":
#         stopPrice = entryPrice * (1 - stop_ratio / 100)
#     else:
#         stopPrice = entryPrice * (1 + stop_ratio / 100)
#
#     if symbol == "BTCUSDT":
#         stopPrice = round(stopPrice, 1)
#     else:
#         stopPrice = round(stopPrice, 2)
#
#     return stopPrice
#
#
# def cal_profit_price(entryPrice, side, symbol):
#     if symbol == "BTCUSDT":
#         profit_ratio = config.tp_btc
#     else:
#         profit_ratio = config.tp_eth
#
#     if side == "BUY":
#         stopPrice = entryPrice * (1 + profit_ratio / 100)
#     else:
#         stopPrice = entryPrice * (1 - profit_ratio / 100)
#
#     if symbol == "BTCUSDT":
#         stopPrice = round(stopPrice, 1)
#     else:
#         stopPrice = round(stopPrice, 2)
#
#     return stopPrice


def cal_stop_loss_atr(entryPrice, ATR, side, symbol):
    if symbol == "BTCUSDT":
        multiply = config.sl_btc_atr
    elif symbol == "ETHUSDT":
        multiply = config.sl_eth_atr
    else:
        multiply = config.sl_sol_atr

    if side == "BUY":
        stopPrice = entryPrice - multiply * ATR
    else:
        stopPrice = entryPrice + multiply * ATR

    if symbol == "BTCUSDT":
        stopPrice = round(stopPrice, 1)
    elif symbol == "ETHUSDT":
        stopPrice = round(stopPrice, 2)
    else:
        stopPrice = round(stopPrice, 3)

    return stopPrice


def cal_take_profit_atr(entryPrice, ATR, side, symbol):
    if symbol == "BTCUSDT":
        multiply = config.tp_btc_atr
    elif symbol == "ETHUSDT":
        multiply = config.tp_eth_atr
    else:
        multiply = config.tp_sol_atr

    if side == "BUY":
        stopPrice = entryPrice + multiply * ATR
    else:
        stopPrice = entryPrice - multiply * ATR

    if symbol == "BTCUSDT":
        stopPrice = round(stopPrice, 1)
    elif symbol == "ETHUSDT":
        stopPrice = round(stopPrice, 2)
    else:
        stopPrice = round(stopPrice, 3)

    return stopPrice
