import logging, asyncio, datetime
from binance.um_futures import UMFutures
from binance.error import ClientError
from functools import partial


async def get_position(key, secret, symbol):
    loop = asyncio.get_running_loop()
    um_futures_client = UMFutures(key=key, secret=secret)
    func = partial(um_futures_client.get_position_risk, symbol=symbol, recvWindow=1000)
    try:
        response = await loop.run_in_executor(None, func)
        return response[0]
    except ClientError as error:
        logging.error(
            f"Found error. status(get_position){symbol}: {error.status_code}, error code: {error.error_code}, error message: {error.error_message}"
        )
    except Exception as error:
        logging.error(f"Unexpected error occurred(get_position){symbol}: {error}")


async def get_balance(key, secret):
    loop = asyncio.get_running_loop()
    um_futures_client = UMFutures(key=key, secret=secret)
    func = partial(um_futures_client.balance, recvWindow=1000)
    try:
        data = await loop.run_in_executor(None, func)
        usdt_data = next((item for item in data if item["asset"] == "USDT"), None)
        if usdt_data:
            balance = float(usdt_data["balance"])
            available_balance = float(usdt_data["availableBalance"])

            return [balance, available_balance]
        else:
            raise Exception("No data found for asset 'USDT'")
    except ClientError as error:
        logging.error(
            f"Found error. status(get_balance): {error.status_code}, error code: {error.error_code}, error message: {error.error_message}"
        )
    except Exception as error:
        logging.error(f"Unexpected error occurred(get_balance): {error}")


async def change_leverage(key, secret, symbol, leverage):
    loop = asyncio.get_running_loop()
    um_futures_client = UMFutures(key=key, secret=secret)
    func = partial(
        um_futures_client.change_leverage,
        symbol=symbol,
        leverage=leverage,
    )
    try:
        await loop.run_in_executor(None, func)

    except ClientError as error:
        logging.error(
            f"Found error. status(change_leverage){symbol}: {error.status_code}, error code: {error.error_code}, error message: {error.error_message}"
        )
    except Exception as error:
        logging.error(f"Unexpected error occurred(change_leverage){symbol}: {error}")


async def open_position(
    key, secret, symbol, side, quantity, price, stopSide, stopPrice, profitPrice
):
    gtd = (
        datetime.datetime.now(datetime.UTC) + datetime.timedelta(minutes=59)
    ).timestamp()
    goodTillDate = int(gtd * 1000)

    loop = asyncio.get_running_loop()
    um_futures_client = UMFutures(key=key, secret=secret)
    func_open = partial(
        um_futures_client.new_order,
        symbol=symbol,
        side=side,
        type="LIMIT",
        price=price,
        quantity=quantity,
        timeInForce="GTD",
        goodTillDate=goodTillDate,
    )
    func_sl = partial(
        um_futures_client.new_order,
        symbol=symbol,
        side=stopSide,
        type="STOP",
        quantity=quantity,
        price=stopPrice,
        stopPrice=stopPrice,
        reduceOnly="true",
    )
    func_tp = partial(
        um_futures_client.new_order,
        symbol=symbol,
        side=stopSide,
        type="TAKE_PROFIT",
        quantity=quantity,
        price=profitPrice,
        stopPrice=profitPrice,
        reduceOnly="true",
    )

    try:
        await loop.run_in_executor(None, func_open)
        await loop.run_in_executor(None, func_sl)
        await loop.run_in_executor(None, func_tp)

    except ClientError as error:
        logging.error(
            f"Found error. status(open_position){symbol}: {error.status_code}, error code: {error.error_code}, error message: {error.error_message}"
        )
    except Exception as error:
        logging.error(f"Unexpected error occurred(open_position){symbol}: {error}")


async def close_position(key, secret, symbol, stopSide, quantity, price):

    loop = asyncio.get_running_loop()
    um_futures_client = UMFutures(key=key, secret=secret)
    func = partial(
        um_futures_client.new_order,
        symbol=symbol,
        side=stopSide,
        type="LIMIT",
        price=price,
        quantity=quantity,
        timeInForce="GTC",
        reduceOnly="true",
    )
    try:
        await loop.run_in_executor(None, func)

    except ClientError as error:
        logging.error(
            f"Found error. close position {symbol}: {error.status_code}, error code: {error.error_code}, error message: {error.error_message}"
        )
    except Exception as error:
        logging.error(f"Unexpected error occurred(tp_sl){symbol}: {error}")


async def cancel_orders(key, secret, symbol):
    loop = asyncio.get_running_loop()
    um_futures_client = UMFutures(key=key, secret=secret)
    func = partial(
        um_futures_client.cancel_open_orders,
        symbol=symbol,
        recvWindow=1000,
    )
    try:
        await loop.run_in_executor(None, func)

    except ClientError as error:
        logging.error(
            f"Found error. status(cancel_orders){symbol}: {error.status_code}, error code: {error.error_code}, error message: {error.error_message}"
        )
    except Exception as error:
        logging.error(f"Unexpected error occurred(cancel_orders){symbol}: {error}")
