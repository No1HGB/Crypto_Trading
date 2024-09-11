import logging, asyncio
import joblib

import config
from preprocess import cal_values, x_data
from fetch import fetch_data_async
from calculate import cal_stop_price, cal_profit_price
from util import (
    setup_logging,
    wait_until_next_interval,
    format_quantity,
    server_connect,
)
from account import (
    get_position,
    get_balance,
    change_leverage,
    open_position,
    close_position,
    cancel_orders,
)


async def main(symbol, leverage, interval):
    key = config.key
    secret = config.secret
    ratio = config.ratio
    data_num = 337
    start = 0
    position_cnt = 0
    model_dir = f"train/models/gb_classifier_{symbol}.pkl"

    # 첫 시작 시 해당 심볼 레버리지 변경
    if start == 0:
        await change_leverage(key, secret, symbol, leverage)
        start += 1
        logging.info(f"{symbol} {interval} trading program start")

    while True:
        # 정시까지 기다리기
        await wait_until_next_interval(interval=interval)
        logging.info(f"{symbol} {interval} next interval")

        # 데이터 로드
        df = await fetch_data_async(symbol, interval, data_num)
        df = cal_values(df)
        last_row = df.iloc[-1]

        # 메인 예측 결과 가져오기
        model = joblib.load(model_dir)
        X_data = x_data(df, symbol)
        pred = model.predict(X_data)

        # 포지션 가져오기
        position = await get_position(key, secret, symbol)
        positionAmt = float(position["positionAmt"])
        [balance, available] = await get_balance(key, secret)

        # 롱 포지션 종료
        if positionAmt > 0:
            position_cnt += 1

            if position_cnt == 6:
                await cancel_orders(key, secret, symbol)
                logging.info(f"{symbol} open orders cancel for close")
                quantity = abs(positionAmt)
                price = last_row["close"]

                await close_position(key, secret, symbol, "SELL", quantity, price)
                position_cnt = 0
                # 로그 기록
                logging.info(f"{symbol} {interval} long position close")

        # 숏 포지션 종료
        elif positionAmt < 0:
            position_cnt += 1

            if position_cnt == 6:
                await cancel_orders(key, secret, symbol)
                logging.info(f"{symbol} open orders cancel for close")
                quantity = abs(positionAmt)
                price = last_row["close"]

                await close_position(key, secret, symbol, "BUY", quantity, price)
                position_cnt = 0
                # 로그 기록
                logging.info(f"{symbol} {interval} short position close")

        # 해당 포지션이 없고 마진이 있는 경우 포지션 진입
        elif positionAmt == 0 and (balance * (ratio / 100) < available):
            await cancel_orders(key, secret, symbol)
            logging.info(f"{symbol} open orders cancel")
            position_cnt = 0

            # 롱
            if pred == 1:
                entryPrice = last_row["close"]
                raw_quantity = balance * (ratio / 100) / entryPrice * leverage
                quantity = format_quantity(raw_quantity, symbol)
                stopPrice = cal_stop_price(entryPrice, "BUY", symbol)
                profitPrice = cal_profit_price(entryPrice, "BUY", symbol)

                await open_position(
                    key,
                    secret,
                    symbol,
                    "BUY",
                    quantity,
                    entryPrice,
                    "SELL",
                    stopPrice,
                    profitPrice,
                )
                position_cnt = 1
                # 로그 기록
                logging.info(f"{symbol} {interval} long position open.")

            # 숏
            elif pred == 0:
                entryPrice = last_row["close"]
                raw_quantity = balance * (ratio / 100) / entryPrice * leverage
                quantity = format_quantity(raw_quantity, symbol)
                stopPrice = cal_stop_price(entryPrice, "SELL", symbol)
                profitPrice = cal_profit_price(entryPrice, "SELL", symbol)

                await open_position(
                    key,
                    secret,
                    symbol,
                    "SELL",
                    quantity,
                    entryPrice,
                    "BUY",
                    stopPrice,
                    profitPrice,
                )
                position_cnt = 1
                # 로그 기록
                logging.info(f"{symbol} {interval} short position open.")


symbols = config.symbols
leverage = config.leverage
interval = config.interval


async def run_multiple_tasks():
    # 여러 매개변수로 main 함수를 비동기적으로 실행
    await asyncio.gather(
        main(symbols[0], leverage, interval),
        main(symbols[1], leverage, interval),
    )


if server_connect():
    setup_logging()
    asyncio.run(run_multiple_tasks())
else:
    print("server connect problem")
