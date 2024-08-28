import logging, asyncio
import joblib

import config
from preprocess import cal_values, x_data
from fetch import fetch_data_async
from calculate import cal_stop_price
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
    tp_sl,
    cancel_orders,
)
from train import training_and_save_model


async def main(symbol, leverage, interval):
    key = config.key
    secret = config.secret
    ratio = config.ratio
    start = 0
    position_cnt = 0
    sl_ratio = config.stop_ratio
    model_dir = f"models/gb_classifier_{symbol}.pkl"

    logging.info(f"{symbol} {interval} trading program start")

    # 첫 시작 시 모델 훈련 및 저장 / 해당 심볼 레버리지 변경
    if start == 0:
        df = await fetch_data_async(symbol, interval, 1080)
        df = cal_values(df)
        await training_and_save_model(symbol, df, model_dir, sl_ratio)
        await change_leverage(key, secret, symbol, leverage)
        start += 1
        logging.info(f"{symbol} {interval} start process success!")

    while True:
        # 정시까지 기다리기
        await wait_until_next_interval(interval=interval)
        logging.info(f"{symbol} {interval} next interval")

        # 데이터 로드
        df = await fetch_data_async(symbol, interval, 1080)
        df = cal_values(df)
        last_row = df.iloc[-1]

        # 예측 결과 가져오기
        model = joblib.load(model_dir)
        X_data = x_data(last_row)
        pred = model.predict(X_data)
        prob_lst = model.predict_proba(X_data)
        prob = max(prob_lst[0])
        logging.info(f"{symbol} {interval} Prediction: {pred}")
        logging.info(f"{symbol} {interval} Probability: {prob}")

        position = await get_position(key, secret, symbol)
        positionAmt = float(position["positionAmt"])

        # 해당 포지션이 있는 경우, 포지션 종료 로직
        if positionAmt > 0:
            position_cnt += 1

            if pred == 1 and (prob >= 0.99 or position_cnt == 6):
                await tp_sl(key, secret, symbol, "SELL", positionAmt)
                position_cnt = 0
                logging.info(f"{symbol} {interval} long position close")
                await asyncio.sleep(1.5)

            elif pred == 2 and position_cnt == 6:
                position_cnt = 0
                logging.info(f"{symbol} {interval} long position cnt init")

        elif positionAmt < 0:
            position_cnt += 1

            if pred == 2 and (prob >= 0.99 or position_cnt == 6):
                await tp_sl(key, secret, symbol, "BUY", abs(positionAmt))
                position_cnt = 0
                logging.info(f"{symbol} {interval} short position close")
                await asyncio.sleep(1.5)

            elif pred == 1 and position_cnt == 6:
                position_cnt = 0
                logging.info(f"{symbol} {interval} short position cnt init")

        # 포지션 다시 가져오기(종료된 경우 고려)
        position = await get_position(key, secret, symbol)
        positionAmt = float(position["positionAmt"])
        [balance, available] = await get_balance(key, secret)

        # 해당 포지션이 없고 마진이 있는 경우 포지션 진입
        if positionAmt == 0 and (balance * (ratio / 100) < available):

            await cancel_orders(key, secret, symbol)
            logging.info(f"{symbol} open orders cancel")

            # 롱
            if pred == 2:

                entryPrice = last_row["close"]
                raw_quantity = balance * (ratio / 100) / entryPrice * leverage
                quantity = format_quantity(raw_quantity, symbol)
                stopPrice = cal_stop_price(entryPrice, "BUY", symbol)

                await open_position(
                    key,
                    secret,
                    symbol,
                    "BUY",
                    quantity,
                    "SELL",
                    stopPrice,
                )

                # 로그 기록
                logging.info(f"{symbol} {interval} long position open.")

            # 숏
            elif pred == 1:

                entryPrice = last_row["close"]
                raw_quantity = balance * (ratio / 100) / entryPrice * leverage
                quantity = format_quantity(raw_quantity, symbol)
                stopPrice = cal_stop_price(entryPrice, "SELL", symbol)

                await open_position(
                    key,
                    secret,
                    symbol,
                    "SELL",
                    quantity,
                    "BUY",
                    stopPrice,
                )

                # 로그 기록
                logging.info(f"{symbol} {interval} short position open.")

        # 모델 업데이트
        await training_and_save_model(symbol, df, model_dir, sl_ratio)


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
