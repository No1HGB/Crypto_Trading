import logging, asyncio
import joblib
import numpy as np

import config
from preprocess import cal_values, x_data
from fetch import fetch_data_async
from calculate import cal_stop_loss_atr, cal_take_profit_atr
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
from logic import trend_long, trend_short


async def main(symbol, leverage, interval):
    key = config.key
    secret = config.secret
    ratio = config.ratio
    data_num = 337
    start = 0
    prob_baseline = 0.6
    model_dir = f"train/models/gb_classifier_BTCUSDT.pkl"

    # 첫 시작 시 해당 심볼 레버리지 변경
    if start == 0:
        await change_leverage(key, secret, symbol, leverage)
        start += 1
        logging.info(f"{symbol} {interval} trading program start")

    while True:
        # 정시까지 기다리기
        await wait_until_next_interval(interval=interval)
        logging.info(f"{symbol} {interval} next interval")

        # BTC 데이터
        df_btc = await fetch_data_async("BTCUSDT", interval, data_num)
        df_btc = cal_values(df_btc)

        # 데이터 로드
        df = await fetch_data_async(symbol, interval, data_num)
        df = cal_values(df)
        last_row = df.iloc[-1]

        # 메인 예측 결과 가져오기
        model = joblib.load(model_dir)
        X_data = x_data(df_btc, symbol)
        pred = model.predict(X_data)
        prob = np.max(model.predict_proba(X_data), axis=1)

        # 추세 장
        t_long = trend_long(df)
        t_short = trend_short(df)

        # 포지션 가져오기
        position = await get_position(key, secret, symbol)
        positionAmt = float(position["positionAmt"])

        # 롱 포지션 종료
        if positionAmt > 0:

            if last_row["ha_close"] < last_row["ha_open"]:
                await cancel_orders(key, secret, symbol)
                logging.info(f"{symbol} open orders cancel for close")
                quantity = abs(positionAmt)

                await close_position(key, secret, symbol, "SELL", quantity)
                await asyncio.sleep(1.5)
                # 로그 기록
                logging.info(f"{symbol} {interval} long position close")

        # 숏 포지션 종료
        elif positionAmt < 0:

            if last_row["ha_close"] > last_row["ha_open"]:
                await cancel_orders(key, secret, symbol)
                logging.info(f"{symbol} open orders cancel for close")
                quantity = abs(positionAmt)

                await close_position(key, secret, symbol, "BUY", quantity)
                await asyncio.sleep(1.5)
                # 로그 기록
                logging.info(f"{symbol} {interval} short position close")

        # 포지션 다시 가져오기(종료된 경우 고려)
        position = await get_position(key, secret, symbol)
        positionAmt = float(position["positionAmt"])
        [balance, available] = await get_balance(key, secret)

        # 해당 포지션이 없고 마진이 있는 경우 포지션 진입
        if positionAmt == 0 and (balance * (ratio / 100) < available):
            await cancel_orders(key, secret, symbol)
            logging.info(f"{symbol} open orders cancel")

            # 롱
            if pred == 2 and prob >= prob_baseline and not t_short:
                entryPrice = last_row["close"]
                ATR = last_row["ATR"]
                raw_quantity = balance * (ratio / 100) / entryPrice * leverage
                quantity = format_quantity(raw_quantity, symbol)
                stopPrice = cal_stop_loss_atr(entryPrice, ATR, "BUY", symbol)
                profitPrice = cal_take_profit_atr(entryPrice, ATR, "BUY", symbol)

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
                # 로그 기록
                logging.info(f"{symbol} {interval} long position open.")

            # 숏
            elif pred == 1 and prob >= prob_baseline and not t_long:
                entryPrice = last_row["close"]
                ATR = last_row["ATR"]
                raw_quantity = balance * (ratio / 100) / entryPrice * leverage
                quantity = format_quantity(raw_quantity, symbol)
                stopPrice = cal_stop_loss_atr(entryPrice, ATR, "SELL", symbol)
                profitPrice = cal_take_profit_atr(entryPrice, ATR, "SELL", symbol)

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
