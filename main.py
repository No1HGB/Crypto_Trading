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
    cancel_orders,
)


async def main(symbol, leverage, interval):
    key = config.key
    secret = config.secret
    ratio = config.ratio
    data_num = 127
    start = 0
    model_dir = f"train/models/gb_classifier_{symbol}.pkl"
    model_small_dir = f"train/models/gb_classifier_{symbol}_small.pkl"

    # 확률 기준
    if symbol == "BTCUSDT":
        prob_line = 0.9
        prob_line_small = 0.7
    else:
        prob_line = 0.7
        prob_line_small = 0.5

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
        X_data = x_data(df, symbol, False)
        pred = model.predict(X_data)
        prob_lst = model.predict_proba(X_data)
        prob = max(prob_lst[0])

        # 작은 예측 결과 가져오기
        if symbol == "BTCUSDT":
            model_small = joblib.load(model_small_dir)
            X_data_small = x_data(df, symbol, True)
            pred_small = model_small.predict(X_data_small)
            prob_small_lst = model_small.predict_proba(X_data_small)
            prob_small = max(prob_small_lst[0])

        # 포지션 가져오기
        position = await get_position(key, secret, symbol)
        positionAmt = float(position["positionAmt"])
        unRealizedProfit = float(position["unRealizedProfit"])
        [balance, available] = await get_balance(key, secret)

        # 롱 포지션 스위칭
        if positionAmt > 0:
            # 메인 롱 스위칭
            if (
                last_row["volume_delta"] >= 1
                and pred == 0
                and prob >= prob_line
                and unRealizedProfit > 0
                and (balance * (ratio / 100) * 2 < available)
            ):
                await cancel_orders(key, secret, symbol)
                logging.info(f"{symbol} open orders cancel for switching.")

                entryPrice = last_row["close"]
                raw_quantity = 2 * balance * (ratio / 100) / entryPrice * leverage
                quantity = format_quantity(raw_quantity, symbol)
                stopPrice = cal_stop_price(entryPrice, "SELL", symbol, False)
                profitPrice = cal_profit_price(entryPrice, "SELL", symbol, False)

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
                logging.info(f"{symbol} {interval} long position switching.")

            # 작은 롱 스위칭
            elif symbol == "BTCUSDT":
                if (
                    last_row["volume_delta"] < 1
                    and pred_small == 0
                    and prob_small >= prob_line_small
                    and unRealizedProfit > 0
                    and (balance * (ratio / 100) * 2 < available)
                ):
                    await cancel_orders(key, secret, symbol)
                    logging.info(f"{symbol} open orders cancel for small switching.")

                    entryPrice = last_row["close"]
                    raw_quantity = 2 * balance * (ratio / 100) / entryPrice * leverage
                    quantity = format_quantity(raw_quantity, symbol)
                    stopPrice = cal_stop_price(entryPrice, "SELL", symbol, True)
                    profitPrice = cal_profit_price(entryPrice, "SELL", symbol, True)

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
                    logging.info(f"{symbol} {interval} long position small switching.")

        # 숏 포지션 스위칭
        elif positionAmt < 0:
            # 메인 숏 스위칭
            if (
                last_row["volume_delta"] >= 1
                and pred == 1
                and prob >= prob_line
                and unRealizedProfit > 0
                and (balance * (ratio / 100) * 2 < available)
            ):
                await cancel_orders(key, secret, symbol)
                logging.info(f"{symbol} open orders cancel for switching.")

                entryPrice = last_row["close"]
                raw_quantity = 2 * balance * (ratio / 100) / entryPrice * leverage
                quantity = format_quantity(raw_quantity, symbol)
                stopPrice = cal_stop_price(entryPrice, "BUY", symbol, False)
                profitPrice = cal_profit_price(entryPrice, "BUY", symbol, False)

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
                logging.info(f"{symbol} {interval} short position switching.")

            # 작은 숏 스위칭
            elif symbol == "BTCUSDT":
                if (
                    last_row["volume_delta"] < 1
                    and pred_small == 1
                    and prob_small >= prob_line_small
                    and unRealizedProfit > 0
                    and (balance * (ratio / 100) * 2 < available)
                ):
                    await cancel_orders(key, secret, symbol)
                    logging.info(f"{symbol} open orders cancel for small switching.")

                    entryPrice = last_row["close"]
                    raw_quantity = 2 * balance * (ratio / 100) / entryPrice * leverage
                    quantity = format_quantity(raw_quantity, symbol)
                    stopPrice = cal_stop_price(entryPrice, "BUY", symbol, True)
                    profitPrice = cal_profit_price(entryPrice, "BUY", symbol, True)

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
                    logging.info(f"{symbol} {interval} short position small switching.")

        # 해당 포지션이 없고 마진이 있는 경우 포지션 진입
        elif positionAmt == 0 and (balance * (ratio / 100) < available):
            await cancel_orders(key, secret, symbol)
            logging.info(f"{symbol} open orders cancel")

            if last_row["volume_delta"] >= 1:
                # 메인 롱
                if pred == 1 and prob >= prob_line:
                    entryPrice = last_row["close"]
                    raw_quantity = balance * (ratio / 100) / entryPrice * leverage
                    quantity = format_quantity(raw_quantity, symbol)
                    stopPrice = cal_stop_price(entryPrice, "BUY", symbol, False)
                    profitPrice = cal_profit_price(entryPrice, "BUY", symbol, False)

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

                # 메인 숏
                elif pred == 0 and prob >= prob_line:
                    entryPrice = last_row["close"]
                    raw_quantity = balance * (ratio / 100) / entryPrice * leverage
                    quantity = format_quantity(raw_quantity, symbol)
                    stopPrice = cal_stop_price(entryPrice, "SELL", symbol, False)
                    profitPrice = cal_profit_price(entryPrice, "SELL", symbol, False)

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

            # BTCUSDT 일 때 작은 거래량 트레이드 적용
            elif symbol == "BTCUSDT" and last_row["volume_delta"] < 1:
                # 작은 롱
                if pred_small == 1 and prob_small >= prob_line_small:
                    entryPrice = last_row["close"]
                    raw_quantity = balance * (ratio / 100) / entryPrice * leverage
                    quantity = format_quantity(raw_quantity, symbol)
                    stopPrice = cal_stop_price(entryPrice, "BUY", symbol, True)
                    profitPrice = cal_profit_price(entryPrice, "BUY", symbol, True)
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
                    logging.info(f"{symbol} {interval} small long position open.")

                # 작은 숏
                elif pred_small == 0 and prob_small >= prob_line_small:
                    entryPrice = last_row["close"]
                    raw_quantity = balance * (ratio / 100) / entryPrice * leverage
                    quantity = format_quantity(raw_quantity, symbol)
                    stopPrice = cal_stop_price(entryPrice, "SELL", symbol, True)
                    profitPrice = cal_profit_price(entryPrice, "SELL", symbol, True)

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
                    logging.info(f"{symbol} {interval} small short position open.")


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
