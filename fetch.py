import datetime, asyncio, logging
from datetime import timedelta
from functools import partial
import pandas as pd
from binance.um_futures import UMFutures
from binance.spot import Spot
from binance.error import ClientError


# fetch_data 를 위한 함수, 정해진 개수의 데이터 가져옴
def fetch_one_data(
    symbol: str,
    interval: str,
    end_time: int,
    limit: int,
    type: str = "future",
) -> pd.DataFrame:

    client = UMFutures()
    if type == "spot":
        client = Spot()

    bars = client.klines(
        symbol=symbol, interval=interval, endTime=end_time, limit=limit
    )
    df = pd.DataFrame(
        bars,
        columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base_asset_volume",
            "taker_buy_quote_asset_volume",
            "ignore",
        ],
    )
    df.drop(
        [
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base_asset_volume",
            "taker_buy_quote_asset_volume",
            "ignore",
        ],
        axis=1,
        inplace=True,
    )
    current_time = datetime.datetime.now(datetime.UTC)
    adjusted_time = current_time
    # 5분 봉
    if interval == "5m":
        remainder = current_time.minute % 5
        adjusted_time = current_time - timedelta(minutes=remainder)
    # 15분 봉
    elif interval == "15m":
        remainder = current_time.minute % 15
        adjusted_time = current_time - timedelta(minutes=remainder)
    # 1시간 봉
    elif interval == "1h":
        adjusted_time = current_time.replace(minute=0)
    # 4시간 봉
    elif interval == "4h":
        adjusted_hour = (current_time.hour // 4) * 4
        adjusted_time = current_time.replace(hour=adjusted_hour, minute=0)
    # 1일 봉
    elif interval == "1d":
        adjusted_time = current_time.replace(hour=0, minute=0)

    # 만약 현재 시간 봉 데이터가 존재하면 마지막 행 제거
    open_time = int(adjusted_time.replace(second=0, microsecond=0).timestamp() * 1000)
    if df.iloc[-1]["open_time"] == open_time:
        df.drop(df.index[-1], inplace=True)

    # 모든 열을 숫자로 변환
    for column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    return df


# interval 에 따른 정해진 개수의 데이터 가져오기
def fetch_data(
    symbol: str,
    interval: str,
    numbers: int,
    type: str = "future",
) -> pd.DataFrame:

    current_time = datetime.datetime.now(datetime.UTC)
    adjusted_time = current_time
    # 5분 봉
    if interval == "5m":
        remainder = current_time.minute % 5
        adjusted_time = current_time - timedelta(minutes=remainder)
    # 15분 봉
    elif interval == "15m":
        remainder = current_time.minute % 15
        adjusted_time = current_time - timedelta(minutes=remainder)
    # 1시간 봉
    elif interval == "1h":
        adjusted_time = current_time.replace(minute=0)
    # 4시간 봉
    elif interval == "4h":
        adjusted_hour = (current_time.hour // 4) * 4
        adjusted_time = current_time.replace(hour=adjusted_hour, minute=0)
    # 1일 봉
    elif interval == "1d":
        adjusted_time = current_time.replace(hour=0, minute=0)
    end_time = int(
        adjusted_time.replace(second=0, microsecond=0).timestamp() * 1000 - 1
    )

    data = []

    cnt: int = 1500

    while numbers > 0:
        if numbers < cnt:
            num = numbers
        else:
            num = cnt

        df = fetch_one_data(
            symbol=symbol,
            interval=interval,
            end_time=end_time,
            limit=num,
            type=type,
        )
        if df.empty:
            break

        data.insert(0, df)

        end_time = int(df.iloc[0]["open_time"]) - 1
        numbers -= num

    data_combined = pd.concat(data, axis=0, ignore_index=True)
    data_combined.dropna(axis=0, inplace=True)
    data_combined.reset_index(drop=True, inplace=True)

    return data_combined


async def fetch_data_async(symbol, interval, numbers) -> pd.DataFrame:
    loop = asyncio.get_running_loop()
    client = UMFutures()

    now = datetime.datetime.now(datetime.UTC)
    end_datetime = now.replace(minute=0, second=0, microsecond=0)
    end_time = int(end_datetime.timestamp() * 1000 - 1)

    func = partial(
        client.klines,
        symbol=symbol,
        interval=interval,
        end_time=end_time,
        limit=numbers,
    )
    try:
        bars = await loop.run_in_executor(None, func)
        df = pd.DataFrame(
            bars,
            columns=[
                "open_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time",
                "quote_asset_volume",
                "number_of_trades",
                "taker_buy_base_asset_volume",
                "taker_buy_quote_asset_volume",
                "ignore",
            ],
        )
        df.drop(
            [
                "quote_asset_volume",
                "number_of_trades",
                "taker_buy_base_asset_volume",
                "taker_buy_quote_asset_volume",
                "ignore",
            ],
            axis=1,
            inplace=True,
        )

        # 만약 현재 시간 봉 데이터가 존재하면 마지막 행 제거
        open_time = int(
            now.replace(minute=0, second=0, microsecond=0).timestamp() * 1000
        )
        if df.iloc[-1]["open_time"] == open_time:
            df.drop(df.index[-1], inplace=True)

        # 모든 열을 숫자형으로 변환
        for column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

        return df

    except ClientError as error:
        logging.error(
            f"Found error. status(fetch_data){symbol}: {error.status_code}, error code: {error.error_code}, error message: {error.error_message}"
        )
    except Exception as error:
        logging.error(f"Unexpected error occurred(fetch_data){symbol}: {error}")
