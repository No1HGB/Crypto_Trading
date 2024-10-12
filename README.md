# **BTC auto-trading using ML**

### backtest : 전략 백테스팅

### train: 머신러닝 모델 훈련

#### 데이터의 정상성(stationary)를 위해 OHLCV 데이터를 Close 가격 또는 이평선으로 나눈 값 사용
#### 예) up_delta = close / open , volume_delta = volume / volume 50 MA
#### XGBoost Classifier 사용
#### main: Binance API를 이용해 매 시간 데이터를 가져오고 매수/매도 주문, 손절/익절 주문, 청산 주문 실행