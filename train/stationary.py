import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose

from fetch import fetch_data
from preprocess import cal_values

# 데이터 가져오기
df: pd.DataFrame = fetch_data(symbol="BTCUSDT", interval="1h", numbers=770)
df = cal_values(df)

# 시계열 데이터 선택
your_time_series = df["volume_delta"]

"""
평균의 일정성 확인
"""
# 시계열 데이터의 평균 확인
rolling_mean = your_time_series.rolling(window=12).mean()

# 원 시계열 데이터와 구간 평균을 그래프로 비교
plt.plot(your_time_series, label="Original Series")
plt.plot(rolling_mean, color="red", label="Rolling Mean (12-period)")
plt.legend(loc="best")
plt.title("Rolling Mean")
plt.show()

"""
분산의 일정성 확인
"""
# 시계열 데이터의 분산 확인
rolling_std = your_time_series.rolling(window=12).std()

# 원 시계열 데이터와 구간 표준편차를 그래프로 비교
plt.plot(your_time_series, label="Original Series")
plt.plot(rolling_std, color="orange", label="Rolling Std (12-period)")
plt.legend(loc="best")
plt.title("Rolling Standard Deviation")
plt.show()

"""
자기상관함수 확인
"""

# ACF 그래프 출력
plot_acf(your_time_series, lags=40)
plt.title("Autocorrelation Function")
plt.show()

# ADF 테스트
result = adfuller(your_time_series)

print("ADF Statistic:", result[0])
print("p-value:", result[1])
print("Critical Values:", result[4])

# p-value가 0.05 미만이면 정상성을 가진다.
if result[1] < 0.05:
    print("시계열 데이터는 정상성을 가집니다.")
else:
    print("시계열 데이터는 비정상적입니다.")
