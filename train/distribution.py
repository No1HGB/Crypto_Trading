import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm, normaltest

from fetch import fetch_data
from preprocess import cal_values

# 데이터 가져오기
df: pd.DataFrame = fetch_data(symbol="BTCUSDT", interval="1h", numbers=770)
df = cal_values(df)

# 시계열 데이터 선택
data = df["delta"]

# 1. 분포 그리기 (히스토그램과 커널 밀도)
plt.figure(figsize=(10, 6))
sns.histplot(data, kde=True, stat="density", linewidth=0)
plt.title("Histogram and KDE of Column")
plt.xlabel("Values")
plt.ylabel("Density")

# 2. 평균과 표준편차 계산
mean = data.mean()
std = data.std()
print(f"Mean: {mean}, Standard Deviation: {std}")

# 3. 정규분포 곡선 추가
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mean, std)
plt.plot(x, p, "k", linewidth=2, label="Normal Distribution")
plt.legend()

plt.show()

# 4. 정규성 검정 (normaltest 사용)
stat, p_value = normaltest(data)
print(f"Normality test p-value: {p_value}")
if p_value > 0.05:
    print("정규성을 따를 가능성이 있습니다 (p-value > 0.05).")
else:
    print("정규성을 따르지 않을 가능성이 있습니다 (p-value <= 0.05).")

print(mean + 5 * std, mean - 2 * std)
