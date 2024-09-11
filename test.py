import pandas as pd

# 예시 데이터프레임 생성
data = {"A": range(1, 101), "B": range(101, 201)}  # 1부터 100까지 숫자
df = pd.DataFrame(data)

# 데이터프레임 출력 (처음 몇 개의 행만)
print("전체 데이터프레임:")
print(df.head())

# 마지막 24개의 행 선택
last_24_rows = df.iloc[-24:]

print("\n마지막 24개 행:")
print(last_24_rows)
