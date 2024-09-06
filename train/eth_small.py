from sklearn.ensemble import GradientBoostingClassifier
import joblib

from preprocess import cal_values, make_data_small
from fetch import fetch_data

# 변수 설정
symbol = "ETHUSDT"
interval = "15m"
data_num = 7000
model_dir = f"models/gb_classifier_{symbol}_small.pkl"

# 실행
df = fetch_data(symbol, interval, data_num)
end_timestamp = df.iloc[-1]["close_time"]
df = cal_values(df)
X_data, y_data = make_data_small(df, symbol)

split = int(len(X_data) / 2)
X_train, X_test = X_data[:split], X_data[split:]
y_train, y_test = y_data[:split], y_data[split:]

# 모델 생성
model = GradientBoostingClassifier(
    max_depth=5,  # 트리 깊이 1-5
    n_estimators=1200,  # 시행 횟수 100-1200
    learning_rate=0.05,  # 학습률 0.01-0.1
    min_samples_split=2,  # 노드 분할 최소 샘플 수 2-10
    min_samples_leaf=1,  # 리프 노드 최소 샘플 수 1-5
    random_state=42,
)

# 모델 학습
model.fit(X_train, y_train)

# 모델 저장
joblib.dump(model, model_dir)

print(f"{symbol}_small Model save success!")
