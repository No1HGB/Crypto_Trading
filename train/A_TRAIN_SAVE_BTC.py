from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import joblib

from preprocess import cal_values, make_data
from fetch import fetch_data

# 변수 설정
symbol = "BTCUSDT"
interval = "1h"
model_dir = f"models/gb_classifier_{symbol}_update.pkl"

# 조정 변수
data_num = 20700
split_ratio = 0.99
is_save = False

# 데이터 로드
df = fetch_data(symbol, interval, data_num)
end_timestamp = df.iloc[-1]["close_time"]
df = cal_values(df)
X_data, y_data = make_data(df, symbol)

# 테스트 데이터 분할
n = len(X_data)
split = int(n * split_ratio)
X_train = X_data[:split]
y_train = y_data[:split]
X_test = X_data[split:]
y_test = y_data[split:]
print("Shape")
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# 모델 생성
model = XGBClassifier(
    max_depth=5,  # 트리 깊이 1-5
    n_estimators=1200,  # 시행 횟수 100-1200
    learning_rate=0.05,  # 학습률 0.01-0.1
    random_state=42,
)

# 모델 학습
model.fit(X_train, y_train)

# validation 예측
y_pred_test = model.predict(X_test)
y_prob_test = model.predict_proba(X_test)
accuracy_test = accuracy_score(y_test, y_pred_test)
print(accuracy_test, len(y_test))

# 모델 저장
if is_save:
    joblib.dump(model, model_dir)
    print(f"{symbol} Model save success!")
