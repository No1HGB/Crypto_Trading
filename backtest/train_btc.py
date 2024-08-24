from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import joblib

from calculate import cal_values, make_data
from fetch import fetch_data

# 변수 설정
symbol = "BTCUSDT"
interval = "1h"
data_num = 1200
window = 12
sl = 1.5
model_dir = f"model/gb_classifier_btc.pkl"

# 실행
df = fetch_data(symbol, interval, data_num)
end_timestamp = df.iloc[-1]["close_time"]
print(end_timestamp)
df = cal_values(df)
print(df.shape)
using_df = df.iloc[:-120]
X_data, y_data, real_x_data = make_data(df, 12, 1.5)

X_train_split, X_val, y_train_split, y_val = train_test_split(
    X_data, y_data, test_size=0.2, random_state=42
)

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
model.fit(X_train_split, y_train_split)

# validation 예측
y_pred_val = model.predict(X_val)
accuracy_val = accuracy_score(y_val, y_pred_val)

# 실제 예측
real_y_pred = model.predict(real_x_data)

# 모델 저장
joblib.dump(model, model_dir)

# 성능 평가
print("Validation Data:", y_val)
print("Prediction Data:", y_pred_val)
print("Validation Accuracy:", accuracy_val)
print("Real Predicted Data:", real_y_pred)
