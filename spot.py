from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

from fetch import fetch_data
from preprocess import cal_values, make_spot_data


df = fetch_data("BTCUSDT", "1h", 720, "spot")
end_timestamp = df.iloc[-1]["close_time"]
print(end_timestamp)
df = cal_values(df)
X_data, y_data, real_x_data = make_spot_data(df, 12)

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
real_y_prob_lst = model.predict_proba(real_x_data)
prob = max(real_y_prob_lst[0])

# 성능 평가
print("Validation Accuracy:", accuracy_val)
print("Real Predicted Data:", real_y_pred)
print("Real Probability:", prob)
