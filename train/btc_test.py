from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

from preprocess import cal_values, make_data
from fetch import fetch_data

# 변수 설정
symbol = "BTCUSDT"
interval = "1h"
data_num = 24000

# 실행
df = fetch_data(symbol, interval, data_num)
end_timestamp = df.iloc[-1]["close_time"]
df = cal_values(df)
X_data, y_data = make_data(df, symbol)

split = int(len(X_data) / 2)
X_train, X_test = X_data[:split], X_data[split:]
y_train, y_test = y_data[:split], y_data[split:]
print("Shape")
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

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

# validation 예측
y_pred_test = model.predict(X_test)
y_prob_test = model.predict_proba(X_test)
accuracy_test = accuracy_score(y_test, y_pred_test)
print(accuracy_test, len(y_test))

# process_y_test = []
# process_y_pred_test = []
# for i, prob_box in enumerate(y_prob_test):
#     prob = max(prob_box)
#     if prob >= 0.5:
#         process_y_test.append(y_test[i])
#         process_y_pred_test.append(y_pred_test[i])
#
# process_accuracy = accuracy_score(
#     np.array(process_y_test), np.array(process_y_pred_test)
# )
# print(process_accuracy, len(process_y_test))
#
# cal_result = (len(process_y_test) / 2) * (2 * process_accuracy - 1) * (1 - 0.1)
# print(cal_result)
