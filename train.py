import logging

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import joblib

from preprocess import make_data


async def training_and_save_model(
    symbol: str, df: pd.DataFrame, model_dir: str, sl_ratio: float
):

    X_data, y_data, _ = make_data(df, 12, sl_ratio)
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

    # 정확도 70% 이상인 경우 모델 저장
    if accuracy_val >= 0.70:
        joblib.dump(model, model_dir)
        logging.info(f"{symbol} accuracy: {accuracy_val}\nModel save successfully.")
    else:
        logging.info(f"{symbol} accuracy: {accuracy_val}\nAccuracy Score is low.")
