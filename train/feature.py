from xgboost import plot_importance
import matplotlib.pyplot as plt
import joblib

symbol = "BTCUSDT"
model_dir = f"models/gb_classifier_{symbol}_v2.pkl"

model = joblib.load(model_dir)
plot_importance(model, importance_type="weight")
plt.show()
