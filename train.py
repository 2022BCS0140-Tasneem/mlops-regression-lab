import json
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

X, y = load_diabetes(return_X_y=True)

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)

mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

joblib.dump(model, "model.pkl")

with open("metrics.json", "w") as f:
    json.dump({"MSE": mse, "R2": r2}, f)

print("MSE:", mse)
print("R2:", r2)
