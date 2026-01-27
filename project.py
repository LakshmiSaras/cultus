# ============================================
# ADVANCED TIME SERIES FORECASTING WITH LSTM
# + MODEL EXPLAINABILITY USING SHAP
# ============================================

# -------- STEP 1: IMPORT LIBRARIES ----------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

import shap
import warnings
warnings.filterwarnings("ignore")

# -------- STEP 2: DATASET GENERATION ----------
np.random.seed(42)

time = np.arange(0, 1000)
trend = time * 0.03
seasonality = 10 * np.sin(2 * np.pi * time / 50)
noise = np.random.normal(0, 2, size=len(time))

value = trend + seasonality + noise

df = pd.DataFrame({
    "time": time,
    "value": value
})

plt.figure()
plt.plot(df["time"], df["value"])
plt.title("Synthetic Time Series (Trend + Seasonality + Noise)")
plt.xlabel("Time")
plt.ylabel("Value")
plt.show()

# -------- STEP 3: SCALING ----------
scaler = MinMaxScaler()
df["scaled_value"] = scaler.fit_transform(df[["value"]])

# -------- STEP 4: SEQUENCE CREATION ----------
def create_sequences(data, window_size=30):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)

WINDOW_SIZE = 30
X, y = create_sequences(df["scaled_value"].values, WINDOW_SIZE)

# Reshape for LSTM [samples, timesteps, features]
X = X.reshape((X.shape[0], X.shape[1], 1))

# -------- STEP 5: TRAIN-TEST SPLIT ----------
split_index = int(0.8 * len(X))

X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# -------- STEP 6: BUILD LSTM MODEL ----------
model = Sequential([
    LSTM(64, activation="tanh", input_shape=(X_train.shape[1], 1)),
    Dense(1)
])

model.compile(
    optimizer="adam",
    loss="mse"
)

model.summary()

# -------- STEP 7: TRAIN MODEL ----------
history = model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

# -------- STEP 8: MODEL EVALUATION ----------
predictions = model.predict(X_test)

rmse = mean_squared_error(y_test, predictions, squared=False)
mae = mean_absolute_error(y_test, predictions)

print("RMSE:", rmse)
print("MAE:", mae)

# -------- STEP 9: PLOT ACTUAL VS PREDICTED ----------
plt.figure()
plt.plot(y_test[:200], label="Actual")
plt.plot(predictions[:200], label="Predicted")
plt.legend()
plt.title("Actual vs Predicted Values")
plt.show()

# -------- STEP 10: MODEL EXPLAINABILITY USING SHAP ----------
# Use a small background set for performance
background = X_train[:100]
test_samples = X_test[:10]

explainer = shap.DeepExplainer(model, background)
shap_values = explainer.shap_values(test_samples)

# SHAP summary plot
shap.summary_plot(shap_values[0], test_samples)

# -------- STEP 11: FINAL SUMMARY ----------
print("\nFINAL MODEL SUMMARY")
print("-------------------")
print("Window Size:", WINDOW_SIZE)
print("Model: LSTM (64 units)")
print("Optimizer: Adam")
print("Loss Function: MSE")
print("Evaluation Metrics: RMSE, MAE")
print("Explainability: SHAP values for time-step importance")
