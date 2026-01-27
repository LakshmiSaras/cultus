# ================================
# 1. IMPORT LIBRARIES
# ================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

import shap

# ================================
# 2. DATASET CREATION (MULTIVARIATE)
# ================================
np.random.seed(42)
t = np.arange(0, 2000)

trend = 0.003 * t
season1 = 2 * np.sin(2 * np.pi * t / 50)
season2 = 1.5 * np.sin(2 * np.pi * t / 200)

x1 = season1 + np.random.normal(0, 0.3, len(t))
x2 = trend + np.random.normal(0, 0.5, len(t))
x3 = np.random.normal(0, 1 + t*0.001)

y = trend + season1 + season2 + 0.5*x1 + 0.3*x2 + x3

data = pd.DataFrame({"y": y, "x1": x1, "x2": x2, "x3": x3})

# ================================
# 3. TIME-BASED FEATURES
# ================================
data['time'] = np.arange(len(data))
data['sin_time'] = np.sin(2 * np.pi * data['time'] / 50)
data['cos_time'] = np.cos(2 * np.pi * data['time'] / 50)

# ================================
# 4. SCALING
# ================================
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# ================================
# 5. CREATE LAG SEQUENCES
# ================================
def create_sequences(data, n_steps_in=30, n_steps_out=7):
    X, y = [], []
    for i in range(len(data) - n_steps_in - n_steps_out):
        X.append(data[i:i+n_steps_in])
        y.append(data[i+n_steps_in:i+n_steps_in+n_steps_out, 0])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data)

# ================================
# 6. TIME SERIES CROSS-VALIDATION
# ================================
tscv = TimeSeriesSplit(n_splits=5)

rmse_scores = []
mae_scores = []

for fold, (train_index, test_index) in enumerate(tscv.split(X)):
    print(f"\nTraining Fold {fold+1}")

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # ================================
    # 7. LSTM MODEL
    # ================================
    model = Sequential([
        LSTM(64, activation='tanh', return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
        Dropout(0.2),
        LSTM(32),
        Dense(7)
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

    # ================================
    # 8. PREDICTION & EVALUATION
    # ================================
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test.flatten(), y_pred.flatten()))
    mae = mean_absolute_error(y_test.flatten(), y_pred.flatten())

    rmse_scores.append(rmse)
    mae_scores.append(mae)

    print(f"Fold {fold+1} RMSE: {rmse:.4f}, MAE: {mae:.4f}")

print("\nAverage RMSE:", np.mean(rmse_scores))
print("Average MAE:", np.mean(mae_scores))

# ================================
# 9. EXPLAINABILITY (SHAP)
# ================================
print("\nRunning SHAP Explainability...")

explainer = shap.DeepExplainer(model, X_train[:100])
shap_values = explainer.shap_values(X_test[:10])

# Plot SHAP summary
shap.summary_plot(shap_values[0], X_test[:10], feature_names=data.columns)

# ================================
# 10. GRADIENT-BASED IMPORTANCE
# ================================
print("\nComputing Gradient-Based Feature Importance...")

sample = tf.convert_to_tensor(X_test[:1])
with tf.GradientTape() as tape:
    tape.watch(sample)
    prediction = model(sample)

grads = tape.gradient(prediction, sample)
importance = tf.reduce_mean(tf.abs(grads), axis=0).numpy()

print("Feature Importance Shape:", importance.shape)
