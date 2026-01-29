# ============================================================
# ADVANCED MULTIVARIATE TIME SERIES FORECASTING USING LSTM
# Synthetic Financial Market Simulation + Baseline Comparison
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# 1. SYNTHETIC FINANCIAL DATA GENERATION
# ============================================================

def generate_synthetic_financial_data(n_steps=2500, n_assets=5, seed=42):
    np.random.seed(seed)

    # Create correlation matrix
    corr_matrix = np.full((n_assets, n_assets), 0.6)
    np.fill_diagonal(corr_matrix, 1.0)
    chol = np.linalg.cholesky(corr_matrix)

    # GARCH-like volatility
    vol = np.zeros(n_steps)
    vol[0] = 0.01
    alpha, beta = 0.1, 0.85

    shocks = np.random.normal(size=(n_steps, n_assets))
    correlated_shocks = shocks @ chol.T

    prices = np.zeros((n_steps, n_assets))
    prices[0] = 100

    for t in range(1, n_steps):
        vol[t] = np.sqrt(0.00001 + alpha * (vol[t-1]**2) + beta * vol[t-1]**2)
        drift = 0.0005
        returns = drift + vol[t] * correlated_shocks[t]
        prices[t] = prices[t-1] * (1 + returns)

    df = pd.DataFrame(prices, columns=[f"Asset_{i}" for i in range(n_assets)])
    return df

# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================

def add_technical_features(df):
    df_feat = df.copy()
    for col in df.columns:
        df_feat[f"{col}_return"] = df[col].pct_change()
        df_feat[f"{col}_ma_10"] = df[col].rolling(10).mean()
        df_feat[f"{col}_ma_30"] = df[col].rolling(30).mean()
        df_feat[f"{col}_volatility"] = df[col].rolling(20).std()
    df_feat.dropna(inplace=True)
    return df_feat

# ============================================================
# 3. SEQUENCE CREATION FOR LSTM
# ============================================================

def create_sequences(data, lookback=30):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i])
        y.append(data[i, 0])  # Predict Asset_0
    return np.array(X), np.array(y)

# ============================================================
# 4. LSTM MODEL DEFINITION
# ============================================================

def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(64),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# ============================================================
# 5. EVALUATION METRICS
# ============================================================

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# ============================================================
# 6. MAIN PIPELINE
# ============================================================

# Generate Data
df_prices = generate_synthetic_financial_data()
df_features = add_technical_features(df_prices)

# Scale Features
scaler = RobustScaler()
scaled_data = scaler.fit_transform(df_features)

# Create Sequences
LOOKBACK = 30
X, y = create_sequences(scaled_data, LOOKBACK)

# Train/Test Split
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build & Train LSTM
model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
early_stop = EarlyStopping(patience=10, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

lstm_preds = model.predict(X_test).flatten()

# ============================================================
# 7. BASELINE MODEL — SARIMAX
# ============================================================

asset_series = df_features["Asset_0"].values
train_series = asset_series[:train_size + LOOKBACK]
test_series = asset_series[train_size + LOOKBACK:]

sarimax_model = SARIMAX(train_series, order=(1,1,1), seasonal_order=(0,0,0,0))
sarimax_result = sarimax_model.fit(disp=False)
sarimax_preds = sarimax_result.forecast(len(test_series))

# ============================================================
# 8. PERFORMANCE EVALUATION
# ============================================================

lstm_mae = mean_absolute_error(y_test, lstm_preds)
lstm_rmse = np.sqrt(mean_squared_error(y_test, lstm_preds))
lstm_mape = mape(y_test, lstm_preds)

sarimax_mae = mean_absolute_error(test_series, sarimax_preds)
sarimax_rmse = np.sqrt(mean_squared_error(test_series, sarimax_preds))
sarimax_mape = mape(test_series, sarimax_preds)

print("\n===== LSTM Performance =====")
print(f"MAE:  {lstm_mae:.4f}")
print(f"RMSE: {lstm_rmse:.4f}")
print(f"MAPE: {lstm_mape:.2f}%")

print("\n===== SARIMAX Performance =====")
print(f"MAE:  {sarimax_mae:.4f}")
print(f"RMSE: {sarimax_rmse:.4f}")
print(f"MAPE: {sarimax_mape:.2f}%")

# ============================================================
# 9. VISUALIZATION
# ============================================================

plt.figure(figsize=(12,6))
plt.plot(y_test, label="Actual", linewidth=2)
plt.plot(lstm_preds, label="LSTM Forecast", linestyle='--')
plt.title("LSTM Forecast vs Actual")
plt.xlabel("Time")
plt.ylabel("Scaled Price")
plt.legend()
plt.grid()
plt.show()
