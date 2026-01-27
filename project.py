# ==========================================================
# 1. IMPORT LIBRARIES
# ==========================================================
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, MultiHeadAttention, LayerNormalization
from tensorflow.keras.models import Model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ==========================================================
# 2. SYNTHETIC MULTIVARIATE DATA GENERATION
# ==========================================================
np.random.seed(42)
t = np.arange(1500)

trend = 0.002 * t
season1 = np.sin(2 * np.pi * t / 50)
season2 = np.sin(2 * np.pi * t / 100)

x1 = season1 + np.random.normal(0, 0.2, len(t))
x2 = season2 + np.random.normal(0, 0.2, len(t))
x3 = np.random.normal(0, 1, len(t))
x4 = trend + np.random.normal(0, 0.3, len(t))
x5 = np.cos(2 * np.pi * t / 30)

y = 2*season1 + 0.5*x2 + 0.3*x4 + x3

data = pd.DataFrame({'y': y, 'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4, 'x5': x5})

# ==========================================================
# 3. SCALING
# ==========================================================
scaler = MinMaxScaler()
scaled = scaler.fit_transform(data)

# ==========================================================
# 4. CREATE SEQUENCES
# ==========================================================
def create_sequences(data, n_steps=30):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i+n_steps])
        y.append(data[i+n_steps, 0])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled, 30)

# ==========================================================
# 5. ROLLING CROSS VALIDATION
# ==========================================================
def rolling_cv(X, y, window=800, step=200):
    splits = []
    for start in range(0, len(X) - window - step, step):
        train = slice(start, start+window)
        test = slice(start+window, start+window+step)
        splits.append((train, test))
    return splits

splits = rolling_cv(X, y)

# ==========================================================
# 6. BASELINE LSTM MODEL
# ==========================================================
def build_lstm(input_shape):
    inp = Input(shape=input_shape)
    x = LSTM(64, return_sequences=False)(inp)
    x = Dropout(0.2)(x)
    out = Dense(1)(x)
    model = Model(inp, out)
    model.compile(optimizer='adam', loss='mse')
    return model

# ==========================================================
# 7. LSTM + SELF ATTENTION MODEL
# ==========================================================
def build_attention_model(input_shape):
    inp = Input(shape=input_shape)
    x = LSTM(64, return_sequences=True)(inp)

    attn_output = MultiHeadAttention(num_heads=2, key_dim=32)(x, x)
    x = LayerNormalization()(x + attn_output)

    x = LSTM(32)(x)
    x = Dropout(0.2)(x)
    out = Dense(1)(x)

    model = Model(inp, out)
    model.compile(optimizer='adam', loss='mse')
    return model

# ==========================================================
# 8. TRAINING + EVALUATION LOOP
# ==========================================================
results = []

for i, (train_idx, test_idx) in enumerate(splits):
    print(f"\nFold {i+1}")

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    lstm_model = build_lstm(X_train.shape[1:])
    attn_model = build_attention_model(X_train.shape[1:])

    lstm_model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)
    attn_model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)

    pred_lstm = lstm_model.predict(X_test)
    pred_attn = attn_model.predict(X_test)

    rmse_lstm = np.sqrt(mean_squared_error(y_test, pred_lstm))
    rmse_attn = np.sqrt(mean_squared_error(y_test, pred_attn))

    mae_lstm = mean_absolute_error(y_test, pred_lstm)
    mae_attn = mean_absolute_error(y_test, pred_attn)

    results.append(["LSTM", rmse_lstm, mae_lstm])
    results.append(["Attention", rmse_attn, mae_attn])

results_df = pd.DataFrame(results, columns=["Model", "RMSE", "MAE"])
print("\nCross-Validation Results")
print(results_df.groupby("Model").mean())

# ==========================================================
# 9. ATTENTION WEIGHT INTERPRETATION
# ==========================================================
attention_layer = attn_model.layers[2]
attention_model = Model(attn_model.input, attention_layer.output)

attention_scores = attention_model.predict(X_test[:1])
print("\nSample Attention Output Shape:", attention_scores.shape)
