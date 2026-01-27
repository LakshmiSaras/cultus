# =========================================
# 1. IMPORT LIBRARIES
# =========================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.api import VAR
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller

from sklearn.metrics import mean_absolute_error, mean_squared_error

# =========================================
# 2. SYNTHETIC MULTIVARIATE DATA GENERATION
# =========================================
np.random.seed(42)
t = np.arange(0, 500)

trend = 0.05 * t
season = 10 * np.sin(2 * np.pi * t / 50)

noise1 = np.random.normal(0, 3, len(t))
noise2 = np.random.normal(0, 2, len(t))

series1 = trend + season + noise1
series2 = 0.5 * series1 + np.random.normal(0, 2, len(t))
series3 = np.cos(2 * np.pi * t / 30) * 5 + noise2

data = pd.DataFrame({
    "y": series1,
    "x1": series2,
    "x2": series3
})

# =========================================
# 3. STATIONARITY CHECK (ADF TEST)
# =========================================
def adf_test(series, name):
    result = adfuller(series)
    print(f"ADF Test for {name}: p-value = {result[1]:.4f}")

for col in data.columns:
    adf_test(data[col], col)

# Differencing if non-stationary
data_diff = data.diff().dropna()

# =========================================
# 4. TRAIN-TEST SPLIT
# =========================================
train_size = int(len(data_diff) * 0.8)
train, test = data_diff[:train_size], data_diff[train_size:]

# =========================================
# 5. BASELINE MODEL (HOLT-WINTERS)
# =========================================
hw_model = ExponentialSmoothing(train['y'], trend='add', seasonal='add', seasonal_periods=50).fit()
hw_forecast = hw_model.forecast(len(test))

# =========================================
# 6. SARIMAX MODEL (STATE SPACE)
# =========================================
sarimax_model = SARIMAX(train['y'],
                        exog=train[['x1', 'x2']],
                        order=(1,1,1),
                        seasonal_order=(1,1,1,50)).fit(disp=False)

sarimax_forecast = sarimax_model.forecast(steps=len(test), exog=test[['x1','x2']])

# =========================================
# 7. VAR MODEL (MULTIVARIATE)
# =========================================
var_model = VAR(train)
var_results = var_model.fit(maxlags=5, ic='aic')

var_forecast = var_results.forecast(train.values[-var_results.k_ar:], steps=len(test))
var_forecast = pd.DataFrame(var_forecast, columns=train.columns)

# =========================================
# 8. EVALUATION FUNCTION
# =========================================
def evaluate(true, pred, name):
    rmse = np.sqrt(mean_squared_error(true, pred))
    mae = mean_absolute_error(true, pred)
    print(f"{name} -> RMSE: {rmse:.3f}, MAE: {mae:.3f}")

print("\nMODEL PERFORMANCE (on differenced data):")
evaluate(test['y'], hw_forecast, "Holt-Winters")
evaluate(test['y'], sarimax_forecast, "SARIMAX")
evaluate(test['y'], var_forecast['y'], "VAR")

# =========================================
# 9. ROLLING WINDOW CROSS-VALIDATION
# =========================================
print("\nRolling Window Backtesting (SARIMAX)")

window_size = 300
step = 20
rmse_scores = []

for start in range(0, len(data_diff) - window_size - step, step):
    train_win = data_diff[start:start+window_size]
    test_win = data_diff[start+window_size:start+window_size+step]

    model = SARIMAX(train_win['y'],
                    exog=train_win[['x1','x2']],
                    order=(1,1,1),
                    seasonal_order=(1,1,1,50)).fit(disp=False)

    pred = model.forecast(steps=len(test_win), exog=test_win[['x1','x2']])
    rmse = np.sqrt(mean_squared_error(test_win['y'], pred))
    rmse_scores.append(rmse)

print("Average Rolling RMSE:", np.mean(rmse_scores))

# =========================================
# 10. PLOT RESULTS
# =========================================
plt.figure(figsize=(12,6))
plt.plot(test.index, test['y'], label="Actual")
plt.plot(test.index, sarimax_forecast, label="SARIMAX Forecast")
plt.plot(test.index, hw_forecast, label="Holt-Winters Forecast")
plt.legend()
plt.title("Model Forecast Comparison")
plt.show()
