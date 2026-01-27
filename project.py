# ======================================================
# IMPORT LIBRARIES
# ======================================================
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.api import VAR
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ======================================================
# DATA GENERATION (Multivariate, Trend + Seasonality)
# ======================================================
np.random.seed(42)
t = np.arange(600)

trend = 0.04 * t
season = 8 * np.sin(2 * np.pi * t / 40)

noise1 = np.random.normal(0, 3, len(t))
noise2 = np.random.normal(0, 2, len(t))

y = trend + season + noise1
x1 = 0.6 * y + noise2
x2 = np.cos(2 * np.pi * t / 30) * 5 + np.random.normal(0, 1, len(t))

data = pd.DataFrame({'y': y, 'x1': x1, 'x2': x2})

# ======================================================
# STATIONARITY CHECK + DIFFERENCING
# ======================================================
def adf_test(series, name):
    p_value = adfuller(series)[1]
    print(f"{name} ADF p-value: {p_value:.4f}")

for col in data.columns:
    adf_test(data[col], col)

data_diff = data.diff().dropna()

# ======================================================
# BASELINE MODEL (Holt-Winters)
# ======================================================
def baseline_hw(train, test):
    model = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=40).fit()
    return model.forecast(len(test))

# ======================================================
# ADVANCED MODEL 1: SARIMAX
# ======================================================
def sarimax_model(train, test):
    model = SARIMAX(train['y'], exog=train[['x1','x2']],
                    order=(1,1,1), seasonal_order=(1,1,1,40)).fit(disp=False)
    return model.forecast(len(test), exog=test[['x1','x2']])

# ======================================================
# ADVANCED MODEL 2: VAR
# ======================================================
def var_model(train, test):
    model = VAR(train)
    results = model.fit(maxlags=5, ic='aic')
    forecast = results.forecast(train.values[-results.k_ar:], steps=len(test))
    return pd.DataFrame(forecast, columns=train.columns)['y']

# ======================================================
# ROLLING WINDOW CROSS-VALIDATION
# ======================================================
window = 350
step = 50

results = []

for start in range(0, len(data_diff) - window - step, step):
    train = data_diff[start:start+window]
    test = data_diff[start+window:start+window+step]

    hw_pred = baseline_hw(train['y'], test['y'])
    sarimax_pred = sarimax_model(train, test)
    var_pred = var_model(train, test)

    for name, pred in zip(['Holt-Winters','SARIMAX','VAR'],
                          [hw_pred, sarimax_pred, var_pred]):
        rmse = np.sqrt(mean_squared_error(test['y'], pred))
        mae = mean_absolute_error(test['y'], pred)
        results.append([name, rmse, mae])

results_df = pd.DataFrame(results, columns=['Model','RMSE','MAE'])

print("\nCross-Validation Summary:")
print(results_df.groupby('Model').agg(['mean','std']))
