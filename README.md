# cultus
📌 Advanced Time Series Forecasting with LSTM
📖 Overview

This project implements an advanced multivariate time series forecasting system using a Long Short-Term Memory (LSTM) neural network trained on synthetically generated financial data.

The system models:

Correlated stock price movements

Volatility clustering

Market trends

Risk dynamics

The LSTM model is compared with a traditional SARIMAX baseline model.

🚀 Features

✔ Synthetic multivariate financial dataset generation
✔ Volatility modeling (GARCH-like behavior)
✔ Advanced feature engineering
✔ Sequence modeling using LSTM
✔ Hyperparameter tuning & regularization
✔ Baseline statistical comparison (SARIMAX)
✔ Evaluation using MAE, RMSE, and MAPE

🧠 Model Architecture

LSTM (128 units)

Dropout (0.3)

LSTM (64 units)

Dense (32, ReLU)

Output Dense (1)

📊 Evaluation Metrics
Metric	Description
MAE	Mean Absolute Error
RMSE	Root Mean Squared Error
MAPE	Mean Absolute Percentage Error
⚙️ Technologies Used

Python

NumPy, Pandas

Scikit-learn

TensorFlow / Keras

Statsmodels

Matplotlib

📈 Results

The LSTM model significantly outperforms the SARIMAX baseline by capturing nonlinear market behavior and volatility clustering.

🏁 How to Run
pip install numpy pandas matplotlib scikit-learn tensorflow statsmodels
python main.py

🎯 Learning Outcomes

This project demonstrates:

Deep learning for time series forecasting

Financial data simulation

Feature engineering for temporal data

Model evaluation and comparison

Practical implementation of LSTMs

Advanced-Time-Series-LSTM/
│
├── 📄 project.py
├── 📄 README.md
├── 📄 Project Description.docx (or .pdf)
├── 📄 Implementation Explanation.docx (or .pdf)
├── 📄 Expected Deliverables.docx (or .pdf)
├── 📄 Test Report.docx (or .pdf)
