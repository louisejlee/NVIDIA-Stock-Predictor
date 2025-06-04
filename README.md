# NVIDIA-Stock-Predictor

### Description:
(Work in progress, still developping and improving as skills and knowledge develops)




## 📊 Project Summary: NVDA Stock Price Forecasting (ARIMA Model)


### 🎯 Objective
To forecast NVIDIA's (NVDA) stock closing prices using historical data and time series modeling, helping visualize short-term market trends.

### 🧰 Tools & Technologies
Python: pandas, matplotlib, statsmodels, plotly

R: forecast = auto.arima() for model selection

Data Source: Kaggle dataset (2015–2024 NVDA stock data)

### 🔄 Process Overview

1. Data Preparation
Loaded and cleaned NVDA stock data

Converted date column to datetime and set as index

Ensured consistent daily frequency (Business Day)

Removed any NaN or infinite values

2. Stationarity Check
Performed Augmented Dickey-Fuller (ADF) test

Applied differencing to achieve stationarity (d=2)

3. Model Selection
Used R’s auto.arima() to identify optimal ARIMA model:
ARIMA(2,2,3)

AR(2): Uses 2 past values

I(2): Differenced twice for stationarity

MA(3): Uses 3 past error terms

4. Model Fitting in Python
Fitted ARIMA(2,2,3) using statsmodels in Python

Forecasted next 30 business days of closing prices

5. Forecast Output
Exported forecast results to .csv

Generated:

Static plots (matplotlib/PDF)

### 📈 Forecast Results (Sample)

Date	Forecasted Price
2024-06-25	110.16
...	...
2024-08-06	123.37

### 💡 Key Learnings

How to clean and format stock data for time series analysis

Interpreting ADF test, ACF/PACF plots

Leveraging R + Python together for model selection and implementation

Creating professional-grade visual and forecasts

### 📁 Deliverables

main.py: Full Python script for ARIMA forecasting

data_analysis.R: R script for auto.arima model selection

nvda_forecast.csv: Forecasted price output

nvda_forecast_plot.pdf: Static visual plot


