# NVIDIA-Stock-Predictor

### Description:
(Work in progress, still developping and improving as skills and knowledge develops)




## 📊 Project Summary: NVDA Stock Price Forecasting (ARIMA Model)


### 🎯 Objective
My main objective for this project was to try and forecast NVIDIA's (NVDA) stock closing prices using historical data from kagglehub using the ARIMA model for time series analysis.

### 🧰 Tools & Technologies used
Python: pandas, matplotlib, statsmodels, plotly

R: auto.arima() for model selection

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

### 💡 Key Learnings

How to clean and format stock data for time series analysis

Interpreting ADF test, ACF/PACF plots

Leveraging R + Python together for model selection and implementation

Creating professional-grade visual and forecasts

### 📁 Challenges

- Had an issue with trying to use auto arima function in Python > veresions did not align. Therefore decided to upload data into a csv file and use RStudio to do auto_arima.
- Still trying to gain deep understanding of the ARIMA model and how it is able to forecast "accurate" predictions.
- Forecast is not completely accurate to real-life, but it makes sense as stock predictions is almost impossible.


