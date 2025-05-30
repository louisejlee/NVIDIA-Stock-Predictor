import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Download dataset
path = kagglehub.dataset_download("haiderrasoolqadri/nvidia-corporation-nvda-stock-2015-2024")
print("Path to dataset files:", path)

# Load dataset
dataframe = pd.read_csv(path + "/nvidia_stock_2015_to_2024.csv")

# Preprocess dataset
dataframe.drop(columns=['Unnamed: 0'], inplace=True)
dataframe['date'] = pd.to_datetime(dataframe['date'])
dataframe.set_index('date', inplace=True)
dataframe = dataframe.asfreq('B')

# Check for duplicate data
if dataframe.duplicated().sum() > 0:
    print("Duplicate data found. Removing duplicates...")
    dataframe.drop_duplicates(inplace=True)

# Check for NaN data
close_prices = dataframe['close'].copy()
close_prices = close_prices.replace([np.inf, -np.inf], np.nan)
close_prices = close_prices.dropna()

# DATA VISUALIZATION -------------
# plot stock closing prices
"""
plt.figure(figsize=(12, 6))
plt.plot(dataframe['close'], label = 'Closing Price (USD)')
plt.title('NVDA Stock Closing Price')
plt.xlabel('Date')
plt.ylabel('Closing Price (USD)')
plt.legend()
#plt.show()
"""
# check for stationary using Augmented Dickey-Fuller test
from statsmodels.tsa.stattools import adfuller
result = adfuller(close_prices)
print("ADF statistic: ", result[0])
print("p-value: ", result[1])
# reject null hypothesis if p<0.05, meaning data is stationary

# result
# adf stat = 5.114
# p-value = 1.0

#thus we must do differencing to make data stationary
df_diff = dataframe['close'].diff().dropna()

#rerun ADF test
result2 = adfuller(df_diff)
print("ADF statistic: ", result2[0])
print("p-value: ", result2[1])
# adf stat = -6.803
# p-value = 2.207e-09

# data is now stationary!
# import ARIMA model
# from statsmodels.tsa.arima.model import ARIMA

# use ACF and PACF plots to find p and q variables
""" from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plt.figure(figsize=(12, 5))

# ACF on differenced data (Autocorrelation Function)
# used to find q value (AutoRegressive AR(q))
plt.subplot(1, 2, 1) # to show plots for ACF and PACF side by side
plot_acf(df_diff, lags=40, ax=plt.gca())
plt.title("ACF Plot")


# PACF on differenced data (Partial Autocorrelation Function)
# used to find p value (Moving Average MA(p))
plt.subplot(1, 2, 2)
plot_pacf(df_diff, lags=40, ax=plt.gca()) #ax ensure plots are in correct space
plt.title("PACF Plot")

plt.tight_layout() # ensures no overlaps between plots
#plt.show()

# from the plots, we can see that p=1 and q=0
"""

# fit model using found p and q values
# model was not optimal, thus we will use R to find values with auto_arima

# send close_prices to a csv file so R can read
# close_prices.to_csv("close_prices.csv")


# fit model using found p and q values
from statsmodels.tsa.arima.model import ARIMA

model = ARIMA(dataframe['close'], order=(2, 2, 3)) # using p=2 d=2 and q=3 from auto_arima
model_fit = model.fit()
print(model_fit.summary())

# FORECASTING --------------------------

n_steps = 30 # number of days to forecast
forecast = model_fit.forecast(steps = n_steps)
forecast_result = forecast

# forecast_result.to_csv("forecast_result.csv")




# plot forecast for visualizations
from datetime import timedelta

# Create the forecast DataFrame
forecast_df = forecast.to_frame(name="Forecast")

# Set real dates as index
close_prices.index = pd.to_datetime(close_prices.index)

last_date = close_prices.index[-1]  # must be a datetime index
forecast_df.index = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30, freq='B')  # 'B' = business days


plt.figure(figsize=(10, 5))
plt.plot(close_prices[-60:], label='Actual')  # last 60 days of actuals
plt.plot(forecast_df, label='Forecast', linestyle='--')
plt.title("NVDA 30-Day Forecast")
plt.xlabel("Date")
plt.ylabel("Predicted Closing Price (USD)")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("nvda_forecast_plot.pdf")

