import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

path = kagglehub.dataset_download("haiderrasoolqadri/nvidia-corporation-nvda-stock-2015-2024")

print("Path to dataset files:", path)

dataframe = pd.read_csv(path + "/nvidia_stock_2015_to_2024.csv")
#drop unnamed column
dataframe.drop(columns = ['Unnamed: 0'], inplace = True)
#convert date to datetime for time series analysis
dataframe['date']= pd.to_datetime(dataframe['date'])
#set date as index column
dataframe.set_index('date', inplace = True)

#check for duplicate data -> no duplicate data
#print(dataframe.duplicated().sum())

# DATA VISUALIZATION -------------
# plot stock closing prices

plt.figure(figsize=(12, 6))
plt.plot(dataframe['close'], label = 'Closing Price (USD)')
plt.title('NVDA Stock Closing Price')
plt.xlabel('Date')
plt.ylabel('Closing Price (USD)')
plt.legend()
#plt.show()

# check for stationary using Augmented Dickey-Fuller test
from statsmodels.tsa.stattools import adfuller
result = adfuller(dataframe['close'])
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
from statsmodels.tsa.arima.model import ARIMA

# use ACF and PACF plots to find p and q variables
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
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
plt.show()

# from the plots, we can see that p=1 and q=0

# fit model using found p and q values
model = ARIMA(dataframe['close'], order=(1, 1, 0))
model_fit = model.fit()
print(model_fit.summary())






