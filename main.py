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







