import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path = kagglehub.dataset_download("haiderrasoolqadri/nvidia-corporation-nvda-stock-2015-2024")

print("Path to dataset files:", path)

dataframe = pd.read_csv(path + "/nvidia_stock_2015_to_2024.csv")
#drop unnamed column
dataframe.drop(columns = ['Unnamed: 0'], inplace = True)
#convert date to datetime for time series analysis
dataframe['date']= pd.to_datetime(dataframe['date'])
#set date as index column
dataframe.set_index('date', inplace = True)

#check for duplicate data

print(dataframe.duplicated().sum())

