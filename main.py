import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path = kagglehub.dataset_download("haiderrasoolqadri/nvidia-corporation-nvda-stock-2015-2024")

print("Path to dataset files:", path)

dataframe = pd.read_csv(path + "/nvidia_stock_2015_to_2024.csv")
#print(dataframe.head())
#checking for null data
print(dataframe.isnull().sum())
