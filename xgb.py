import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")
diamonds = sns.load_dataset("diamonds")
print(diamonds.head())
print(diamonds.shape)
print(diamonds.describe())

from sklearn.model_selection import train_test_split
x = diamonds.drop('price', axis=1)
y = diamonds[['price']]

cats = x.select_dtypes(exclude=np.number).colums.tolist()
for col in cats:
    x[col] = x[col].astype('category')
    print(x.dtypes)

x_train, y_train, x_test, y_test = train_test_split(x, y, random_state=1)

import xgboost as xgb
dtrain_reg = xgb.DMatrix(x_train, y_train, enable_categorical=True)
dtest_reg = xgb.DMatrix(x_test, y_test, enable_categorical=True)



params = {"objective": "reg:squarederror", "tree_method": "gpu_hist"}

n = 100
model = xgb.train(
   params=params,
   dtrain=dtrain_reg,
   num_boost_round=n,
)
from sklearn.metrics import mean_squared_error

preds = model.predict(dtest_reg)

rmse = mean_squared_error(y_test, preds, squared=False)

print(f"RMSE of the base model: {rmse:.3f}")
