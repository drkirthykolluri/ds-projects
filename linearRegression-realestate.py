import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#step 2 read from dataset

dataset = pd.read_csv('/Users/kirthy/Desktop/PythonPractice/data/Real estate.csv')
df_binary = dataset[['X1 transaction date', 'X2 house age', 'X3 distance to the nearest MRT station', 'X4 number of convenience stores', 'X5 latitude', 'X6 longitude', 'Y house price of unit area' ]]
df_binary.columns = [['X1 transaction date', 'X2 house age', 'X3 distance to the nearest MRT station', 'X4 number of convenience stores', 'X5 latitude', 'X6 longitude', 'Y house price of unit area' ]]
df_binary.head()

#step 3- plot
sns.lmplot(x ="X3 distance to the nearest MRT station", y ="house price of unit area", data = df_binary, order = 2, ci = None)
plt.show()

#step 4 clean the data
df_binary.fillna(method ='ffill', inplace = True)

#step 5 - train the model
X = np.array(df_binary.loc['X1 transaction date', 'X2 house age', 'X3 distance to the nearest MRT station', 'X4 number of convenience stores', 'X5 latitude', 'X6 longitude', ]).reshape(-1, 1)
Y = np.array(df_binary['Y house price of unit area']).reshape(-1, 1)
df_binary.dropna(inplace = True)
X_train, Y_train, X_test, Y_test = train_test-split(X, Y, test_size = 0.25)
regr = LinearRegression()
regr.fit(X_train, Y_train)
print(regr.score(X_test, Y_test))

y_pred = regr.predict(X_test)
plt.scatter(X_test, Y_test, color = 'b')
plt.plot(X_test, Y_pred, color = 'k')
plt.show()
