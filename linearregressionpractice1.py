import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#step 2
df = pd.read_csv('/Users/kirthy/Desktop/PythonPractice/data/car data.csv')
df_binary = df[['Car_Name', 'Year', 'Selling_Price', 'Present_Price', 'Kms_Driven', 'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner']]
df_binary.columns = ['Car_Name', 'Year', 'Selling_Price', 'Present_Price', 'Kms_Driven', 'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner']
df_binary.head()

#step 3
sns.lmplot(x ="Year", y ="Present_Price", data = df_binary, order = 2, ci = None)
plt.show()

#step 4 clean the data
df_binary.fillna(method ='ffill', inplace = True)

#step 5 TRaining our model
x = np.array(df_binary.loc['Car_Name', 'Year', 'Selling_Price', 'Present_Price', 'Kms_Driven', 'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner']).reshape(-1, 1)
y = np.array(df_binary['Present_Price']).reshape(-1, 1)
df_binary.dropna(inplace = True)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.25)
regr = LinearRegression()
regr.fit(x_train, y_train)
print(regr.score(x_test, y_test))

y_pred = regr.predict(x_test)
plt.scatter(x_test, y_test, color = 'b')
plt.plot(x_test, y_pred, color = 'k')

plt.show()

