import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression

import pandas as pd
import seaborn as sns
data = pd.read_csv('/Users/kirthy/Desktop/PythonPractice/data/advertising.csv')
data_binary = data[['TV', 'Radio', 'Newspaper', 'Sales']]

data_binary = data.loc[:, ['TV', 'Radio', 'Newspaper', 'Sales']]
data_binary.head(200)

# Step 2
import matplotlib.pyplot as plt
data_binary.plot(x = 'Radio', y = 'Sales', style = 'o')
plt.xlabel('Radio')
plt.ylabel('Sales')
plt.show()

x= pd.DataFrame(data, columns= ['TV', 'Radio', 'Newspaper'])
y= pd.DataFrame(data, columns= ['Sales'])

#step 3
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state= 1)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

#Step 4
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)
# X is a numpy array with your features
# y is the label array
enc = OneHotEncoder.__init__(sparse=False)
x_transform = enc.fit_transform(x)

# apply your linear regression as you want
model = LinearRegression()
model.fit(x_transform, y)


#step 5
y_predictions = regressor.predict(x_test)
print('Predictions:', y_predictions)
print('Intercepts:', regressor.intercept_)
print('Coefficients:', regressor.coef_)
print("Mean squared error: %.2f %" , np.mean((model.predict(x_transform) - y) ** 2))