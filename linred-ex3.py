import pandas as pd
data = pd.read_csv('/Users/kirthy/Desktop/PythonPractice/data/advertising.csv')
data_binary = data[['TV', 'Radio', 'Newspaper', 'Sales']]
data.head()

data_binary = data.loc[:, ['TV', 'Radio', 'Newspaper', 'Sales']]
data_binary.head(200)

import matplotlib.pyplot as plt
data_binary.plot(x = 'TV', y= 'Sales', style = 'o')
plt.xlabel('TV')
plt.ylabel('Sales')
plt.show()

x = pd.DataFrame(data, columns = ['TV', 'Radio', 'Newspaper'])
y = pd.DataFrame(data, columns= ['Sales'])


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state= 2)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)


print(regressor.intercept_)
print(regressor.coef_)