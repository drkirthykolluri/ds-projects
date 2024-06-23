import pandas as pd

data = pd.read_csv('/Users/kirthy/Desktop/PythonPractice/data/Real estate.csv')
data_binary = data[['X1 transaction date', 'X2 house age', 'X3 distance to the nearest MRT station', 'X4 number of convenience stores', 'X5 latitude', 'X6 longitude', 'Y house price of unit area' ]]
# step 2- examine data set dimensions
data.head()

# step 3- preview predictor and response variables
data_binary = data.loc[:, ['X2 house age', 'Y house price of unit area']]
data_binary.head(413)

# step 4- visulaize
import matplotlib.pyplot as plt

data_binary.plot(x='X2 house age', y='Y house price of unit area', style='o')
plt.xlabel('X2 house age')
plt.ylabel('Y house price of unit area')
plt.show()

# step 5- segregate data
x = pd.DataFrame(data['X2 house age'])
y = pd.DataFrame(data['Y house price of unit area'])

# Step 6- Partition data
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

#step 7- train the model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

print(regressor.intercept_)
print(regressor.coef_)

