import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv('/Users/kirthy/Desktop/PythonPractice/data/Salary_Data.csv')
df_binary = df[['YearsExperience', 'Salary']]

# Taking only the selected two attributes from the dataset
df_binary.columns = ['YearsExperience', 'Salary']
# display the first 5 rows
df_binary.head()

# plotting the Scatter plot to check relationship between Sal and Temp
sns.lmplot(x="YearsExperience", y="Salary", data=df_binary, order=2, ci=None)
plt.show()
df_binary.fillna(method='ffill', inplace=True)

X = np.array(df_binary['YearsExperience']).reshape(-1, 1)
y = np.array(df_binary['Salary']).reshape(-1, 1)

# Separating the data into independent and dependent variables
# Converting each dataframe into a numpy array
# since each dataframe contains only one column
df_binary.dropna(inplace=True)

# Dropping any rows with Nan values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# Splitting the data into training and testing data
regr = LinearRegression()

regr.fit(X_train, y_train)
print(regr.score(X_test, y_test))

y_pred = regr.predict(X_test)
plt.scatter(X_test, y_test, color='b')
plt.plot(X_test, y_pred, color='k')

plt.show()
# Data scatter of predicted values
