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

#step 5
y_predictions = regressor.predict(x_test)
print('Predictions:', y_predictions)
print('Intercepts:', regressor.intercept_)
print('Coefficients:', regressor.coef_)

#step 6 calculating y-y hat
residuals= y_test - y_predictions
print('Residual:', residuals)

#step 7 plot for y tets and y pred
#import matplotlib.pyplot as plt
#import seaborn as sns
#sns.scatterplot(x=y_test, y = y_predictions, ci=None, s=140)
#plt.xlabel('y_test')
#plt.ylabel('y_predictions')
#plt.show()

#step 8
from sklearn.metrics import mean_absolute_error
print('MAE:', mean_absolute_error(y_test,y_predictions))

#step 9
from sklearn.metrics import mean_squared_error
print("MSE",mean_squared_error(y_test,y_predictions))

#Step 10- Root Mean Squared Error
import numpy as np
print("RMSE",np.sqrt(mean_squared_error(y_test,y_predictions)))

#step 11- Rsquare-R Squared(R2): R2 is also called the coefficient of determination or goodness of fit score regression function.
# It measures how much irregularity in the dependent variable the model can explain. The R2 value is between 0 to 1,
# and a bigger value shows a better fit between prediction and actual value.
from sklearn.metrics import r2_score
r2 = r2_score(y_test,y_predictions)
print("Rsquare:", r2)
