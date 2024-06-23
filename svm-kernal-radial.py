from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import accuracy_score
#import numpy as np
from sklearn import preprocessing
# Read the Dataset
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