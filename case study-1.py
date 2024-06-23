import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('/Users/kirthy/Desktop/PythonPractice/data/insurance.csv')
data.head()
data_binary = data[['age', 'sex', '	bmi', 'children', 'smoker', ' region', 'charges']]
data_binary.head(1338)
all_data = pd.get_dummies(data)
all_data.head()
#plt.style.use('classic')
#first_hist =

label_encoder = preprocessing.LabelEncoder()
data["sex"]= label_encoder.fit_transform(data["sex"]) # Encoding Output Variable