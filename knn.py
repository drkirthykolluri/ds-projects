import pandas as pd
df = pd.read_csv('/Users/kirthy/Desktop/PythonPractice/data/insurance.csv')
print(df.head())
print(df.info())

import seaborn as sns
import matplotlib.pyplot as plt
sns.scatterplot(x=df['smoker'], y=df['charges'], hue=df['bmi'])
plt.xticks(rotation=45, ha='right')

pre_df = pd.get_dummies(df, columns=['smoker', 'sex', 'region'], drop_first=True)
print(pre_df.head())

from sklearn.model_selection import train_test_split

x = pre_df['age', 'sex', 'bmi', 'children', 'smoker', 'region']
y = pre_df['charges']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.33, random_state=125
)