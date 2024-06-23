import pandas as pd
import seaborn as sns
#from statsmodels.formula.api import ols
#from scipy import stats
#import statsmodel.api as sm
data = pd.read_csv('/Users/kirthy/Desktop/PythonPractice/data/insurance.csv')
data_binary = data[['age', 'sex', 'bmi', 'children', 'smoker', 'region', 'charges']]
df_dummy = pd.get_dummies(data, columns = ["sex", "region", "smoker"])
df_dummy.head()

