# https://www.kaggle.com/teertha/ushealthinsurancedataset
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
import numpy as np
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')

# Read dataset
df = pd.read_csv('insurance.csv')
print(df.head())
print(df.describe())

# Dirty value detection
print(df.isnull().sum())
df.dropna(axis=0, inplace=True)
print(df.shape)

# Encoding categorical value with LabelEncoder
label = LabelEncoder()
df['sex'] = label.fit_transform(df['sex'])
df['smoker'] = label.fit_transform(df['smoker'])
df['region'] = label.fit_transform(df['region'])

# Draw heat map
heatmap_data = df
colormap = plt.cm.PuBu
plt.figure(figsize=(15, 15))
plt.title("Correlation of Features", y=1.05, size=15)
sns.heatmap(heatmap_data.corr(), linewidths=0.1, square=False, cmap=colormap, linecolor="white",
            annot=True, annot_kws={"size": 8})
plt.show()

# Feature selection
df.drop('sex', axis=1, inplace=True)
df.drop('children', axis=1, inplace=True)
df.drop('region', axis=1, inplace=True)


X = df[['age', 'bmi', 'smoker']]
y = df['charges']

# Scaling with MinMaxScaler
minMaxScaler = MinMaxScaler()
df_scaled = minMaxScaler.fit_transform(X)
df_scaled = pd.DataFrame(df_scaled, columns=['age', 'bmi', 'smoker'])


# Regression with OLS
linReg = LinearRegression()
MSE = cross_val_score(linReg, X, y, scoring='neg_mean_squared_error', cv=5)
mean_MSE = np.mean(MSE)
print("\n--------------------------------")
print("MSE using OLS: ", mean_MSE)


# Regression with Ridge
print("--------------------------------")
print("MSE using Ridge")
ridge = Ridge()
parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,  1e-2, 1, 5, 10, 20]}
ridgeReg = GridSearchCV(ridge, parameters, scoring='neg_mean_squared_error', cv=5)
ridgeReg.fit(X, y)
for i in range(len(ridgeReg.cv_results_["params"])):
    print(ridgeReg.cv_results_["params"][i],
          "Accuracy : ", ridgeReg.cv_results_["mean_test_score"][i])

print('Best parameter : ', ridgeReg.best_params_)
print('Best accuracy : ', ridgeReg.best_score_)


# Regression with Lasso
print("--------------------------------")
print("MSE using Lasso")
lasso = Lasso()
lassoReg = GridSearchCV(lasso, parameters, scoring='neg_mean_squared_error', cv=5)
lassoReg.fit(X, y)
for i in range(len(lassoReg.cv_results_["params"])):
    print(lassoReg.cv_results_["params"][i],
          "Accuracy : ", lassoReg.cv_results_["mean_test_score"][i])

print('Best parameter : ', lassoReg.best_params_)
print('Best accuracy : ', lassoReg.best_score_)


# Regression with Elastic Net
print("--------------------------------")
print("MSE using Elastic Net")
elastic = ElasticNet()
elasReg = GridSearchCV(elastic, parameters, scoring='neg_mean_squared_error', cv=5)
elasReg.fit(X, y)
for i in range(len(elasReg.cv_results_["params"])):
    print(elasReg.cv_results_["params"][i], "Accuracy : ", elasReg.cv_results_["mean_test_score"][i])

print('Best parameter : ', elasReg.best_params_)
print('Best accuracy : ', elasReg.best_score_)
