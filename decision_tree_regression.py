import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import tree

df = pd.DataFrame(np.array([['SW', 2, 'F', 20],
                            ['Math', 3, 'M', 20],
                            ['Art', 3, 'F', 15],
                            ['English', 3, 'M', 28],
                            ['Math', 3, 'F', 26],
                            ['English', 3, 'M', 17],
                            ['Math', 3, 'F', 26],
                            ['SW', 3, 'F', 40],
                            ['SW', 3, 'M', 33],
                            ['English', 3, 'M', 18],
                            ['Math', 3, 'M', 25],
                            ['Math', 3, 'F', 30],
                            ['SW', 3, 'F', 45],
                            ['Art', 3, 'M', 20]]),
                  columns=['Major', 'Year', 'Gender', 'StudyHours'])


# Convert categorical features to numeric values using oneHotEncoder
df_oneHot = df.copy()
df_oneHot['StudyHours'] = df_oneHot['StudyHours'].astype(np.int64)

categoricalColumn = df_oneHot.columns[df_oneHot.dtypes == np.object].tolist()
for col in categoricalColumn:
    if(len(df_oneHot[col].unique()) == 2):
        df_oneHot[col] = pd.get_dummies(df_oneHot[col], drop_first=True)

df_oneHot = pd.get_dummies(df_oneHot)


# Split dataset into train and test sets
X = df_oneHot[['Year', 'Gender', 'Major_Art', 'Major_English', 'Major_Math', 'Major_SW']]
y = df_oneHot[['StudyHours']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state=20)


# Decision Tree Regression
model = DecisionTreeRegressor(max_depth=5, random_state=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


# Show the result
y_pred_df = pd.DataFrame(y_pred, columns = ['Prediction'])
y_test.reset_index(inplace=True)
y_test.drop(['index'], axis=1, inplace=True)
df_result = pd.concat([y_test, y_pred_df], axis=1, join='inner')
print(df_result)

fn=['Year', 'Gender', 'Major_Art', 'Major_English', 'Major_Math', 'Major_SW']
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
tree.plot_tree(model,
               feature_names = fn,
               filled = True);
fig.savefig('decisionTree.png')