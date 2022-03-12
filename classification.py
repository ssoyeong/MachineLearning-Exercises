import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
warnings.filterwarnings('ignore')

def scaleDF(X):

    # scaling with MinMaxScaler()
    minMaxScaler = MinMaxScaler()
    X_minMax = minMaxScaler.fit_transform(X)
    X_minMax = pd.DataFrame(X_minMax, columns=x_columns)

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(15, 15))
    ax1.set_title('Before Scaling')
    ax2.set_title('After MinMax Scaling')

    for i in x_columns:
        sns.kdeplot(X[i], ax=ax1)
        sns.kdeplot(X_minMax[i], ax=ax2)

    # scaling with MaxAbsScaler()
    maxAbsScaler = MaxAbsScaler()
    X_maxAbs = maxAbsScaler.fit_transform(X)
    X_maxAbs = pd.DataFrame(X_maxAbs, columns=x_columns)

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(15, 15))
    ax1.set_title('Before Scaling')
    ax2.set_title('After MaxAbs Scaling')

    for i in x_columns:
        sns.kdeplot(X[i], ax=ax1)
        sns.kdeplot(X_maxAbs[i], ax=ax2)

    plt.show()

    return X_minMax, X_maxAbs


def fitAlgorithm(X, k):

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    print("\n\nBefore Oversampling, the shape of X_train: {}".format(X_train.shape))
    print("Before Oversampling, the shape of y_train: {}".format(y_train.shape))
    print("Before Oversampling, counts of class '1': {}".format(sum(y_train == 1)))
    print("Before Oversampling, counts of class '0': {}".format(sum(y_train == 0)))

    # SMOTE resampling
    sm = SMOTE(random_state=2)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train.ravel())
    print("---------------------------------------------------")
    print("After Oversampling, the shape of X_train: {}".format(X_train_res.shape))
    print("After Oversampling, the shape of y_train: {}".format(y_train_res.shape))
    print("After Oversampling, counts of class '1': {}".format(sum(y_train_res == 1)))
    print("After Oversampling, counts of class '0': {}".format(sum(y_train_res == 0)))

    kfold = KFold(k, shuffle=True)

    ####### Decision tree with entropy #######

    # various parameters
    parameters = { 'max_depth': [2, 3, 4, 5, 6, 7, 8],
                   'min_samples_leaf': [1, 2, 3, 4],
                   'min_samples_split': [2, 3, 4]}

    # Make the model
    dtEtp = DecisionTreeClassifier(criterion='entropy')
    dtEtp_model = GridSearchCV(dtEtp, parameters, cv=kfold)
    dtEtp_model.fit(X_train, y_train)

    dtEtp_params = dtEtp_model.best_params_
    dtEtp_score = round(dtEtp_model.best_score_, 3)
    dtEtp_best = dtEtp_model.best_estimator_
    print('\n================= Decision Tree(entropy) =================')
    print('Best parameter : ', dtEtp_params)
    print('Best score : ', dtEtp_score)

    # Predict y
    dtEtp_y_pred = dtEtp_best.predict(X_test)

    # Make confusion matrix
    dtEtp_cf = confusion_matrix(y_test, dtEtp_y_pred)
    dtEtp_total = np.sum(dtEtp_cf, axis=1)
    dtEtp_cf = dtEtp_cf / dtEtp_total[:, None]
    dtEtp_cf = pd.DataFrame(dtEtp_cf, index=["TN", "FN"], columns=["FP", "TP"])

    # Visualization
    plt.figure(figsize=(10, 7))
    plt.title("Confusion Matrix with Decision tree(entropy)")
    sns.heatmap(dtEtp_cf, annot=True, annot_kws={"size": 20})
    plt.show()

    # Precision, recall, f1 score
    dtEtp_p = round(precision_score(y_test, dtEtp_y_pred), 3)
    print("precision score :", dtEtp_p)
    dtEtp_r = round(recall_score(y_test, dtEtp_y_pred), 3)
    print("recall score :", dtEtp_r)
    dtEtp_f = round(f1_score(y_test, dtEtp_y_pred), 3)
    print("F1 score :", dtEtp_f)

    ####### Decision tree with gini #######

    # various parameters
    parameters = {'max_depth': [2, 3, 4, 5, 6, 7, 8],
                  'min_samples_leaf': [1, 2, 3, 4],
                  'min_samples_split': [2, 3, 4]}

    # Make the model
    dtGN = DecisionTreeClassifier(criterion='gini')
    dtGN_model = GridSearchCV(dtGN, parameters, cv=kfold)
    dtGN_model.fit(X_train, y_train)

    dtGN_params = dtGN_model.best_params_
    dtGN_score = round(dtGN_model.best_score_, 3)
    dtGN_best = dtGN_model.best_estimator_
    print('\n================= Decision Tree(gini) =================')
    print('Best parameter : ', dtGN_params)
    print('Best score : ', dtGN_score)

    # Predict y
    dtGN_y_pred = dtGN_best.predict(X_test)

    # Make confusion matrix
    dtGN_cf = confusion_matrix(y_test, dtGN_y_pred)
    dtGN_total = np.sum(dtEtp_cf, axis=1)
    dtGN_cf = dtGN_cf / dtGN_total[:, None]
    dtGN_cf = pd.DataFrame(dtGN_cf, index=["TN", "FN"], columns=["FP", "TP"])

    # Visualization
    plt.figure(figsize=(10, 7))
    plt.title("Confusion Matrix with Decision tree(gini)")
    sns.heatmap(dtGN_cf, annot=True, annot_kws={"size": 20})
    plt.show()

    # Precision, recall, f1 score
    dtGN_p = round(precision_score(y_test, dtGN_y_pred), 3)
    print("precision score :", dtGN_p)
    dtGN_r = round(recall_score(y_test, dtGN_y_pred), 3)
    print("recall score :", dtGN_r)
    dtGN_f = round(f1_score(y_test, dtGN_y_pred), 3)
    print("F1 score :", dtGN_f)


    ####### Logistic regression #######

    # various parameters
    parameters = {'C': [0.1, 1.0, 10.0],
                  'solver': ["liblinear", "lbfgs", "sag"],
                  'max_iter': [50, 100, 200]}

    # Make the model
    logisticRegr = LogisticRegression()
    lr_model = GridSearchCV(logisticRegr, parameters, cv=kfold)
    lr_model.fit(X_train, y_train)

    lr_params = lr_model.best_params_
    lr_score = round(lr_model.best_score_, 3)
    lr_best = lr_model.best_estimator_
    print('\n================= Logistic Regression =================')
    print('Best parameter : ', lr_params)
    print('Best score : ', lr_score)

    # Predict y
    lr_y_pred = lr_best.predict(X_test)

    # Make confusion matrix
    lr_cf = confusion_matrix(y_test, lr_y_pred)
    lr_total = np.sum(lr_cf, axis=1)
    lr_cf = lr_cf / lr_total[:, None]
    lr_cf = pd.DataFrame(lr_cf, index=["TN", "FN"], columns=["FP", "TP"])

    # Visualization
    plt.figure(figsize=(10, 7))
    plt.title("Confusion Matrix with Logistic Regression")
    sns.heatmap(lr_cf, annot=True, annot_kws={"size": 20})
    plt.show()

    # Precision, recall, f1 score
    lr_p = round(precision_score(y_test, lr_y_pred), 3)
    print("precision score :", lr_p)
    lr_r = round(recall_score(y_test, lr_y_pred), 3)
    print("recall score :", lr_r)
    lr_f = round(f1_score(y_test, lr_y_pred), 3)
    print("F1 score :", lr_f)


    ####### Support vector machine #######

    # various parameters
    parameters = {'C': [0.1, 1.0, 10.0],
                  'kernel': ["linear", "rbf", "sigmoid"],
                  'gamma': [0.01, 0.1, 1.0, 10.0]}

    # Make the model
    svc = SVC()
    sv_model = GridSearchCV(svc, parameters, cv=kfold)
    sv_model.fit(X_train, y_train)

    sv_params = sv_model.best_params_
    sv_score = round(sv_model.best_score_, 3)
    sv_best = sv_model.best_estimator_
    print('\n================= SVM =================')
    print('Best parameter : ', sv_params)
    print('Best score : ', sv_score)

    # Predict y
    sv_y_pred = sv_best.predict(X_test)

    # Make confusion matrix
    sv_cf = confusion_matrix(y_test, sv_y_pred)
    sv_total = np.sum(sv_cf, axis=1)
    sv_cf = sv_cf / sv_total[:, None]
    sv_cf = pd.DataFrame(sv_cf, index=["TN", "FN"], columns=["FP", "TP"])

    # Visualization
    plt.figure(figsize=(10, 7))
    plt.title("Confusion Matrix with SVM")
    sns.heatmap(sv_cf, annot=True, annot_kws={"size": 20})
    plt.show()

    # Precision, recall, f1 score
    sv_p = round(precision_score(y_test, sv_y_pred), 3)
    print("precision score :", sv_p)
    sv_r = round(recall_score(y_test, sv_y_pred), 3)
    print("recall score :", sv_r)
    sv_f = round(f1_score(y_test, sv_y_pred), 3)
    print("F1 score :", sv_f)

    print("\n================= The Best Scores of each classifier =================")
    print("Decision Tree Classification(entropy) score: ", dtEtp_score)
    print("Decision Tree Classification(gini) score: ", dtGN_score)
    print("Logistic Regression score: ", lr_score)
    print("Support Vector Machine score: ", sv_score)

    best_score = max(dtEtp_score, dtGN_score, lr_score, sv_score)

    if best_score == dtEtp_score:
        best_params = dtEtp_params
        best_classifier = 'Decision Tree(entropy)'
    elif best_score == dtGN_score:
        best_params = dtGN_params
        best_classifier = 'Decision Tree(gini)'
    elif best_score == lr_score:
        best_params = lr_params
        best_classifier = 'Logistic Regression'
    elif best_score == sv_score:
        best_params = sv_params
        best_classifier = 'Support Vector Machine'

    return best_classifier, best_params, best_score


def findBestOfBest(X_dataset, k):

    max_score = 0.0
    max_i = 0
    max_j = 0

    for i in range(len(X_dataset)):
        for j in range(len(k)):
            best_classifier, best_params, best_score = fitAlgorithm(X_dataset[i], k[j])

            if i == 0:
                print("================= With MinMaxScaler and k of {} =================".format(k[j]))
            elif i == 1:
                print("================= With MaxAbsScaler and k of {} =================".format(k[j]))

            if max_score < best_score:
                max_score = best_score
                max_i = i
                max_j = j

    best_scaler = ""
    if max_i == 0:
        best_scaler = 'MinMaxScaler'
    elif max_i == 1:
        best_scaler = 'MaxAbsScaler'

    best_classifier, best_params, best_score = fitAlgorithm(X_dataset[max_i], k[max_j])
    return best_classifier, best_params, best_score, best_scaler, k[max_j]


# Read dataset
df = pd.read_csv('breast-cancer-wisconsin.data', header=None)
df.columns = ['ID', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion',
              'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class']
print(df.head())
print(df.describe())

# Dirty value detection
df = df.apply(pd.to_numeric, errors='coerce').fillna(np.nan)
# print(df.isnull().sum())
df.dropna(axis=0, inplace=True)
# print(df.shape)

# Draw heat map
heatmap_data = df
colormap = plt.cm.PuBu
plt.figure(figsize=(15, 15))
plt.title("Correlation of Features", y=1.05, size=15)
sns.heatmap(heatmap_data.corr(), linewidths=0.1, square=False, cmap=colormap, linecolor="white",
            annot=True, annot_kws={"size": 8})
plt.show()

# Feature selection
df_ID = df["ID"]
df.drop('ID', axis=1, inplace=True)
# print(df.head())


X = df.drop(['Class'], 1)
y = df['Class'].replace({2:0, 4:1})
x_columns = X.columns

# Scaling using MinMaxScaler and MaxAbsScaler
X_minMax, X_maxAbs = scaleDF(X)

# Fit algorithm
X_dataset = [X_minMax, X_maxAbs]
k = [10, 5, 3]

final_classifier, final_params, final_score, final_scaler, final_k = findBestOfBest(X_dataset, k)

print('\n======= The best of the best combination =======')
print('Using the dataset with {} and k of {}.'.format(final_scaler, final_k))
print('The score of the classification using {} with parameters {} is {}.'
      .format(final_classifier, final_params, final_score))
