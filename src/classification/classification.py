from multiprocessing import Pool

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import cross_val_predict, cross_val_score, LeaveOneGroupOut
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from evaluation._evaluation import auc_roc_cv
from output.output import output_figure

models = [('Logistic Regression', LogisticRegression(solver='liblinear', multi_class='ovr')), ('LDA', LinearDiscriminantAnalysis()), \
          ('LinearSVC', LinearSVC()), ('CART', DecisionTreeClassifier()), ('Random Forest', RandomForestClassifier(n_estimators=100)), \
          ('NB', GaussianNB()), ('SVC', SVC()), ('XGBoost binary', XGBClassifier(objective="binary:logistic", random_state=42)), ('XGBoost mult', XGBClassifier(objective="multi:softprob", random_state=42))]

# , ('XGBoost binary', XGBClassifier(objective="binary:logistic", random_state=42)), ('XGBoost mult', XGBClassifier(objective="multi:softprob", random_state=42))
X_g = None
y_g = None
path_g = None
binary_g = None
groups_for_loocv_g = None

def classify_all(X, y, path, binary, groups_for_loocv=None):
    global X_g
    global y_g
    global path_g
    global binary_g
    global groups_for_loocv_g
    X_g = X
    y_g = y
    path_g = path
    binary_g = binary
    groups_for_loocv_g = groups_for_loocv
    with Pool(9) as p:
        p.map(classify_process, models)


def classify_process(models):
    name = models[0]
    model = models[1]
    if (not (binary_g and name == 'XGBoost mult')) and (not (not binary_g and name == 'XGBoost binary')):
        scoring_func = 'f1_weighted'
        if groups_for_loocv_g is not None:
            scores = cross_val_score(model, X_g, y_g, cv=LeaveOneGroupOut(), groups=groups_for_loocv_g, scoring=scoring_func)
            y_pred = cross_val_predict(model, X_g, y_g, cv=LeaveOneGroupOut(), groups=groups_for_loocv_g)
        else:
            k_folds = 10
            scores = cross_val_score(model, X_g, y_g, cv=k_folds, scoring=scoring_func)
            y_pred = cross_val_predict(model, X_g, y_g, cv=k_folds)

        print('{}: {:1.2f} +/- {:1.2f}'.format(name, scores.mean(), scores.std()))

        if binary_g:
            label_names = set(y_g)
            for l in label_names:
                print("{}: F1 score for class {}: {:1.2f}".format(name, l, f1_score(y_g, y_pred, pos_label=l)))

        # confusion matrix
        labels_set = sorted(list(set(y_g)))
        conf_mat = confusion_matrix(y_g, y_pred)
        df_cm = pd.DataFrame(conf_mat, index=labels_set,
                             columns=labels_set)
        df_cm["sum"] = df_cm.sum(axis=1)
        df_cm = df_cm.loc[:, labels_set].div(df_cm["sum"], axis=0)
        df_cm = df_cm.round(2)
        fig = plt.figure(figsize=(10, 7))
        sn.heatmap(df_cm, annot=True, fmt='g')
        plt.show()
        if path_g is not None:
            output_figure(fig=fig, path=path_g, name=("confusion_matrix_"+name), format="png")
            if binary_g:
                output_figure(fig=auc_roc_cv(X = X_g, y = y_g, model = model), path=path_g, name=("auc_roc_"+name), format="png")