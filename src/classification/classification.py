from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd

models = [('Logistic Regression', LogisticRegression(solver='liblinear', multi_class='ovr')), ('LDA', LinearDiscriminantAnalysis()), ('LinearSVC', LinearSVC()), ('CART', DecisionTreeClassifier()), ('NB', GaussianNB())]


def classify_all(X, y, label_ids=None):
    for name, model in models:
        scores = cross_val_score(model, X, y, cv=8)
        print('{}: {:1.2f} +/- {:1.2f}'.format(name, scores.mean(), scores.std()))

        if label_ids:
            # confusion matrix
            y_pred = cross_val_predict(model, X, y, cv=8)
            conf_mat = confusion_matrix(y, y_pred)
            #print(conf_mat)
            df_cm = pd.DataFrame(conf_mat, index = label_ids.keys(),
                      columns = label_ids.keys())
            df_cm["sum"] = df_cm.sum(axis=1)
            df_cm = df_cm.loc[:,label_ids.keys()].div(df_cm["sum"], axis=0)
            plt.figure(figsize = (10,7))
            sn.heatmap(df_cm, annot=True)
            plt.show()
