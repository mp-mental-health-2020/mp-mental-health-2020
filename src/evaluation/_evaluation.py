import time

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, plot_roc_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import fbeta_score
from sklearn.model_selection import StratifiedKFold


def evaluate(X_train, Y_train, X_test, Y_test, clf):
    start = time.clock()
    clf.fit(X_train, Y_train)
    end = time.clock()
    print('train_time: {f:.3f} sec.'.format(f=end - start))

    start = time.clock()
    prediction = clf.predict(X_test)
    end = time.clock()
    print('predict_time: {f:.3f} sec.'.format(f=end - start))
    f2_score = fbeta_score(Y_test, prediction, beta=2)
    precision, recall, f_score, support = precision_recall_fscore_support(Y_test, prediction)  # what kind of f_score?
    f_score_macro = f1_score(Y_test, prediction, average='macro')

    f_score_micro = f1_score(Y_test, prediction, average='micro')

    f_score_weighted = f1_score(Y_test, prediction, average='weighted')

    # f_score_samples = f1_score(Y_test, prediction, average='samples')

    f_score_none = f1_score(Y_test, prediction, average=None)
    f_score_binary = f1_score(Y_test, prediction, average='binary')

    accuracy = accuracy_score(Y_test, prediction)
    cohen_kappa = cohen_kappa_score(Y_test, prediction)
    matthews = matthews_corrcoef(Y_test, prediction)
    print('precision: ', precision)
    print('recall: ', recall)
    print('f_score: ', f_score)
    print('f2_score: ', f2_score)
    print('f1_score_macro: ', f_score_macro)
    print('f1_score_micro: ', f_score_micro)
    print('f1_score_weighted: ', f_score_weighted)
    print('f1_score_none: ', f_score_none)
    print('f1_score_binary: ', f_score_binary)
    print('support: ', support)
    print('accuracy: ', accuracy)
    print('cohen_kappa: ', cohen_kappa)
    print('matthews_corrcoef: ', matthews)
    fig = roc_curve_auc(X_train, X_test, Y_train, Y_test, clf)
    # fig = plt.figure()
    return fig, prediction


def roc_curve_auc(X_test, Y_test, clf):
    Y_score = clf.decision_function(X_test)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    fpr[0], tpr[0], _ = roc_curve(Y_test, Y_score)
    roc_auc[0] = auc(fpr[0], tpr[0])
    fig = display_roc_curve_auc(fpr, tpr, roc_auc)
    return fig


def display_roc_curve_auc(fpr, tpr, roc_auc, cls=0):
    fig = plt.figure()
    lw = 2
    plt.plot(fpr[cls], tpr[cls], color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[cls])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
    return fig

def auc_roc_cv(X, y, model):
    # Run classifier with cross-validation and plot ROC curves
    cv = StratifiedKFold(n_splits=6)
    classifier = model

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    fig, ax = plt.subplots()
    for i, (train, test) in enumerate(cv.split(X, y)):
        classifier.fit(X[train], y[train])
        viz = plot_roc_curve(classifier, X[test], y[test],
                             name='ROC fold {}'.format(i),
                             alpha=0.3, lw=1, ax=ax)
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
            label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color='b',
            label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label=r'$\pm$ 1 std. dev.')

    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
           title="Receiver operating characteristic")
    ax.legend(loc="lower right")
    return fig