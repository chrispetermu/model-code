import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
### Import data
# Always good to set a seed for reproducibility
SEED =8899
np.random.seed(SEED)
#D://sigma_0/data4.csv
df=pd.read_csv('D://sigma_1/data5.csv', index_col=0)
print(df)

### Training and test set


def get_train_test(test_size=0.2):
    """Split Data into train and test sets."""
    y=df.outcomes
    x=df.drop(["outcomes","int"],axis=1)

    return train_test_split(x, y, test_size=test_size, random_state=SEED)

xtrain, xtest, ytrain, ytest = get_train_test()



df.outcomes.value_counts(normalize=True).plot(
    kind="bar", title="Share of outcome donations")
plt.show()
# A look at the data

from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.kernel_approximation import Nystroem
from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import make_pipeline

def get_models():
    """Generate a library of base learners."""
    nb = GaussianNB()
    svc = SVC(C=0.1, probability=True,class_weight='balanced')
    knn = KNeighborsClassifier(n_neighbors=50)
    lr = LogisticRegression(C=100, random_state=SEED,class_weight='balanced')
    nn = MLPClassifier((25, 5), early_stopping=False,random_state=SEED)
    gb = GradientBoostingClassifier(n_estimators=100,random_state=SEED)
    rf = RandomForestClassifier(n_estimators=200,class_weight='balanced', random_state=SEED)

    models = {'svm': svc,
              'knn': knn,
              'naive bayes': nb,

              'random forest': rf,
         
              'logistic': lr,
              }

    return models

def train_predict(model_list):
    """Fit models in list on training set and return preds"""
    P = np.zeros((ytest.shape[0], len(model_list)))
    P = pd.DataFrame(P)
    P_r = np.zeros((ytest.shape[0], len(model_list)))
    P_r = pd.DataFrame(P_r)

    print("Fitting models.")
    cols = list()
    for i, (name, m) in enumerate(models.items()):
        print("%s..." % name, end=" ", flush=False)
        m.fit(xtrain, ytrain)
        P.iloc[:, i] = m.predict(xtest)
        P_r.iloc[:, i] = m.predict_proba(xtest)[:, 1]
        cols.append(name)
        print("done")

    P.columns = cols
    P_r.columns = cols
    print("Done.\n")
    return P,P_r


def score_models_roc(Pr, y):
    """Score model in prediction DF"""
    print("Scoring models roc.")
    for m in Pr.columns:
        score = roc_auc_score(y, Pr.loc[:, m])
        print("%-26s: %.3f" % (m, score))
    print("Done.\n")

def score_models_recall(P, y):
    """Score model in prediction DF"""
    print("Scoring models sensitive.")
    for m in P.columns:
        score = metrics.recall_score(y, P.loc[:, m].astype(int))
        print("%-26s: %.3f" % (m, score))
    print("Done.\n")

models = get_models()
P,P_r = train_predict(models)
score_models_roc(P_r, ytest)
score_models_recall(P,ytest)

print("Ensemble ROC-AUC score: %.3f" % roc_auc_score(ytest, P.mean(axis=1)))

from sklearn.metrics import roc_curve

def plot_roc_curve_nonsemble(ytest,p_base,labels):
    plt.figure(figsize=(10, 8))
    plt.plot([0, 1], [0, 1], 'k--')

    cm = [plt.cm.rainbow(i)
          for i in np.linspace(0, 1.0, p_base.shape[1] + 1)]

    for i in range(p_base.shape[1]):
        p = p_base[:, i]
        fpr, tpr, _ = roc_curve(ytest, p)
        plt.plot(fpr, tpr, label=labels[i], c=cm[i + 1])

    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(frameon=False)
    plt.show()

plot_roc_curve_nonsemble(ytest, P_r.values, list(P_r.columns))

def plot_roc_curve(ytest, P_base_learners, P_ensemble, labels, ens_label):
    """Plot the roc curve for base learners and ensemble."""
    plt.figure(figsize=(10, 8))
    plt.plot([0, 1], [0, 1], 'k--')

    cm = [plt.cm.rainbow(i)
          for i in np.linspace(0, 1.0, P_base_learners.shape[1] + 1)]

    for i in range(P_base_learners.shape[1]):
        p = P_base_learners[:, i]
        fpr, tpr, _ = roc_curve(ytest, p)
        plt.plot(fpr, tpr, label=labels[i], c=cm[i + 1])

    fpr, tpr, _ = roc_curve(ytest, P_ensemble)
    plt.plot(fpr, tpr, label=ens_label, c=cm[0])

    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(frameon=False)
    plt.show()


plot_roc_curve(ytest, P_r.values, P_r.mean(axis=1), list(P_r.columns), "ensemble")





# base_learners = get_models()
# meta_learner = GradientBoostingClassifier(
#     n_estimators=1000,
#     loss="exponential",
#     max_features=4,
#     max_depth=3,
#     subsample=0.5,
#     learning_rate=0.005,
#     random_state=SEED
# )
#
# xtrain_base, xpred_base, ytrain_base, ypred_base = train_test_split(
#     xtrain, ytrain, test_size=0.5, random_state=SEED)
#
# def train_base_learners(base_learners, inp, out, verbose=True):
#     """Train all base learners in the library."""
#     if verbose: print("Fitting models.")
#     for i, (name, m) in enumerate(base_learners.items()):
#         if verbose: print("%s..." % name, end=" ", flush=False)
#         m.fit(inp, out)
#         if verbose: print("done")
# train_base_learners(base_learners, xtrain_base, ytrain_base)
#
# def predict_base_learners(pred_base_learners, inp, verbose=True):
#     """Generate a prediction matrix."""
#     P = np.zeros((inp.shape[0], len(pred_base_learners)))
#
#     if verbose: print("Generating base learner predictions.")
#     for i, (name, m) in enumerate(pred_base_learners.items()):
#         if verbose: print("%s..." % name, end=" ", flush=False)
#         p = m.predict_proba(inp)
#         # With two classes, need only predictions for one class
#         P[:, i] = p[:, 1]
#         if verbose: print("done")
#
#     return P
# P_base = predict_base_learners(base_learners, xpred_base)
# meta_learner.fit(P_base, ypred_base)
#
# P_base = predict_base_learners(base_learners, xpred_base)
# def ensemble_predict(base_learners, meta_learner, inp, verbose=True):
#     """Generate predictions from the ensemble."""
#     P_pred = predict_base_learners(base_learners, inp, verbose=verbose)
#     return P_pred, meta_learner.predict(P_pred)
# P_pred, p = ensemble_predict(base_learners, meta_learner, xtest)
# print("\nEnsemble recall score: %.3f" % metrics.recall_score(ytest, p))








def get_train_test_external(test_size=0.9):
    """Split Data into train and test sets."""
    y=df.outcomes
    x=df.drop(["outcomes","int"],axis=1)

    return train_test_split(x, y, test_size=test_size, random_state=SEED)






df =pd.read_csv('D://sigma_0/data6.csv', index_col=0)

xxtrain, xxtest, yytrain, yytest = get_train_test_external()

pp=np.zeros((yytest.shape[0],6))

lr1  = LogisticRegression(C=0.01, random_state=SEED,class_weight='balanced')

lr1.fit(xtrain,ytrain)
pp[:,0]=lr1.predict_proba(xxtest)[:,1]


scoref = roc_auc_score(yytest, pp[:,0])
print(scoref)

df=pd.read_csv('D://sigma_0/data2.csv', index_col=0)


### Training and test set
xtrain, xtest, ytrain, ytest = get_train_test()

lr2  = LogisticRegression(C=0.01, random_state=SEED,class_weight='balanced')

lr2.fit(xtrain,ytrain)
pp[:,1]=lr2.predict_proba(xxtest)[:,1]


scoref = roc_auc_score(yytest, pp[:,1])
print(scoref)

df=pd.read_csv('D://sigma_0/data3.csv', index_col=0)


### Training and test set
xtrain, xtest, ytrain, ytest = get_train_test()

lr3  = LogisticRegression(C=0.01, random_state=SEED,class_weight='balanced')

lr3.fit(xtrain,ytrain)
pp[:,2]=lr3.predict_proba(xxtest)[:,1]


scoref = roc_auc_score(yytest, pp[:,2])
print(scoref)

df=pd.read_csv('D://sigma_0/data4.csv', index_col=0)


### Training and test set
xtrain, xtest, ytrain, ytest = get_train_test()

lr4  = LogisticRegression(C=0.01, random_state=SEED,class_weight='balanced')

lr4.fit(xtrain,ytrain)
pp[:,3]=lr4.predict_proba(xxtest)[:,1]


scoref = roc_auc_score(yytest, pp[:,3])
print(scoref)

df=pd.read_csv('D://sigma_0/data5.csv', index_col=0)


### Training and test set
xtrain, xtest, ytrain, ytest = get_train_test()

lr5 = LogisticRegression(C=0.01, random_state=SEED,class_weight='balanced')

lr5.fit(xtrain,ytrain)
pp[:,4]=lr5.predict_proba(xxtest)[:,1]


scoref = roc_auc_score(yytest, pp[:,4])
print(scoref)

scoref=roc_auc_score(yytest,pp.mean(axis=1))
print(scoref)

df = df.reset_index()

df=pd.read_csv('D://sigma_1/FF.csv', index_col=0)
XFtrain,XFtest,YFtrain,YFtest =get_train_test()

lr6 = LogisticRegression(C=0.1, random_state=SEED,class_weight='balanced')

lr6.fit(XFtrain,YFtrain)
FF=lr6.predict_proba(XFtest)[:,1]


scoref = roc_auc_score(YFtest, FF)
print(scoref)
