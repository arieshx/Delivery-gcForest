#import matplotlib
#matplotlib.use('agg')
#%matplotlib inline
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import os, shutil, json
from collections import Counter
from gcforest.gcforest import GCForest

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, ExtraTreesRegressor, RandomForestRegressor
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier, RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve, KFold
from sklearn.externals import joblib


file_name = '../csv_data/result3.csv'
dataset = pd.read_csv(file_name)
dataset.head()

x = dataset.iloc[:,1:]
y = dataset['category']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.03, random_state=1)  
print x_train.shape

kfold = StratifiedKFold(n_splits=10)


#random_state = 2
#simple_gbc = GradientBoostingClassifier(random_state=random_state)
GBC_best = GradientBoostingClassifier(min_samples_leaf=100,n_estimators=300,max_features=3,max_depth=8)
GBC_best.fit(x_train, y_train)


LIST_err_idx = []
y_pred = GBC_best.predict(x_test)
correct_count = 0
cnt = 0
for t1, t2 in zip(y_test, y_pred):
    print type(t1),type(t2)
    if t1==t2:
        correct_count += 1
    else:
        LIST_err_idx.append(y_test.index[cnt])
        print t1, t2
    cnt += 1
print 1.0 * correct_count / len(y_pred)
print LIST_err_idx
joblib.dump(GBC_best, './models/GBC_origin.model')
'''
GBC = GradientBoostingClassifier()
gb_param_grid = {'loss' : ["deviance"],
                  'n_estimators' : [100,200,300],
                 'learning_rate': [0.1, 0.05, 0.01],
                  'max_depth': [4, 8],
                  'min_samples_leaf': [100,150],
                  'max_features': [1,3,10] 
                  }

gsGBC = GridSearchCV(GBC,param_grid = gb_param_grid, n_jobs= -1, verbose = 1)

gsGBC.fit(x_train,y_train)

GBC_best = gsGBC.best_estimator_
print gsGBC.best_score_
print gsGBC.best_params_

y_pred = GBC_best.predict(x_test)
LIST_err_idx1 = []
correct_count = 0
cnt = 0
for t1, t2 in zip(y_test, y_pred):
    if abs(t1 - t2) == 0:
        correct_count += 1
    else:
        LIST_err_idx1.append(y_test.index[cnt])
        print t1, t2
    cnt += 1
print 1.0 * correct_count / len(y_pred)
print LIST_err_idx1
'''

'''
g = sns.kdeplot(y_pred, color="Blue", label='prediction',shade= True)
g = sns.kdeplot(y_test, color="Red", label='test', shade = True)
g.set_xlabel("Category")
g.set_ylabel("Frequency")
g = g.legend()
def calc_score_consistency(test_pred,test_origin, max_score):
    num_correct = 0
    for score_test, score_test_org in zip(test_pred, test_origin):
        if score_test == score_test_org:
            num_correct += 1
    return 1.0 * num_correct / len(test_pred)


def calc_correl(test_pred, test_origin):
    test_prs, _ = pearsonr(test_pred, test_origin)
    test_spr, _ = spearmanr(test_pred, test_origin)
    test_tau, _ = kendalltau(test_pred, test_origin)
    return test_spr, test_tau


def calc_qwk(test_pred, test_origin, self_low, self_high):
    test_pred_int = np.rint(test_pred).astype('int32')
    test_origin_int = np.rint(test_origin).astype('int32')
    test_qwk = qwk(test_origin_int, test_pred_int, self_low, self_high)
    test_lwk = lwk(test_origin_int, test_pred_int, self_low, self_high)
    return test_qwk,test_lwk"


calc_score_consistency(y_pred, y_valid, 5)
calc_correl(y_pred, y_valid)
calc_qwk(y_pred, y_valid, 0, 5)
fig, ax = plt.subplots(figsize=(12,12))
g = sns.heatmap(dataset[dataset.columns.tolist()[0:]].corr(),annot=True, fmt = ".2f", cmap = "coolwarm", ax=ax)

with open('./111/examples/demo_mnist-gc.json', 'r') as f:
    config = json.load(f)
gc = GCForest(config)
print x_train.shape
x_train = np.asarray(x_train)
new_x_train = x_train[ np.newaxis, np.newaxis,:, :]
print new_x_train.shape
X_train_enc = gc.fit_transform(new_x_train, np.asarray(y_train))
y_pred = gc.predict(x_valid)'''
