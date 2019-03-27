# -*- coding:utf-8 -*-
"""
分开对每一种图训练一个分类树
"""
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier, RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve, KFold
from sklearn.externals import joblib

file_name = '../csv_data/data_161_1.csv'
dataset = pd.read_csv(file_name)
dataset.head()

x = dataset.iloc[:,1:]
y = dataset['category']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.05, random_state=1)
print x_train.shape

kfold = StratifiedKFold(n_splits=10)

GBC_best = GradientBoostingClassifier(learning_rate=0.05, min_samples_leaf=3, n_estimators=700, max_features=4, max_depth=6)
GBC_best.fit(x_train, y_train)


LIST_err_idx = []
y_pred = GBC_best.predict(x_test)
correct_count = 0
cnt = 0
for t1, t2 in zip(y_test, y_pred):
    # print type(t1),type(t2)
    if t1 == t2:
        correct_count += 1
    else:
        LIST_err_idx.append(y_test.index[cnt])
        #print t1, t2
    cnt += 1
print 1.0 * correct_count / len(y_pred)
print LIST_err_idx
joblib.dump(GBC_best, './models/GBC_origin_161_1.model')
