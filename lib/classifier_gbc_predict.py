import pandas as pd
import numpy as np
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

def predict(data):
    gbc = joblib.load('./models/GBC_dp.model')
    pred_prob = gbc.predict_proba(data)
    pred = np.argmax(pred_prob, axis=1)
    return gbc.predict(data),pred,pred_prob
    
data = [[0.153061,0.60063,0.721939,0.146793,1.4,0,1,1,0,0,0,3,1.372807688415404]]
print predict(data)