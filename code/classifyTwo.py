
import pandas as pd

import keras
import lightgbm as lgb
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
# from scipy import interp
from numpy import interp
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
import pickle
import matplotlib
import matplotlib.pyplot as plt
from sklearn import metrics
import os

num_classes = 3
epochs = 5
model_name = 'keras_cnn_trained_model_shallow.h5'
number=5

def load_data():
    with open('./data/x_train{}.pkl'.format(number), 'rb') as f:
        x_train = pickle.load(f)
    with open('./data/y_train{}.pkl'.format(number), 'rb') as f:
        y_train = pickle.load(f)
    with open('./data/x_test{}.pkl'.format(number), 'rb') as f:
        x_test = pickle.load(f)
    with open('./data/y_test{}.pkl'.format(number), 'rb') as f:
        y_test = pickle.load(f)
    x_train = x_train[['NN_normalized_entropy_A->B_normalized_entropy_0',
                       'NN_normalized_entropy_B->A_normalized_entropy_0',
                       'NN_normalized_entropy_normalized_entropy_0_difference',
                       'NN_skew_A->B_skew_0', 'NN_skew_B->A_skew_0',
                       'NN_skew_skew_0_difference', 'NN_kurtosis_A->B_kurtosis_0',
                       'NN_kurtosis_B->A_kurtosis_0', 'NN_kurtosis_kurtosis_0_difference',
                       'NN_shapiro_A->B_shapiro_0',
                       'NN_shapiro_B->A_shapiro_0',
                       'NN_shapiro_shapiro_0_difference',
                       'CC_num_unique_A->B_num_unique_0', 'CC_num_unique_B->A_num_unique_0',
                       'CC_num_unique_num_unique_0_difference']]
    x_test = x_test[['NN_normalized_entropy_A->B_normalized_entropy_0',
                     'NN_normalized_entropy_B->A_normalized_entropy_0',
                     'NN_normalized_entropy_normalized_entropy_0_difference',
                     'NN_skew_A->B_skew_0', 'NN_skew_B->A_skew_0',
                     'NN_skew_skew_0_difference', 'NN_kurtosis_A->B_kurtosis_0',
                     'NN_kurtosis_B->A_kurtosis_0', 'NN_kurtosis_kurtosis_0_difference',
                     'NN_shapiro_A->B_shapiro_0',
                     'NN_shapiro_B->A_shapiro_0',
                     'NN_shapiro_shapiro_0_difference',
                     'CC_num_unique_A->B_num_unique_0', 'CC_num_unique_B->A_num_unique_0',
                     'CC_num_unique_num_unique_0_difference']]
    print(x_train.columns)
    print(x_test.columns)
    x_train_numpy = np.array(x_train)
    x_test_numpy = np.array(x_test)
    y_train_numpy = y_train['label'].values
    y_test_numpy = y_test['label'].values

    x_train_numpy = x_train_numpy[y_train_numpy != 3]
    x_test_numpy = x_test_numpy[y_test_numpy != 3]
    y_train_numpy = y_train_numpy[y_train_numpy != 3]
    y_test_numpy = y_test_numpy[y_test_numpy != 3]

    scaler = StandardScaler()
    scaler.fit(x_train_numpy)
    x_train_numpy = scaler.transform(x_train_numpy)
    x_test_numpy = scaler.transform(x_test_numpy)
    return x_train_numpy,x_test_numpy,y_train_numpy,y_test_numpy
def load_all_data():
    with open('./data/x_train{}.pkl'.format(number), 'rb') as f:
        x_train = pickle.load(f)
    with open('./data/y_train{}.pkl'.format(number), 'rb') as f:
        y_train = pickle.load(f)

    x_train = x_train[['NN_normalized_entropy_A->B_normalized_entropy_0',
                       'NN_normalized_entropy_B->A_normalized_entropy_0',
                       'NN_normalized_entropy_normalized_entropy_0_difference',
                       'NN_skew_A->B_skew_0', 'NN_skew_B->A_skew_0',
                       'NN_skew_skew_0_difference', 'NN_kurtosis_A->B_kurtosis_0',
                       'NN_kurtosis_B->A_kurtosis_0', 'NN_kurtosis_kurtosis_0_difference',
                       'NN_shapiro_A->B_shapiro_0',
                       'NN_shapiro_B->A_shapiro_0',
                       'NN_shapiro_shapiro_0_difference',
                       'CC_num_unique_A->B_num_unique_0', 'CC_num_unique_B->A_num_unique_0',
                       'CC_num_unique_num_unique_0_difference']]
    return x_train,y_train
def classify():
    x_train,y_train=load_all_data()
    y_train = y_train['label'].values
    x_train=x_train[y_train!= 3]
    print(y_train.shape)
    print(x_train.shape)

    y_train= y_train[y_train != 3]
    print(y_train.shape)
    print(x_train.shape)
    seed = 5
    xtrain, xtest, ytrain, ytest = train_test_split(x_train, y_train, test_size=0.3, random_state=seed)
    param_test3 = {'max_depth': range(3, 30, 2)}
    gsearch3 = GridSearchCV(estimator=RandomForestClassifier(n_estimators=300,
                                                             min_samples_split=60,
                                                             min_samples_leaf=10,
                                                             random_state=10),
                            param_grid=param_test3,
                            # scoring='roc_auc',
                            cv=5)

    gsearch3.fit(xtrain, ytrain)
    print(gsearch3.best_params_, gsearch3.best_score_)

classify()