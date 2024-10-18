import tqdm
import pandas as pd

import keras
import lightgbm as lgb
import numpy as np
from sklearn.preprocessing import StandardScaler
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

print(os.path.abspath('.'))

num_classes = 3
epochs = 5
model_name = 'keras_cnn_trained_model_shallow.h5'
number=7
with open('./data/x_train{}.pkl'.format(number), 'rb') as f:
    x_train = pickle.load(f)
with open('./data/y_train{}.pkl'.format(number), 'rb') as f:
    y_train = pickle.load(f)
with open('./data/x_test{}.pkl'.format(number), 'rb') as f:
    x_test = pickle.load(f)
with open('./data/y_test{}.pkl'.format(number), 'rb') as f:
    y_test = pickle.load(f)

count = y_train.loc[:, 'label'].value_counts()
print(count)
print(x_train.columns)
x_train=x_train[['NN_normalized_entropy_A->B_normalized_entropy_0',
       'NN_normalized_entropy_B->A_normalized_entropy_0',
       'NN_normalized_entropy_normalized_entropy_0_difference',
       'NN_skew_A->B_skew_0', 'NN_skew_B->A_skew_0',
       'NN_skew_skew_0_difference', 'NN_kurtosis_A->B_kurtosis_0',
       'NN_kurtosis_B->A_kurtosis_0', 'NN_kurtosis_kurtosis_0_difference',
       'NN_shapiro_A->B_shapiro_0',
       'NN_shapiro_B->A_shapiro_0',
       'NN_shapiro_shapiro_0_difference',
       'NN_chi_square_A->B_chi_square_0',
       'NN_chi_square_B->A_chi_square_0',
       'NN_chi_square_chi_square_0_difference',
       'NN_chi_square_chi_square_1_difference',
        'NN_pearsonr_A->B_pearsonr_0',
       'NN_pearsonr_B->A_pearsonr_0',
       'NN_pearsonr_pearsonr_0_difference',
       'CC_mutual_info_score_A->B_mutual_info_score_0',
       'CC_mutual_info_score_B->A_mutual_info_score_0',
       'CC_mutual_info_score_mutual_info_score_0_difference',
       'CC_normalized_mutual_info_score__A->B_normalized_mutual_info_score__0',
       'CC_normalized_mutual_info_score__B->A_normalized_mutual_info_score__0',
       'CC_normalized_mutual_info_score__normalized_mutual_info_score__0_difference']]
x_test=x_test[['NN_normalized_entropy_A->B_normalized_entropy_0',
       'NN_normalized_entropy_B->A_normalized_entropy_0',
       'NN_normalized_entropy_normalized_entropy_0_difference',
       'NN_skew_A->B_skew_0', 'NN_skew_B->A_skew_0',
       'NN_skew_skew_0_difference', 'NN_kurtosis_A->B_kurtosis_0',
       'NN_kurtosis_B->A_kurtosis_0', 'NN_kurtosis_kurtosis_0_difference',
       'NN_shapiro_A->B_shapiro_0',
       'NN_shapiro_B->A_shapiro_0',
       'NN_shapiro_shapiro_0_difference',
       'NN_chi_square_A->B_chi_square_0',
       'NN_chi_square_B->A_chi_square_0',
       'NN_chi_square_chi_square_0_difference',
       'NN_chi_square_chi_square_1_difference',
        'NN_pearsonr_A->B_pearsonr_0',
       'NN_pearsonr_B->A_pearsonr_0',
       'NN_pearsonr_pearsonr_0_difference',
       'CC_mutual_info_score_A->B_mutual_info_score_0',
       'CC_mutual_info_score_B->A_mutual_info_score_0',
       'CC_mutual_info_score_mutual_info_score_0_difference',
       'CC_normalized_mutual_info_score__A->B_normalized_mutual_info_score__0',
       'CC_normalized_mutual_info_score__B->A_normalized_mutual_info_score__0',
       'CC_normalized_mutual_info_score__normalized_mutual_info_score__0_difference']]
# x_train=x_train[['NN_normalized_entropy_A->B_normalized_entropy_0',
#        'NN_normalized_entropy_B->A_normalized_entropy_0',
#        'NN_normalized_entropy_normalized_entropy_0_difference',
#        'NN_skew_A->B_skew_0', 'NN_skew_B->A_skew_0',
#        'NN_skew_skew_0_difference', 'NN_kurtosis_A->B_kurtosis_0',
#        'NN_kurtosis_B->A_kurtosis_0', 'NN_kurtosis_kurtosis_0_difference',
#        'NN_shapiro_A->B_shapiro_0',
#        'NN_shapiro_B->A_shapiro_0',
#        'NN_shapiro_shapiro_0_difference',
#        'CC_num_unique_A->B_num_unique_0', 'CC_num_unique_B->A_num_unique_0',
#        'CC_num_unique_num_unique_0_difference']]
# x_test=x_test[['NN_normalized_entropy_A->B_normalized_entropy_0',
#        'NN_normalized_entropy_B->A_normalized_entropy_0',
#        'NN_normalized_entropy_normalized_entropy_0_difference',
#        'NN_skew_A->B_skew_0', 'NN_skew_B->A_skew_0',
#        'NN_skew_skew_0_difference', 'NN_kurtosis_A->B_kurtosis_0',
#        'NN_kurtosis_B->A_kurtosis_0', 'NN_kurtosis_kurtosis_0_difference',
#        'NN_shapiro_A->B_shapiro_0',
#        'NN_shapiro_B->A_shapiro_0',
#        'NN_shapiro_shapiro_0_difference',
#        'CC_num_unique_A->B_num_unique_0', 'CC_num_unique_B->A_num_unique_0',
#        'CC_num_unique_num_unique_0_difference']]
print(x_train.columns)
print(x_test.shape)
x_train_numpy = np.array(x_train)
x_test_numpy = np.array(x_test)
y_train_numpy = y_train['label'].values
y_test_numpy = y_test['label'].values

x_train_numpy = x_train_numpy[y_train_numpy != 3]
x_test_numpy = x_test_numpy[y_test_numpy != 3]
y_train_numpy = y_train_numpy[y_train_numpy != 3]
y_test_numpy = y_test_numpy[y_test_numpy != 3]
print(x_train_numpy.shape)
print(x_test_numpy.shape)
scaler = StandardScaler()
scaler.fit(x_train_numpy)
x_train_numpy = scaler.transform(x_train_numpy)
x_test_numpy = scaler.transform(x_test_numpy)

################################
gbc_params = {
    'loss': 'deviance',
    'learning_rate': 0.1,
    # 'n_estimators': 500,
    'n_estimators': 50,#黄加的
    'subsample': 1.0,
    'min_samples_split': 8,
    'min_samples_leaf': 1,
    'max_depth': 9,
    'init': None,
    'random_state': 1,
    'max_features': None,
    'verbose': 0
}

# clf = GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=500, min_samples_split=2)
# clf.fit(x_train_numpy, y_train_numpy)
# y_pred = clf.predict_proba(x_test_numpy)
# y_pred2 = y_pred.argmax(axis=1)
# print(accuracy_score(y_test_numpy, y_pred2))
# aaa = confusion_matrix(y_test_numpy, y_pred2)
# bbb = (aaa.T / aaa.sum(axis=1)).T
# print(pd.DataFrame(bbb))
#################################
print("randomForest start")
# clf = RandomForestClassifier(n_estimators=500, criterion='gini', max_depth=10, random_state=0, min_samples_leaf=1)
#黄加的
# clf=RandomForestClassifier(criterion='entropy', max_features= 'auto', min_samples_leaf= 1, min_samples_split=3, n_estimators= 80,n_jobs=-1)
clf =RandomForestClassifier()
clf.fit(x_train_numpy, y_train_numpy)
y_pred = clf.predict_proba(x_test_numpy)
y_pred2 = y_pred.argmax(axis=1)
print(accuracy_score(y_test_numpy, y_pred2))
aaa = confusion_matrix(y_test_numpy, y_pred2)
bbb = (aaa.T / aaa.sum(axis=1)).T
print(pd.DataFrame(bbb))
print("randomForest end")
#################################
# validation_num = int(x_train_numpy.shape[0]/3)
# train_data = lgb.Dataset(x_train_numpy[validation_num:], label=y_train_numpy[validation_num:])
# validation_data = lgb.Dataset(x_train_numpy[:validation_num], label=y_train_numpy[:validation_num])
train_data = lgb.Dataset(x_train_numpy, label=y_train_numpy)
validation_data = lgb.Dataset(x_test_numpy, label=y_test_numpy)

params = {
    # 'num_leaves': 31,
    # 'min_data_in_leaf': 30,
    'objective': 'multiclass',
    'num_class': num_classes,
    'max_depth': 10,
    'learning_rate': 0.1,
    # 'min_sum_hessian_in_leaf': 6,
    'boosting': 'gbdt',
    # 'feature_fraction': 0.9,
    # 'bagging_freq': 1,
    # 'bagging_fraction': 0.8,
    # 'bagging_seed': 11,
    # 'lambda_l1': 0.1,
    # 'lambda_l2': 0.2,
    'verbosity': -1,
    'metric': 'multi_logloss',
    # 'metric': 'multi_error',
    'random_state': 0,
}
clf = lgb.train(params, train_data, valid_sets=[train_data, validation_data],
                num_boost_round=100, early_stopping_rounds=5)

y_pred = clf.predict(x_test_numpy)
y_pred2 = y_pred.argmax(axis=1)
print(accuracy_score(y_test_numpy, y_pred2))
aaa = confusion_matrix(y_test_numpy, y_pred2)
bbb = (aaa.T / aaa.sum(axis=1)).T
print(pd.DataFrame(bbb))

f = pd.DataFrame({'feature_name': x_train.columns, 'feature_importance': clf.feature_importance()})
f = f.sort_values(by='feature_importance', ascending=False)
print(f)

x_gene, y_gene, xx, yy = sc_pairs[0]
aa = np.histogram2d(xx, yy, bins=32)[0].T
new_aa = np.flip(aa, axis=0)
plt.imshow(np.log1p(new_aa))
plt.show()

y_pred = clf.predict(x_train_numpy)
y_pred2 = y_pred.argmax(axis=1)
print(accuracy_score(y_train_numpy, y_pred2))
aaa = confusion_matrix(y_train_numpy, y_pred2)
bbb = (aaa.T / aaa.sum(axis=1)).T
print(pd.DataFrame(bbb))

yy_test_numpy = keras.utils.to_categorical(y_test_numpy, 3)
plt.figure(figsize=(20, 6))
for i in range(3):
    y_test_x = [j[i] for j in yy_test_numpy]
    y_predict_x = [j[i] for j in y_pred]
    fpr, tpr, thresholds = metrics.roc_curve(y_test_x, y_predict_x, pos_label=1)
    plt.subplot(1, 3, i + 1)
    plt.plot(fpr, tpr)
    plt.grid()
    plt.plot([0, 1], [0, 1])
    plt.xlabel('FP')
    plt.ylabel('TP')
    plt.ylim([0, 1])
    plt.xlim([0, 1])
    auc = np.trapz(tpr, fpr)
    print('AUC:', auc)
    plt.title('label' + str(i) + ', AUC:' + str(auc))
plt.show()

fig = plt.figure(figsize=(10, 10))
plt.plot([0, 1], [0, 1])
plt.ylim([0, 1])
plt.xlim([0, 1])
plt.xlabel('FP')
plt.ylabel('TP')
# plt.grid()
AUC_set = []
y_testy = yy_test_numpy
y_predicty = y_pred
tprs = []
mean_fpr = np.linspace(0, 1, 100)
# s = open(save_dir + '/divided_AUCs1vs2.txt', 'w')
for jj in range(len(count_set) - 1):  # len(count_set)-1):
    if count_set[jj] < count_set[jj + 1]:
        y_test_aaa = y_testy[count_set[jj]:count_set[jj + 1]]
        y_predict_aaa = y_predicty[count_set[jj]:count_set[jj + 1]]
        y_predict1 = []
        y_test1 = []
        x = 2
        for i in range(int(len(y_predict_aaa) / 3)):
            y_predict1.append(y_predict_aaa[3 * i][x] - y_predict_aaa[3 * i + 1][x])
            y_predict1.append(-y_predict_aaa[3 * i][x] + y_predict_aaa[3 * i + 1][x])
            y_test1.append(y_test_aaa[3 * i][x])
            y_test1.append(y_test_aaa[3 * i + 1][x])
        fpr, tpr, thresholds = metrics.roc_curve(y_test1, y_predict1, pos_label=1)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        plt.plot(fpr, tpr, color='0.5', lw=0.1)
        auc = np.trapz(tpr, fpr)
        # s.write(str(jj) + '\t' + str(count_set[jj]) + '\t' + str(count_set[jj + 1]) + '\t' + str(auc) + '\n')
        AUC_set.append(auc)
median_tpr = np.median(tprs, axis=0)
mean_tpr = np.mean(tprs, axis=0)
median_tpr[-1] = 1.0
mean_tpr[-1] = 1.0
per_tpr = np.percentile(tprs, [25, 50, 75], axis=0)
median_auc = np.trapz(median_tpr, mean_fpr)
mean_auc = np.trapz(mean_tpr, mean_fpr)
plt.plot(mean_fpr, median_tpr, 'k', lw=3, label='median ROC')
plt.title(f'{str(median_auc)}({str(mean_auc)})')
plt.fill_between(mean_fpr, per_tpr[0, :], per_tpr[2, :], color='g', alpha=.2, label='Quartile')
plt.legend(loc='lower right')
plt.show()

f = pd.DataFrame({'feature_name': x_train.columns, 'feature_importance': clf.feature_importance()})
f.sort_values(by='feature_importance', ascending=False).to_clipboard()
lgb.create_tree_digraph(clf, tree_index=1)
lgb.plot_tree(clf, tree_index=0, figsize=(100, 50))
plt.savefig('test.png')

##############