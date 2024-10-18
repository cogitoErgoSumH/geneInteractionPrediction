import numpy as np
import pandas as pd
import math
from math import log
from scipy.stats import pearsonr, chisquare, f_oneway, kruskal, ansari, mood, levene, fligner, bartlett, mannwhitneyu
from scipy.spatial.distance import braycurtis, canberra, chebyshev, cityblock, correlation, cosine, dice, euclidean, \
    hamming, jaccard, kulsinski, rogerstanimoto, russellrao, sokalmichener, sokalsneath, sqeuclidean, yule
from sklearn.decomposition import FastICA
from sklearn.metrics.cluster import adjusted_mutual_info_score, adjusted_rand_score, completeness_score, \
    homogeneity_completeness_v_measure, homogeneity_score, mutual_info_score, normalized_mutual_info_score, \
    v_measure_score
from collections import defaultdict

from feature_engineering.classification_metrics import categorical_gini_coefficient
from feature_engineering.regression_metrics import gini_coefficient


def adjusted_mutual_info_score_(x, y):
    return adjusted_mutual_info_score(x, y, average_method='arithmetic')


def normalized_mutual_info_score_(x, y):
    return normalized_mutual_info_score(x, y, average_method='arithmetic')


def correlation_magnitude(x, y):
    return abs(pearsonr(x, y)[0])


def chi_square(x, y):
    return chisquare(x - min(x) + 1, y - min(y) + 1)


def categorical_categorical_homogeneity(x, y):
    grouped = defaultdict(list)
    [grouped[x_val].append(y_val) for x_val, y_val in zip(x, y)]
    homogeneity = [categorical_gini_coefficient(val) for val in grouped.values()]
    return max(homogeneity), np.mean(homogeneity), min(homogeneity), np.std(homogeneity)


def categorical_numerical_homogeneity(x, y):
    grouped = defaultdict(list)
    [grouped[x_val].append(y_val) for x_val, y_val in zip(x, y)]
    homogeneity = [gini_coefficient(val) for val in grouped.values()]
    return max(homogeneity), np.mean(homogeneity), min(homogeneity), np.std(homogeneity)


def anova(x, y):
    grouped = defaultdict(list)
    [grouped[x_val].append(y_val) for x_val, y_val in zip(x, y)]
    grouped_values = grouped.values()
    if len(grouped_values) < 2:
        return 0, 0, 0, 0
    f_oneway_res = list(f_oneway(*grouped_values))
    try:
        kruskal_res = list(kruskal(*grouped_values))
    except ValueError:
        kruskal_res = [0, 0]
    return f_oneway_res + kruskal_res

# 获取平均标准差和方差
def bucket_variance(x, y):
    grouped = defaultdict(list)
    [grouped[x_val].append(y_val) for x_val, y_val in zip(x, y)]
    grouped_values = grouped.values()
    weighted_avg_var = 0.0
    max_var = 0.0
    for bucket in grouped_values:
        var = np.std(bucket) ** 2
        max_var = max(var, max_var)
        weighted_avg_var += len(bucket) * var
    weighted_avg_var /= len(x)
    return max_var, weighted_avg_var

#  独立成分分析算法
def independent_component(x, y):
    clf = FastICA(random_state=1)
    try:
        clf.fit(x.reshape(-1, 1), y)
        comp = clf.components_[0][0]
        mm = clf.mixing_[0][0]
        sources = clf.fit_transform(x.reshape([-1, 1])).flatten()
        src_max = max(sources)
        src_min = min(sources)
        print("当前w值为"+str(clf._unmixing))
    except ValueError:
        comp = mm = src_max = src_min = 0.
    return [comp, mm, src_max, src_min]

# dice距离用于计算两个样本的相似度
def dice_(x, y):
    try:
        return dice(x, y)
    except (ZeroDivisionError, TypeError):
        return 0

#田本罗杰斯差异
def rogerstanimoto_(x, y):
    try:
        return rogerstanimoto(x, y)
    except ZeroDivisionError:
        return 0

#索卡尔米切纳差异
def sokalmichener_(x, y):
    try:
        return sokalmichener(x, y)
    except ZeroDivisionError:
        return 0

#索卡尔雪差异
def sokalsneath_(x, y):
    try:
        return sokalsneath(x, y)
    except ValueError:
        return 0

# 曼-惠特尼U检验
def mannwhitneyu_(x, y):
    try:
            return mannwhitneyu(x, y)
    except ValueError:
        return [0, 0]

#Yule差异
def yule_(x, y):
    try:
        return yule(x, y)
    except ZeroDivisionError:
        return 0
# 计算离散的kl散度
def kl_divergence(x,y):
    print("开始计算kl散度")
    print(x)
    print(y)
    x=list(x)
    y=list(y)
    x_new=[math.ceil(el) for el in x]
    y_new=[math.ceil(el) for el in y]
    x,y=x_new,y_new
    k_x = set(x)
    kl_divergence_result=0
    flag =True
    count=0 #count用来记录x中有但是y中没有
    for i in k_x:
        x_prob=x.count(i)/len(x)
        y_prob=y.count(i)/len(y)
        if y_prob!=0:
            flag=False
            kl_divergence_result += x_prob * log(x_prob / y_prob, 2)
        else:
            count+=1
    if flag:
        kl_divergence_result=-1
    else:
        kl_divergence_result=kl_divergence_result*(1+count/len(k_x))
    print("kl散度为"+str(kl_divergence_result))
    return (round(kl_divergence_result,2),)



ALL_BINARY_FEATURES = (
    chi_square,
    pearsonr,
    correlation_magnitude,
    braycurtis,
    canberra,
    chebyshev,
    cityblock,
    correlation,
    cosine,
    euclidean,
    hamming,
    sqeuclidean,
    ansari,
    mood,
    levene,
    fligner,
    bartlett,
    mannwhitneyu_,
)

NN_BINARY_FEATURES = (
    independent_component,
    kl_divergence
)

CN_BINARY_FEATURES = (
    categorical_numerical_homogeneity,
    bucket_variance,
    anova,
)

CC_BINARY_FEATURES = (
    categorical_categorical_homogeneity,
    anova,
    dice_,
    jaccard,
    kulsinski,
    rogerstanimoto_,
    russellrao,
    sokalmichener_,
    sokalsneath_,
    yule_,
    adjusted_mutual_info_score_,
    adjusted_rand_score,
    completeness_score,
    homogeneity_completeness_v_measure,
    homogeneity_score,
    mutual_info_score,
    normalized_mutual_info_score_,
    v_measure_score,
    #以下是新加
    condition_entropy

)

NC_BINARY_FEATURES = (
)
# 此处令所有的二元特征为0，只保存NN的
ALL_BINARY_FEATURES=NN_BINARY_FEATURES=CN_BINARY_FEATURES=CC_BINARY_FEATURES=()
CC_BINARY_FEATURES=(kl_divergence,condition_entropy)

def binary_feature_wrapper(f):
    def inner_func(x, y):
        result = f(x, y)
        try:
            list_result = list(result)
        except TypeError:
            list_result = [result]
        feature_names = ["{}_{}".format(f.__name__, i) for i in range(len(list_result))]
        features = list(zip(feature_names, list_result))
        return features

    return inner_func


BINARY_FEATURES = []

for f in ALL_BINARY_FEATURES:
    for desired_type in ["NN", "NC", "CN", "CC"]:
        name = "{}_{}".format(desired_type, f.__name__)
        BINARY_FEATURES.append((name, binary_feature_wrapper, f))

for f in NN_BINARY_FEATURES:
    desired_type = "NN"
    name = "{}_{}".format(desired_type, f.__name__)
    BINARY_FEATURES.append((name, binary_feature_wrapper, f))

for f in CN_BINARY_FEATURES:
    desired_type = "CN"
    name = "{}_{}".format(desired_type, f.__name__)
    BINARY_FEATURES.append((name, binary_feature_wrapper, f))

for f in CC_BINARY_FEATURES:
    desired_type = "CC"
    name = "{}_{}".format(desired_type, f.__name__)
    BINARY_FEATURES.append((name, binary_feature_wrapper, f))

for f in NC_BINARY_FEATURES:
    desired_type = "NC"
    name = "{}_{}".format(desired_type, f.__name__)
    BINARY_FEATURES.append((name, binary_feature_wrapper, f))
