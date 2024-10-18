#!/usr/bin/env python
# coding=utf-8

import pandas as pd
import pickle
from feature_engineering.random_functions import preprocess

import os
if __name__ == '__main__':
    missing_data = []
    for i in range(3000):
        if not os.path.exists(r'./data/features2/sc_features_{i}.pkl'):
            missing_data.append(i)

    TEST_NUM = 3057
    NUM = 3057
    file_error = []
    # for i in range(0,NUM):
    for i in [1400]:

        try:
            # j = j + 1
            # path1='./data/samples/bulk_{}.pkl'.format(j)
            # print(path1)
            # with open('./data/samples/bulk_{}.pkl'.format(j), 'rb') as f:
            #     bulk_pairs = pickle.load(f)
            with open('./data/samples/sc_{}.pkl'.format(i), 'rb') as f:
                sc_pairs = pickle.load(f)

            # bulk_store = pd.DataFrame(bulk_pairs)
            # del bulk_store[0]
            # del bulk_store[1]
            # # store = bulk_store
            # bulk_features = preprocess(bulk_store)

            sc_store = pd.DataFrame(sc_pairs)
            del sc_store[0]#删除的0和1是基因的名字，sc_store中存储的是经过log的基因表达值，里面有较多的的值为-2，因为基因表达为0的值经过处理后值为-2
            del sc_store[1]
            try:
                sc_features = preprocess(sc_store)
                print(sc_features.columns)
                print(sc_features)
            except Exception as e:
                print(e)

            # with open(f'feature_engineering/data/features2/bulk_features_{i}.pkl', 'wb') as f:
            #     pickle.dump(bulk_features, f)

            # with open('./data/features/sc_features_{}.pkl'.format(j), 'wb') as f:
            with open('./data/features_test_two/sc_features_{}.pkl'.format(i), 'wb') as f:
                pickle.dump(sc_features, f)
        except FileNotFoundError:
            print("error")
            file_error.append(i)
