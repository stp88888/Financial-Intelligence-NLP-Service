# -*- coding: utf-8 -*-
"""
Created on Mon May 21 22:38:06 2018

@author: STP
"""

import sys
import pandas as pd
import numpy as np
import jieba
import jieba.analyse
import re
import logging
from gensim.models import word2vec
import lightgbm as lgb
from sklearn.externals import joblib
from sklearn.utils import shuffle
from sklearn.svm import SVC

#input_path = sys.argv[1]
#output_path = sys.argv[2]

jieba.load_userdict('./new_add.txt')
jieba.analyse.set_stop_words('./stop.txt')

stopwords = pd.Series([line.strip() for line in open('./stop.txt').readlines()]).str.decode('utf-8')

def clean_word(word):
    word = word.replace('***', '一').decode('utf-8')
    word = re.findall(u'[\u4e00-\u9fa5]', word)
    word_str = ''
    for i in word:
        word_str += i
    return word_str

def clean_word_stop(word, stopwords):
    output = []
    for i in word:
        if i not in stopwords:
            if i !='\t':
                output.append(i)
    return output

def cut_word(word, cut_all=False, stop=True):
    word = clean_word(word)
    word_list = jieba.cut(word, cut_all=cut_all)
    if stop:
        return clean_word_stop(word_list, stopwords)
    return list(word_list)

def cut_for_search_word(word, HMM=False, stop=True):
    word = clean_word(word)
    word_list = jieba.cut(word, HMM=HMM)
    if stop:
        return clean_word_stop(word_list, stopwords)
    return list(word_list)

def cut_idf_word(word, topK=20, stop=True):
    word = clean_word(word)
    word_list = jieba.analyse.extract_tags(word, topK=topK)
    if stop:
        return clean_word_stop(word_list, stopwords)
    return list(word_list)

def cut_textrank_word(word, stop=True):
    word = clean_word(word)
    word_list = jieba.analyse.textrank(word)
    if stop:
        return clean_word_stop(word_list, stopwords)
    return list(word_list)

def getAvgFeatureVecs(reviews, model):
    nwords = 0
    featureVec = np.zeros((num_features,), dtype='float')
    index2word_set = set(model.wv.index2word)
    for word in reviews:
        if word in index2word_set:
            nwords += 1
            featureVec = np.add(featureVec, model.wv[word])
    if nwords == 0:
        return featureVec
    else:
        featureVec = np.divide(featureVec, nwords)
        return featureVec

data = pd.read_csv('./atec_nlp_sim_train.csv', sep='	', header=None)
data = data.drop([0], axis=1)
data.columns = ['review1', 'review2', 'y']

#devide word
review_list = []
for _, each_review in data.iterrows():
    each_review1 = cut_idf_word(each_review['review1'])
    each_review2 = cut_idf_word(each_review['review2'])
    review_list.append([each_review1, each_review2])

simmilar = []
for each_review in review_list:
    each_review1, each_review2 = set(each_review[0]), set(each_review[1])
    same = each_review1 & each_review2
    if len(same) > 0.8*len(each_review1) or len(same) > 0.8*len(each_review2) or len(each_review1)-len(same) <= 2 or len(each_review2)-len(same) <= 2:
        simmilar.append(1)
    else:
        simmilar.append(0)

review_new_list = []
for i in review_list:
    for j in i:
        review_new_list.append(j)

num_features = 500
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
model = word2vec.Word2Vec(review_new_list, workers=8, size=num_features, min_count=5, window=5)

train = np.zeros((len(review_list),num_features),dtype='float')
for i, each_review in enumerate(review_list):
    each_review1, each_review2 = getAvgFeatureVecs(each_review[0], model), getAvgFeatureVecs(each_review[1], model)
    train[i] = each_review1 - each_review2

'''
params = {'application': 'binary',
            'boosting': 'gbdt',
            'lambda_l2': 1,
            #'gamma': 5,
            'slient': 1,
            'lambda_l1': 0,
            'num_leaves': 80,
            'is_unbalance':True,
            #'subsample': 0.7,
            #'colsample_bytree': 0.7,
            'learning_rate': 0.01,
            'max_depth':6,
            'min_data_in_leaf':20,
            #'seed': 1,
            'num_threads': 8}
lgb_train = lgb.Dataset(train, label=data['y'])
watchlist = {lgb_train:'train'}
lgb_model = lgb.train(params, lgb_train, num_boost_round=350, evals_result=watchlist, verbose_eval=50)
lgb_model.save_model('lgb.model')
'''
piece = 5
data_true_index = data[data.y == 1].index
data_flase_index = data[data.y == 0].index
data_flase_index = shuffle(data_flase_index)
data_flase_index = np.array_split(data_flase_index, piece)
for i in range(piece):
    model_name = './svm_' + str(i)
    index_temp = data_true_index | data_flase_index[i]
    train_temp = train[index_temp]
    label_temp = data.y.values[[index_temp]]
    svm = SVC()
    svm.fit(train_temp, data['y'].values[index_temp])
    joblib.dump(svm, model_name)
    print 'model:', i, 'finished'
