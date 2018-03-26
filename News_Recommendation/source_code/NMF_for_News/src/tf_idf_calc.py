import jieba
import jieba.analyse
import math

import pandas as pd
import numpy as np
import re

from sklearn.metrics import pairwise_distances

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

k = 20

news_id_df = pd.read_csv("./Data/news_id.txt", header=-1, sep=' ')
data_df = pd.read_csv("./Data/user_click_data.txt", header=-1, sep='\t')
f = open("./Data/stop_words.txt", encoding = 'utf-8')
words = f.read()
stop_words = set(words.split('\n'))

news_id_dict = dict()
for i in range(news_id_df.shape[0]):
    new_news_id = news_id_df.iloc[i, 0]
    old_news_id = news_id_df.iloc[i, 1]
    if old_news_id not in news_id_dict:
        news_id_dict[old_news_id] = new_news_id
    else:
        continue

news_body_dict = dict()
for i in range(data_df.shape[0]):
    news_id = data_df.iloc[i, 1]
    news_title = data_df.iloc[i, 3]
    news_body = data_df.iloc[i, 4]
    if news_id not in news_body_dict:
        try:
            news_body_dict[news_id] = news_title + '+' + news_body
        except:
            news_body_dict[news_id] = news_title + '+' + 'NULL'
    else:
        continue

n = 6183
idf = {}
for news_id,news_body in news_body_dict.items():
    body = news_body.split('+')[1]
    title = news_body.split('+')[0]
    rule = re.compile("[^\u4e00-\u9fa5]")
    body = rule.sub('',body)
    seg_list = jieba.lcut(title + '。' + body, cut_all=False, HMM=False)
    seg_list = set(seg_list) - stop_words
    for word in seg_list:
        word = word.strip().lower()
        if word == '' or is_number(word):
            continue
        if word not in idf:
            idf[word] = 1
        else:
            idf[word] = idf[word] + 1
idf_file = open("./Data/td_idf.txt", 'w', encoding = 'utf-8')
for word, df in idf.items():
    idf_file.write('%s %.9f\n'%(word, math.log(n / df)))
idf_file.close()

jieba.analyse.set_stop_words("./Data/stop_words.txt")
jieba.analyse.set_idf_path("./Data/td_idf.txt")
dt = []
M = 6183
N = 1
terms = {}

for news_id,news_body in news_body_dict.items():
    body = news_body.split('+')[1]
    title = news_body.split('+')[0]
    docid = int(news_id)
    try:
        tags = jieba.analyse.extract_tags(title + '。' + body, topK=k, withWeight=True)
    except:
        print("Out of TOP-K Range.")
        continue
    #tags = jieba.analyse.extract_tags(title, topK=topK, withWeight=True)
    cleaned_dict = {}
    for word, tfidf in tags:
        word = word.strip().lower()
        cleaned_dict[word] = tfidf
        if word not in terms:
            terms[word] = N
            N += 1
    dt.append([docid, cleaned_dict])
dt_matrix = [[0 for i in range(N)] for j in range(M)]
i =0
for docid, t_tfidf in dt:
    print(docid)
    print(news_id_dict[docid])
    dt_matrix[news_id_dict[docid]][0] = news_id_dict[docid]
    for word, tfidf in t_tfidf.items():
        dt_matrix[news_id_dict[docid]][terms[word]] = tfidf
    i += 1
#print(dt_matrix)
dt_matrix = pd.DataFrame(dt_matrix)
dt_matrix.index = dt_matrix[0]
print('dt_matrix shape:(%d %d)'%(dt_matrix.shape))

tmp = np.array(1 - pairwise_distances(dt_matrix[dt_matrix.columns[1:]], metric = "cosine"))
np.savetxt("./Data/similarity_matrix.txt", tmp)
print(tmp)
