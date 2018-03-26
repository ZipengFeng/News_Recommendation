import pandas as pd
import time
import datetime

def extract():
    data_df = pd.read_csv("./Data/user_click_data.txt", sep='\t', header=-1)
    user_dict = dict()
    news_dict = dict()
    uid = 0
    nid = 0
    for i in range(data_df.shape[0]):
        user_id = data_df.iloc[i, 0]
        news_id = data_df.iloc[i, 1]
        if user_id not in user_dict:
            user_dict[user_id] = uid
            uid += 1
        if news_id not in news_dict:
            news_dict[news_id] = nid
            nid += 1
    user_list = sorted(user_dict.items(), key=lambda e: e[1], reverse=False)
    news_list = sorted(news_dict.items(), key=lambda e: e[1], reverse=False)

    user_news_df = data_df.loc[:, 0:2]
    news_id_file = open("./Data/news_id.txt", 'w', encoding='utf-8')
    for i in range(data_df.shape[0]):
        news_id_file.write(str(news_dict[user_news_df.iloc[i, 1]])+ '\t' + str(user_news_df.iloc[i, 1]) + '\n')
        print('%d st. Done.' % i)
    news_id_file.close()

def time_transform(value):
    dt = datetime.datetime.fromtimestamp(
        int(value)
    ).strftime('%Y-%m-%d %H:%M:%S')
    return dt

def split_data():
    data_df = pd.read_csv("./Data/user_news_id.csv", header=-1, sep='\t')
    train_file = open("./Data/train_data.txt", 'w', encoding='utf-8')
    test_file = open("./Data/test_data.txt", 'w', encoding='utf-8')
    for i in range(data_df.shape[0]):
        read_time = data_df.iloc[i, 2]
        date = time_transform(read_time)
        day = date.split(' ')[0].split('-')[2]
        uid = str(data_df.iloc[i, 0])
        nid = str(data_df.iloc[i, 1])
        if int(day) < 20:
            train_file.write(uid + '\t' + nid + '\t' + date + '\n')
        else:
            test_file.write(uid + '\t' + nid + '\t' + date + '\n')
    train_file.close()
    test_file.close()

extract()
split_data()
