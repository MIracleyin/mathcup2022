import os.path
from typing import Optional
import jieba
import re

import pandas as pd
import numpy as np
import pyLDAvis
import pyLDAvis.sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

raw_part1_path = "dataset/dataset-2018-2019.xlsx"
raw_part2_path = "dataset/dataset-2020-2021.xlsx"
processed_travel = "dataset/processed/travelid_text.csv"
processed_news = "dataset/processed/newsid_text.csv"
stop_path = "dataset/stopwords.txt"


def clean_text(text: str):
    if not isinstance(text, str):
        text = str(text)
    text = text.replace('\n', " ")  # 新行，我们是不需要的
    text = re.sub("\d+", '', text)  # 删除数字
    text = re.sub('[a-zA-Z]', '', text)  # 删除字母
    text = re.sub('[\s]', '', text)  # 删除空格
    print(text)
    pure_text = ''
    # 以防还有其他特殊字符（数字）等等，我们直接把他们loop一遍，过滤掉
    for letter in text:
        if letter.isalpha() or letter == ' ':
            pure_text += letter
    text = ' '.join(word for word in pure_text.split() if len(word) > 1)
    return text


def cut_text(text: str) -> str:
    text_with_spaces = ''
    text_cut = jieba.cut(text)
    for word in text_cut:
        text_with_spaces += (word) + ' '
    return text_with_spaces


def get_stop_words(path: str):
    with open(path, encoding='utf-8') as f:
        stopword = f.read()
        stopword_list = stopword.splitlines()
    return stopword_list


def filter_stop_words(text: str) -> str:
    with open('dataset/stopwords.txt', encoding='utf-8') as f:
        stopword = f.read()
        stopword_list = stopword.splitlines()
        text_list = text.split(' ')
        filted_text = ''
        for word in text_list:
            if word not in stopword_list:
                filted_text += word + ' '

    return filted_text


def get_topic_list():
    topic = "旅游、活动、节庆、特产、交通、酒店、景区、景点、文创、文化、乡村旅游、民宿、假日、假期、游客、采摘、赏花、春游、踏青、康养、公园、滨海游、度假、农家乐、剧本杀、旅行、徒步、工业旅游、线路、自驾游、 团队游、攻略、游记、包车、玻璃栈道、游艇、高尔夫、温泉"
    return topic.split("、")


def get_topic_words(model, feature_names, n_top_words):
    tword = []
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        topic_w = " ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
        tword.append(topic_w)
        print(topic_w)
    return tword


def resolve_q1(process_data: pd.DataFrame, save_path: str,
               is_vis=True,
               n_topics=5,
               n_topic_words=30,
               threshold=0,
               n_feature=1000,
               ):
    assert 'ID' in process_data.columns
    assert 'text' in process_data.columns
    process_data = process_data.dropna()
    # get data
    corpus = process_data['text'].tolist()

    # transform to vector by cnt
    cnt_vector = CountVectorizer(strip_accents='unicode',
                                 # max_features=n_feature,
                                 stop_words=get_stop_words(stop_path),
                                 max_df=0.5,
                                 min_df=10)
    cnt_tf = cnt_vector.fit_transform(corpus) 

    # build LDA model
    lda = LatentDirichletAllocation(n_components=n_topics,
                                    max_iter=50,
                                    learning_method='batch',
                                    learning_offset=50.,
                                    # doc_topic_prior=0.1,
                                    # topic_word_prior=0.01,
                                    random_state=0)
    docs_res = lda.fit_transform(cnt_tf)

    # get each topic words
    tf_feature_names = cnt_vector.get_feature_names_out()
    topic_word = get_topic_words(lda, tf_feature_names, n_topic_words)
    print(topic_word)

    # get topic count(appare in topic_list) map
    topic_list = get_topic_list()
    topic_count_dict = {}
    for t_id, topic in enumerate(topic_word):
        t_list = topic.split()
        t_count = 0
        for w in t_list:
            if w in topic_list:
                t_count += 1
        topic_count_dict[t_id] = t_count
    print(topic_count_dict)

    # get results
    topics = lda.transform(cnt_tf) # 200 [(200,6)]
    is_related = []
    for t in topics: # [1, 6]
        related_number = topic_count_dict[list(t).index(np.max(t))] 
        is_related.append(1 if related_number > threshold else 0) # 
    process_data['is_realted'] = is_related
    process_data = pd.DataFrame(process_data, columns=['ID', 'is_realted'])
    process_data.to_csv(os.path.join("dataset/results", save_path), index=0)

    if is_vis:
        pic = pyLDAvis.sklearn.prepare(lda, cnt_tf, cnt_vector)
        pyLDAvis.display(pic)
        pyLDAvis.save_html(pic, 'lda_pass' + str(n_topics) + '.html')
        pyLDAvis.display(pic)
