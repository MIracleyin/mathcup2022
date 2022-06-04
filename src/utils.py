from typing import Optional
import jieba
import re

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

