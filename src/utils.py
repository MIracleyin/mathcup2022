import os.path
from typing import Optional
import jieba
import re

import pandas as pd
import numpy as np
import pyLDAvis
import pyLDAvis.sklearn
import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nerpy import NERModel

from transformers import (
    BertTokenizerFast,
    AutoTokenizer,
    AutoModelForTokenClassification,
    AutoModel,
    pipeline,
    BertForSequenceClassification,
    BertTokenizer

)

import torch

raw_part1_path = "dataset/dataset-2018-2019.xlsx"
raw_part2_path = "dataset/dataset-2020-2021.xlsx"
processed_travel = "dataset/processed/travelid_text.csv"
processed_news = "dataset/processed/newsid_text.csv"
processed_path = "dataset/processed"
results_path = "dataset/results"
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


def clean_textv2(text: str):
    if not isinstance(text, str):
        text = str(text)
    text = text.replace('\n', " ")  # 新行，我们是不需要的
    text = re.sub("\d+", '', text)  # 删除数字
    text = re.sub('[a-zA-Z]', '', text)  # 删除字母
    text = re.sub('[\s]', '', text)  # 删除空格
    print(text)
    pure_text = ''
    # 以防还有其他特殊字符（数字）等等，我们直接把他们loop一遍，过滤掉
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
    topics = lda.transform(cnt_tf)  # 200 [(200,6)]
    is_related = []
    for t in topics:  # [1, 6]
        related_number = topic_count_dict[list(t).index(np.max(t))]
        is_related.append(1 if related_number > threshold else 0)  #
    process_data['is_realted'] = is_related
    process_data = pd.DataFrame(process_data, columns=['ID', 'is_realted'])
    process_data.to_csv(os.path.join("dataset/results", save_path), index=0)

    if is_vis:
        pic = pyLDAvis.sklearn.prepare(lda, cnt_tf, cnt_vector)
        pyLDAvis.display(pic)
        pyLDAvis.save_html(pic, 'lda_pass' + str(n_topics) + '.html')
        pyLDAvis.display(pic)


def get_class_infors():
    dataset_hotel_1 = pd.read_excel(raw_part1_path, sheet_name='酒店评论')
    dataset_hotel_2 = pd.read_excel(raw_part2_path, sheet_name='酒店评论')

    dataset_scenic_1 = pd.read_excel(raw_part1_path, sheet_name='景区评论')
    dataset_scenic_2 = pd.read_excel(raw_part2_path, sheet_name='景区评论')

    dataset_travel_1 = pd.read_excel(raw_part1_path, sheet_name='游记攻略')
    dataset_travel_2 = pd.read_excel(raw_part2_path, sheet_name='游记攻略')

    dataset_food_1 = pd.read_excel(raw_part1_path, sheet_name='餐饮评论')
    dataset_food_2 = pd.read_excel(raw_part2_path, sheet_name='餐饮评论')

    dataset_news_1 = pd.read_excel(raw_part1_path, sheet_name='微信公众号新闻')
    dataset_news_2 = pd.read_excel(raw_part2_path, sheet_name='微信公众号新闻')

    hotel = pd.concat([dataset_hotel_1, dataset_hotel_2], axis=0)
    scenic = pd.concat([dataset_scenic_1, dataset_scenic_2], axis=0)
    travel = pd.concat([dataset_travel_1, dataset_travel_2], axis=0)
    food = pd.concat([dataset_food_1, dataset_food_2], axis=0)
    news = pd.concat([dataset_news_1, dataset_news_2], axis=0)
    travel = pd.read_csv("dataset/processed/travel_related.csv")
    news = pd.read_csv("dataset/processed/news_related.csv")

    return hotel, scenic, travel, food, news


def modify_infors(hotel, scenic, travel, food, news):
    def addstr(pre: str, text):
        if not isinstance(text, str):
            text = str(text)
        return pre + '-' + text

    hotel['curpusID'] = hotel['酒店评论ID'].apply(lambda x: addstr('酒店评论', x))
    hotel['text'] = hotel['评论内容']
    hotel['product'] = hotel['酒店名称']
    hotel['years'] = pd.to_datetime(hotel['评论日期']).dt.year

    scenic['curpusID'] = scenic['景区评论ID'].apply(lambda x: addstr('景区评论', x))
    scenic['text'] = scenic['评论内容']
    scenic['product'] = scenic['景区名称']
    scenic['years'] = pd.to_datetime(scenic['评论日期']).dt.year

    travel['curpusID'] = travel['游记ID'].apply(lambda x: addstr('旅游攻略', x))
    travel['text'] = travel['游记标题'] + '\n' + travel['正文']
    travel['years'] = pd.to_datetime(travel['发布时间']).dt.year
    travel['text'] = travel['text'].apply(lambda x: clean_textv2(x))
    text_list = travel['text'].values.tolist()
    travel['product'] = get_ner_infer(text_list)

    food['curpusID'] = food['餐饮评论ID'].apply(lambda x: addstr('餐饮评论', x))
    food['text'] = food['评论内容'] + '\n' + food['标题']
    food['product'] = food['餐饮名称']
    food['years'] = pd.to_datetime(food['评论日期']).dt.year

    news['curpusID'] = news['文章ID'].apply(lambda x: addstr('微信公共号文章', x))
    news['text'] = news['公众号标题'] + '\n' + news['正文']
    news['years'] = pd.to_datetime(news['发布时间']).dt.year
    news['text'] = news['text'].apply(lambda x: clean_textv2(x))
    text_list = news['text'].values.tolist()
    news['product'] = get_ner_infer(text_list)

    hotel = pd.DataFrame(hotel, columns=['curpusID', 'text', 'product', 'years'])
    scenic = pd.DataFrame(scenic, columns=['curpusID', 'text', 'product', 'years'])
    travel = pd.DataFrame(travel, columns=['curpusID', 'text', 'product', 'years'])
    food = pd.DataFrame(food, columns=['curpusID', 'text', 'product', 'years'])
    news = pd.DataFrame(news, columns=['curpusID', 'text', 'product', 'years'])

    return hotel, scenic, travel, food, news


def get_all_infors(hotel, scenic, travel, food, news):
    df = pd.concat([hotel, scenic, travel, food, news], axis=0)
    df = df.dropna().drop_duplicates(['product'])
    product_id = ['ID' + str(i + 1) for i in range(len(df))]
    df['productID'] = product_id
    return df


def get_ner_infer(text):
    # nlp task model
    # tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
    # model = AutoModelForTokenClassification.from_pretrained('ckiplab/bert-base-chinese-ner')
    p = "黑龙江省、辽宁省、吉林省、河北省、河南省、湖北省、湖南省、山东省、山西省、陕西省、安徽省、浙江省、江苏省、福建省、广东省、海南省、四川省、云南省、贵州省、青海省、甘肃省、江西省、台湾省".split('、')
    p2 = "黑龙江省、辽宁省、吉林省、河北省、河南省、湖北省、湖南省、山东省、山西省、陕西省、安徽省、浙江省、江苏省、福建省、广东省、海南省、四川省、云南省、贵州省、青海省、甘肃省、江西省、台湾省".replace("省","").split('、')
    c = "珠海市、汕头市、佛山市、韶关市、湛江市、肇庆市、江门市、茂名市、惠州市、梅州市、汕尾市、河源市、阳江市、清远市、东莞市、中山市、潮州市、揭阳市、云浮市".split('、')
    c2 = "珠海市、汕头市、佛山市、韶关市、湛江市、肇庆市、江门市、茂名市、惠州市、梅州市、汕尾市、河源市、阳江市、清远市、东莞市、中山市、潮州市、揭阳市、云浮市".replace("市","").split('、')
    tmp = ["中国", "广西"]
    f = p + p2 + c + c2 + tmp
    model = NERModel("bert", "shibing624/bert4ner-base-chinese")
    _, _, entities = model.predict(text)
    long_word = []
    for ent in entities:
        word_list = []
        for word_type in ent:
            word, type = word_type
            if type == 'LOC':
                if word not in f:
                    word_list.append(word)
        len_word = max(word_list, key=len, default='')
        long_word.append(len_word)

    return long_word

def get_sentiment(text_list):
    print("------------------bert----------------------")
    outputs = []
    tokenizer = BertTokenizer.from_pretrained('IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment')
    # inputs = tokenizer(batch_list, padding=True, truncation=True, return_tensors="pt",
    #                    max_length=512).to('cuda')
    model = BertForSequenceClassification.from_pretrained('IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment')
    pbar = tqdm.tqdm(text_list)
    for text in pbar:
        output = model(torch.tensor([tokenizer.encode(text,max_length=512,truncation=True)]))
        output = torch.nn.functional.softmax(output.logits, dim=-1).tolist()[0]
        if output[0] > output[1]:
            score = - output[0]
        elif output[0] < output[1]:
            score = output[1]
        else:
            score = 0
        outputs.append(round(score,4))
    return outputs

def get_frequence(all_df):
    year_2018_count = all_df[all_df['years'] == 2018]
    year_2019_count = all_df[all_df['years'] == 2019]
    year_2020_count = all_df[all_df['years'] == 2020]
    year_2021_count = all_df[all_df['years'] == 2021]

    dict_2018 = dict(year_2018_count['product'].value_counts())

    def get_frequency(s):
        print(s)
        if s is np.nan:
            fre = 0
        else:
            fre = dict_2018[s]
        return fre

    year_2018_count['frequence'] = year_2018_count['product'].apply(get_frequency)

    dict_2019 = dict(year_2019_count['product'].value_counts())

    def get_frequency(s):
        fre = dict_2019[s]
        return fre

    year_2019_count['frequence'] = year_2019_count['product'].apply(get_frequency)

    dict_2020 = dict(year_2020_count['product'].value_counts())

    def get_frequency(s):
        fre = dict_2020[s]
        return fre

    year_2020_count['frequence'] = year_2020_count['product'].apply(get_frequency)

    dict_2021 = dict(year_2021_count['product'].value_counts())

    def get_frequency(s):
        fre = dict_2021[s]
        return fre

    year_2021_count['frequence'] = year_2021_count['product'].apply(get_frequency)

    # return dict_2018,dict_2019,dict_2020,dict_2021
    return year_2018_count, year_2019_count, year_2020_count, year_2021_count


def get_final_socre(year_2018_count, year_2019_count, year_2020_count, year_2021_count):
    # 计算综合得分
    year_2018_count['hot_score'] = 0.8 * year_2018_count['frequence'] + 200 * year_2018_count['emotion_score'] + 0
    year_2019_count['hot_score'] = 0.8 * year_2019_count['frequence'] + 200 * year_2019_count['emotion_score'] + 1 * 10
    year_2020_count['hot_score'] = 0.8 * year_2020_count['frequence'] + 200 * year_2020_count['emotion_score'] + 2 * 15
    year_2021_count['hot_score'] = 0.8 * year_2021_count['frequence'] + 200 * year_2021_count['emotion_score'] + 3 * 20


    product_hot_score = pd.concat([year_2018_count, year_2019_count, year_2020_count, year_2021_count], axis=0)

    year_2018_count['final_hot'] = year_2018_count['hot_score'].div(np.sum(product_hot_score['hot_score']),                                                    axis=0)  # 化成一个小数 加起来为1
    year_2019_count['final_hot'] = year_2019_count['hot_score'].div(np.sum(product_hot_score['hot_score']), axis=0)
    year_2020_count['final_hot'] = year_2020_count['hot_score'].div(np.sum(product_hot_score['hot_score']), axis=0)
    year_2021_count['final_hot'] = year_2021_count['hot_score'].div(np.sum(product_hot_score['hot_score']), axis=0)

    year_2018 = year_2018_count.sort_values(by="final_hot", ascending=False).reset_index(drop=True)
    year_2019 = year_2019_count.sort_values(by="final_hot", ascending=False).reset_index(drop=True)
    year_2020 = year_2020_count.sort_values(by="final_hot", ascending=False).reset_index(drop=True)
    year_2021 = year_2021_count.sort_values(by="final_hot", ascending=False).reset_index(drop=True)

    product_hot_score_sort = pd.concat([year_2018, year_2019, year_2020, year_2021], axis=0)
    # product_hot_score['文本'] = product_hot_score['文本'].progress_apply(clearTxt)
    product_hot_score_sort['productJudge'] = product_hot_score_sort['curpusID'] + ' ' + product_hot_score_sort['text']
    product_hot_score_sort['productClass'] = product_hot_score_sort['productJudge'].apply(get_product_type)
    # 去除重复的产品
    product_hot_score_sort= product_hot_score_sort.drop_duplicates(['product'])
    # 产品 ID 产品类型 产品名称 产品热度 年份
    result2_2 = product_hot_score_sort[['productID', 'productClass', 'product', 'hot_score', 'years']]
    result2_2['productID'] = ['ID' + str(i + 1) for i in range(len(result2_2))]

    return result2_2


def get_product_type(s):
    if '景区' in s:
        return '景区'
    elif '酒店' in s:
        return '酒店'
    elif '餐饮' in s:
        return '特色餐饮'
    elif '景点' in s:
        return '景点'
    elif '民宿' in s:
        return '民宿'
    elif '乡村' in s:
        return '乡村旅游'
    elif '文创' in s:
        return '文创'
    else:
        return '景点'

def clearTxt(line):
    stopword_list = [k.strip() for k in open(
        './datasets/stopwords.txt', encoding='utf8').readlines() if k.strip() != '']
    if line != '':
        line = str(line).strip()
        # 去除文本中的英文和数字
        line = re.sub("[a-zA-Z0-9]", "", line)
        # 只保留中文、大小写字母
        reg = "[^0-9A-Za-z\u4e00-\u9fa5]"
        line = re.sub(reg, '', line)
        # 分词
        segList = jieba.cut(line, cut_all=False)
        segSentence = ''
        for word in segList:
            if word != '\t':
                segSentence += word + " "
    # 去停用词
    wordList = segSentence.split(' ')
    sentence = ''
    for word in wordList:
        word = word.strip()
        if word not in stopword_list:
            if word != '\t':
                sentence += word + " "
    return sentence.strip()


if __name__ == '__main__':
    t = ["今天心情不好", "今天心情好"]
    o = get_sentiment(t)
    print(o)
