import re
import jieba
import pandas as pd
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import metrics
from textrank4zh import TextRank4Keyword  # 导入textrank4zh模块
from cnsenti import Sentiment
import numpy as np

from collections import defaultdict
import matplotlib.pyplot as plt
import networkx as nx



warnings.filterwarnings('ignore')

def cut_words(file_path):
    """
    对文本进行切词
    :param file_path: txt文本路径
    :return: 用空格分词的字符串
    """
    text_with_spaces = ''
    text = open(file_path, 'r', encoding='UTF-8').read()
    textcut = jieba.cut(text)
    for word in textcut:
        text_with_spaces += word + ' '
    return text_with_spaces

def clean_text(text):
    text = text.replace('\n', " ")  # 新行，我们是不需要的
    text = re.sub("\d+", '', text)  # 删除数字
    text = re.sub('[a-zA-Z]', '', text)  # 删除字母
    text = re.sub('[\s]', '', text)  # 删除空格
    pure_text = ''
    # 以防还有其他特殊字符（数字）等等，我们直接把他们loop一遍，过滤掉
    for letter in text:
        if letter.isalpha() or letter == ' ':
            pure_text += letter
    text = ' '.join(word for word in pure_text.split() if len(word) > 1)
    return text

def load_infos(dataset_2018, dataset_2020):

    '''
    旅游产品，亦称旅游服务产品。是指由实物和服务构成。包括旅行商集合景点、交通、食宿、娱乐等设施设备、
    项目及相应服务出售给旅游者的旅游线路类产品，旅游景区、旅游饭店等单个企业提供给旅游者的活动项目类产品
    '''

    Hotel_Info1 = pd.read_excel(
        dataset_2018, sheet_name=0)  # 酒店评论
    Scenic_Info1 = pd.read_excel(
        dataset_2018, sheet_name=1)  # 景区评论
    Travel_Info1 = pd.read_excel(
        dataset_2018, sheet_name=2)  # 游记攻略
    Dining_Info1 = pd.read_excel(
        dataset_2018, sheet_name=3)  # 餐饮评论
    Wechat_Info1 = pd.read_excel(
        dataset_2018, sheet_name=4)  # 微信公众号文章

    Hotel_Info2 = pd.read_excel(
        dataset_2020, sheet_name=0)  # 酒店评论
    Scenic_Info2 = pd.read_excel(
        dataset_2020, sheet_name=1)  # 景区评论
    Travel_Info2 = pd.read_excel(
        dataset_2020, sheet_name=2)  # 游记攻略
    Dining_Info2 = pd.read_excel(
        dataset_2020, sheet_name=3)  # 餐饮评论
    Wechat_Info2 = pd.read_excel(
        dataset_2020, sheet_name=4)  # 微信公众号文章

    Hotel_Infos = pd.concat([Hotel_Info1, Hotel_Info2], axis=0)  # 酒店评论
    Scenic_Infos = pd.concat([Scenic_Info1, Scenic_Info2], axis=0)  # 景区评论
    Travel_Infos = pd.concat([Travel_Info1, Travel_Info2], axis=0)  # 游记攻略
    Dining_Infos = pd.concat([Dining_Info1, Dining_Info2], axis=0)  # 餐饮评论
    Wechat_Infos = pd.concat([Wechat_Info1, Wechat_Info2], axis=0)  # 微信公众号文章

    return Hotel_Infos, Scenic_Infos, Travel_Infos, Dining_Infos, Wechat_Infos

def modify_infos(Hotel_Infos, Scenic_Infos, Travel_Infos, Dining_Infos, Wechat_Infos):

    def addstr(s):
        return '景区评论-' + str(s)

    Scenic_Infos['语料ID'] = Scenic_Infos['景区评论ID'].progress_apply(addstr)
    Scenic_Infos['文本'] = Scenic_Infos['评论内容']
    Scenic_Infos['产品名称'] = Scenic_Infos['景区名称']
    Scenic_Infos['年份'] = pd.to_datetime(Scenic_Infos['评论日期']).dt.year

    # Hotel_Infos.head(10)

    def addstr(s):
        return '酒店评论-' + str(s)

    Hotel_Infos['语料ID'] = Hotel_Infos['酒店评论ID'].progress_apply(addstr)
    Hotel_Infos['文本'] = Hotel_Infos['评论内容']
    Hotel_Infos['产品名称'] = Hotel_Infos['酒店名称']
    Hotel_Infos['年份'] = pd.to_datetime(Hotel_Infos['评论日期']).dt.year

    def addstr(s):
        return '餐饮评论-' + str(s)

    Dining_Infos['语料ID'] = Dining_Infos['餐饮评论ID'].progress_apply(addstr)
    Dining_Infos['文本'] = Dining_Infos['评论内容'] + '\n' + Dining_Infos['标题']
    Dining_Infos['产品名称'] = Dining_Infos['餐饮名称']
    Dining_Infos['年份'] = pd.to_datetime(Dining_Infos['评论日期']).dt.year

    def addstr(s):
        return '旅游攻略-' + str(s)

    Travel_Infos['语料ID'] = Travel_Infos['游记ID'].progress_apply(addstr)
    Travel_Infos['文本'] = Travel_Infos['游记标题'] + '\n' + Travel_Infos['正文']
    Travel_Infos['年份'] = pd.to_datetime(Travel_Infos['发布时间']).dt.year
    Travel_Infos['产品名称'] = Travel_Infos['文本'].progress_apply(get_keyphrase)

    # 微信公众号文章
    # Wechat_Infos = pd.concat([Wechat_Info1, Wechat_Info2], axis=0)  # 微信公众号文章

    def addstr(s):
        return '微信公共号文章-' + str(s)

    Wechat_Infos['语料ID'] = Wechat_Infos['文章ID'].progress_apply(addstr)
    Wechat_Infos['文本'] = Wechat_Infos['公众号标题'] + '\n' + Wechat_Infos['正文']
    Wechat_Infos['年份'] = pd.to_datetime(Wechat_Infos['发布时间']).dt.year
    Wechat_Infos['产品名称'] = Wechat_Infos['文本'].progress_apply(get_keyphrase)

    Travel_Infos = Travel_Infos.dropna(subset=["产品名称"])
    Wechat_Infos = Wechat_Infos.dropna(subset=["产品名称"])

    return Hotel_Infos, Scenic_Infos, Travel_Infos, Dining_Infos, Wechat_Infos

# 采用Textrank提取关键词组算法
# 这部分待改进

def get_keyphrase(s):
    tr4w = TextRank4Keyword(
        allow_speech_tags=['n', 'nr', 'nr1', 'nr2', 'nrf', 'ns', 'nsf', 'nt', 'nz', 'nz', 'nl', 'ng'])
    tr4w.analyze(text=str(s), lower=True, window=5)  # 文本分析，文本小写，窗口为2
    # 最多5个关键词组，有可能一个也没有。词组在原文中出现次数最少为1。
    phase_list = tr4w.get_keyphrases(keywords_num=5, min_occur_num=1)
    if len(phase_list) == 0:
        return np.nan
    else:
        return phase_list[0]



def emotion_score(s):

    senti = Sentiment(pos='./datasets/pos.txt',  # 正面词典txt文件相对路径
                      neg='./datasets/neg.txt',  # 负面词典txt文件相对路径
                      merge=True,  # 融合cnsenti自带词典和用户导入的自定义词典
                      encoding='utf-8')  # 两txt均为utf-8编码
    r = senti.sentiment_count(str(s))
    if r['pos'] > r['neg']:
        score = (r['pos'] - r['neg']) / r['words']
    elif r['pos'] < r['neg']:
        score = (r['pos'] - r['neg']) / r['words']
    else:
        score = 0
    return score

def hot_score(all_df):
    year_2018_count = all_df[all_df['年份'] == 2018]
    year_2019_count = all_df[all_df['年份'] == 2019]
    year_2020_count = all_df[all_df['年份'] == 2020]
    year_2021_count = all_df[all_df['年份'] == 2021]

    dict_2018 = dict(year_2018_count['产品名称'].value_counts())

    def get_frequency(s):
        fre = dict_2018[s]
        return fre

    year_2018_count['出现频次'] = year_2018_count['产品名称'].progress_apply(get_frequency)

    dict_2019 = dict(year_2019_count['产品名称'].value_counts())

    def get_frequency(s):
        fre = dict_2019[s]
        return fre

    year_2019_count['出现频次'] = year_2019_count['产品名称'].progress_apply(get_frequency)

    dict_2020 = dict(year_2020_count['产品名称'].value_counts())

    def get_frequency(s):
        fre = dict_2020[s]
        return fre

    year_2020_count['出现频次'] = year_2020_count['产品名称'].progress_apply(get_frequency)

    dict_2021 = dict(year_2021_count['产品名称'].value_counts())

    def get_frequency(s):
        fre = dict_2021[s]
        return fre

    year_2021_count['出现频次'] = year_2021_count['产品名称'].progress_apply(get_frequency)

    # 计算综合得分
    year_2018_count['产品热度总分'] = 0.8 * year_2018_count['出现频次'] + 200 * year_2018_count['情感得分'] + 0
    year_2019_count['产品热度总分'] = 0.8 * year_2019_count['出现频次'] + 200 * year_2019_count['情感得分'] + 1 * 10
    year_2020_count['产品热度总分'] = 0.8 * year_2020_count['出现频次'] + 200 * year_2020_count['情感得分'] + 2 * 15
    year_2021_count['产品热度总分'] = 0.8 * year_2021_count['出现频次'] + 200 * year_2021_count['情感得分'] + 3 * 20

    year_2018_count['产品热度'] = year_2018_count['产品热度总分'].div(np.sum(year_2018_count['产品热度总分']),
                                                            axis=0)  # 化成一个小数 加起来为1
    year_2019_count['产品热度'] = year_2019_count['产品热度总分'].div(np.sum(year_2019_count['产品热度总分']), axis=0)
    year_2020_count['产品热度'] = year_2020_count['产品热度总分'].div(np.sum(year_2020_count['产品热度总分']), axis=0)
    year_2021_count['产品热度'] = year_2021_count['产品热度总分'].div(np.sum(year_2021_count['产品热度总分']), axis=0)

    year_2018 = year_2018_count.sort_values(by="产品热度", ascending=False).reset_index(drop=True)
    year_2019 = year_2019_count.sort_values(by="产品热度", ascending=False).reset_index(drop=True)
    year_2020 = year_2020_count.sort_values(by="产品热度", ascending=False).reset_index(drop=True)
    year_2021 = year_2021_count.sort_values(by="产品热度", ascending=False).reset_index(drop=True)

    product_hot_score = pd.concat([year_2018_count, year_2019_count, year_2020_count, year_2021_count], axis=0)
    product_hot_score_sort = pd.concat([year_2018, year_2019, year_2020, year_2021], axis=0)
    product_hot_score['文本'] = product_hot_score['文本'].progress_apply(clearTxt)
    product_hot_score['产品类型判断文本'] = product_hot_score['语料ID'] + ' ' + product_hot_score['文本']
    product_hot_score['产品类型'] = product_hot_score['产品类型判断文本'].progress_apply(get_product_type)
    # 去除重复的产品
    product_hot_score2 = product_hot_score.drop_duplicates(['产品名称'])
    # 产品 ID 产品类型 产品名称 产品热度 年份
    result2_2 = product_hot_score2[['产品ID', '产品类型', '产品名称', '产品热度', '年份']]
    result2_2['产品ID'] = ['ID' + str(i + 1) for i in range(len(result2_2))]

    return result2_2


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

 # 景区、酒店、网红景点、民宿、特色餐饮、乡村旅游、文创
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

def prepare_all(Hotel_Infos, Scenic_Infos, Travel_Infos, Dining_Infos):

    all_df = pd.DataFrame(columns=['语料ID', '文本', '产品名称'])
    all_df['语料ID'] = pd.concat([Dining_Infos['语料ID'], Hotel_Infos['语料ID'],
                                Scenic_Infos['语料ID'], Travel_Infos['语料ID']], axis=0)
    all_df['产品名称'] = pd.concat([Dining_Infos['产品名称'], Hotel_Infos['产品名称'],
                                Scenic_Infos['产品名称'], Travel_Infos['产品名称']], axis=0)
    all_df['文本'] = pd.concat([Dining_Infos['文本'], Hotel_Infos['文本'],
                              Scenic_Infos['文本'], Travel_Infos['文本']], axis=0)
    all_df['年份'] = pd.concat([Dining_Infos['年份'], Hotel_Infos['年份'],
                              Scenic_Infos['年份'], Travel_Infos['年份']], axis=0)

    product_id = ['ID' + str(i + 1) for i in range(len(all_df))]
    all_df['产品ID'] = product_id

    return all_df

#计算相关度
def calculate(data_vector):
    print('=' * 50)

    n_samples, n_features = data_vector.shape
    print('特征数: ', n_features)
    print('样本数: ', n_samples)

    support_dict = defaultdict(float)
    confidence_dict = defaultdict(float)
    lift_dict = defaultdict(float)

    together_appear_dict = defaultdict(int)

    feature_num_dict = defaultdict(int)

#计算共同出现的特征次数
    for line in data_vector:
        for i in range(n_features):
            if line[i] == 0:
                continue
            feature_num_dict[i] += 1

            for j in range(n_features):
                if i == j:
                    continue
                if line[j] == 1:
                    together_appear_dict[(i, j)] += 1

    # 计算支持度，置信度，提升度
    for k, v in together_appear_dict.items():
        support_dict[k] = v / n_samples  #支持度
        confidence_dict[k] = v / feature_num_dict[k[0]]  #置信度
        lift_dict[k] = v * n_samples / \
            (feature_num_dict[k[0]] * feature_num_dict[k[1]])
        #提升度

    return support_dict, confidence_dict, lift_dict
    # 返回支持度，置信度，提升度

#对样本进行one-hot编码
def create_one_hot(data):
    data['所有文本'] = data['文本']+' '+data['产品名称']
    all_feature_li = data['产品名称']
    all_feature_set_li = list(set(all_feature_li))
#将所有样本编码成0-1数据形式
    feature_dict = defaultdict(int)
    for n, feat in enumerate(all_feature_set_li):
        feature_dict[feat] = n
    # print(feature_dict)

    out_li = list()
    for j in range(len(data)):
        text = data.iloc[j]['所有文本']
        feature_num_li = []
        for i, f in enumerate(feature_dict):
            if f in text:
                feature_num_li.append(feature_dict[f])
        inner_li = [1 if num in feature_num_li else 0 for num in range(
            len(all_feature_set_li))]

        out_li.append(inner_li)
    out_array = np.array(out_li)
    return out_array, feature_dict
#以one-hot形式存储数据

#提取特定词性的关键词
def dataPrepos(text, stopkey):
    l = []
    pos = ['n', 'nz', 'v', 'vd', 'vn', 'l', 'a', 'd']  # 定义选取的词性
    seg = jieba.posseg.cut(text)  # 分词
    for i in seg:
        if i.word not in l and i.word not in stopkey and i.flag in pos:  # 去重 + 去停用词 + 词性筛选
            # print i.word
            l.append(i.word)
    return l

# #计算相关度
# def calculate(data_vector):
#     print('=' * 50)
#
#     n_samples, n_features = data_vector.shape
#     print('特征数: ', n_features)
#     print('样本数: ', n_samples)
#
#     support_dict = defaultdict(float)
#     confidence_dict = defaultdict(float)
#     lift_dict = defaultdict(float)
#
#     together_appear_dict = defaultdict(int)
#
#     feature_num_dict = defaultdict(int)
#
# #计算共同出现的特征次数
#     for line in data_vector:
#         for i in range(n_features):
#             if line[i] == 0:
#                 continue
#             feature_num_dict[i] += 1
#
#             for j in range(n_features):
#                 if i == j:
#                     continue
#                 if line[j] == 1:
#                     together_appear_dict[(i, j)] += 1
#
#     # 计算支持度，置信度，提升度
#     for k, v in together_appear_dict.items():
#         support_dict[k] = v / n_samples  #支持度
#         confidence_dict[k] = v / feature_num_dict[k[0]]  #置信度
#         lift_dict[k] = v * n_samples / \
#             (feature_num_dict[k[0]] * feature_num_dict[k[1]])
#         #提升度
#
#     return support_dict, confidence_dict, lift_dict
#     # 返回支持度，置信度，提升度

def convert_to_sample(feature_dict, s, c, l):

    print('=' * 50)
    # print(feature_dict)
    feature_mirror_dict = dict()
    for k, v in feature_dict.items():
        feature_mirror_dict[v] = k

    support_sample_li = [[feature_mirror_dict[i[0][0]],
                          feature_mirror_dict[i[0][1]], round(i[1], 3)] for i in s]
    confidence_sample_li = [[feature_mirror_dict[i[0][0]],
                             feature_mirror_dict[i[0][1]], round(i[1], 3)] for i in c]
    lift_sample_li = [[feature_mirror_dict[i[0][0]],
                       feature_mirror_dict[i[0][1]], round(i[1], 3)] for i in l]
    return support_sample_li, confidence_sample_li, lift_sample_li

def calculate_frequence(all_df):
    pre_data = all_df[all_df['年份'] < 2020]
    after_data = all_df[all_df['年份'] > 2019]
    dict_pre = dict(pre_data['产品名称'].value_counts())
    dict_after = dict(after_data['产品名称'].value_counts())

    def get_pre_frequency(s):
        fre = dict_pre[s]
        return fre

    def get_after_frequency(s):
        fre = dict_after[s]
        return fre

    pre_data['出现频次'] = pre_data['产品名称'].progress_apply(get_pre_frequency)
    after_data['出现频次'] = after_data['产品名称'].progress_apply(get_after_frequency)

    return  pre_data, after_data

def calculate_score( pre_data, after_data):
    pre_data['产品热度总分'] = 3 * pre_data['出现频次'] + 2 * pre_data['情感得分']
    after_data['产品热度总分'] = 3 * after_data['出现频次'] + 2 * after_data['情感得分']

    pre_data['产品热度'] = pre_data['产品热度总分'].div(np.sum(pre_data['产品热度总分']), axis=0)
    after_data['产品热度'] = after_data['产品热度总分'].div(np.sum(after_data['产品热度总分']), axis=0)

    pre_data_sort = pre_data.sort_values(by="产品热度", ascending=False).reset_index(drop=True)
    after_data_sort = after_data.sort_values(
        by="产品热度", ascending=False).reset_index(drop=True)

    return pre_data_sort, after_data_sort

def judge(pre_data_sort, after_data_sort):
    pre_data_sort['文本'] = pre_data_sort['文本'].progress_apply(clearTxt)
    pre_data_sort['产品类型判断文本'] = pre_data_sort['语料ID'] + ' ' + pre_data_sort['文本']

    pre_data_sort['产品类型'] = pre_data_sort['产品类型判断文本'].progress_apply(get_product_type)

    after_data_sort['文本'] = after_data_sort['文本'].progress_apply(clearTxt)
    after_data_sort['产品类型判断文本'] = after_data_sort['语料ID'] + \
                                  ' ' + after_data_sort['文本']

    after_data_sort['产品类型'] = after_data_sort['产品类型判断文本'].progress_apply(
        get_product_type)

    return pre_data_sort, after_data_sort