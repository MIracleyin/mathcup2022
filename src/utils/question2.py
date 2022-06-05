import pandas as pd
import numpy as np
from .paths import raw_part1_path, raw_part2_path
from .base import clean_textv2
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
import tqdm


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
    # travel = pd.concat([dataset_travel_1, dataset_travel_2], axis=0)
    food = pd.concat([dataset_food_1, dataset_food_2], axis=0)
    # news = pd.concat([dataset_news_1, dataset_news_2], axis=0)
    travel = pd.read_csv("dataset/processed/travel_related.csv")
    news = pd.read_csv("dataset/processed/news_related.csv")

    return hotel, scenic, travel, food, news


def get_all_infors(hotel, scenic, travel, food, news):
    df = pd.concat([hotel, scenic, travel, food, news], axis=0)
    df = df.dropna().drop_duplicates(['product'])
    product_id = ['ID' + str(i + 1) for i in range(len(df))]
    df['productID'] = product_id
    return df


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


def get_ner_infer(text):
    # nlp task model
    # tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
    # model = AutoModelForTokenClassification.from_pretrained('ckiplab/bert-base-chinese-ner')
    p = "黑龙江省、辽宁省、吉林省、河北省、河南省、湖北省、湖南省、山东省、山西省、陕西省、安徽省、浙江省、江苏省、福建省、广东省、海南省、四川省、云南省、贵州省、青海省、甘肃省、江西省、台湾省".split('、')
    p2 = "黑龙江省、辽宁省、吉林省、河北省、河南省、湖北省、湖南省、山东省、山西省、陕西省、安徽省、浙江省、江苏省、福建省、广东省、海南省、四川省、云南省、贵州省、青海省、甘肃省、江西省、台湾省".replace("省",
                                                                                                                "").split(
        '、')
    c = "珠海市、汕头市、佛山市、韶关市、湛江市、肇庆市、江门市、茂名市、惠州市、梅州市、汕尾市、河源市、阳江市、清远市、东莞市、中山市、潮州市、揭阳市、云浮市".split('、')
    c2 = "珠海市、汕头市、佛山市、韶关市、湛江市、肇庆市、江门市、茂名市、惠州市、梅州市、汕尾市、河源市、阳江市、清远市、东莞市、中山市、潮州市、揭阳市、云浮市".replace("市", "").split('、')
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
        output = model(torch.tensor([tokenizer.encode(text, max_length=512, truncation=True)]))
        output = torch.nn.functional.softmax(output.logits, dim=-1).tolist()[0]
        if output[0] > output[1]:
            score = - output[0]
        elif output[0] < output[1]:
            score = output[1]
        else:
            score = 0
        outputs.append(round(score, 4))
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

    year_2018_count['final_hot'] = year_2018_count['hot_score'].div(np.sum(year_2018_count['hot_score']),
                                                                    axis=0)  # 化成一个小数 加起来为1
    year_2019_count['final_hot'] = year_2019_count['hot_score'].div(np.sum(year_2019_count['hot_score']), axis=0)
    year_2020_count['final_hot'] = year_2020_count['hot_score'].div(np.sum(year_2020_count['hot_score']), axis=0)
    year_2021_count['final_hot'] = year_2021_count['hot_score'].div(np.sum(year_2021_count['hot_score']), axis=0)

    year_2018 = year_2018_count.sort_values(by="final_hot", ascending=False).reset_index(drop=True)
    year_2019 = year_2019_count.sort_values(by="final_hot", ascending=False).reset_index(drop=True)
    year_2020 = year_2020_count.sort_values(by="final_hot", ascending=False).reset_index(drop=True)
    year_2021 = year_2021_count.sort_values(by="final_hot", ascending=False).reset_index(drop=True)

    product_hot_score_sort = pd.concat([year_2018, year_2019, year_2020, year_2021], axis=0)
    # product_hot_score['文本'] = product_hot_score['文本'].progress_apply(clearTxt)
    product_hot_score_sort['productJudge'] = product_hot_score_sort['curpusID'] + ' ' + product_hot_score_sort['text']
    product_hot_score_sort['productClass'] = product_hot_score_sort['productJudge'].apply(get_product_type)
    # 去除重复的产品
    product_hot_score_sort = product_hot_score_sort.drop_duplicates(['product'])
    # 产品 ID 产品类型 产品名称 产品热度 年份
    result2_2 = product_hot_score_sort[['productID', 'productClass', 'product', 'final_hot', 'years']]
    result2_2['final_hot'] = result2_2['final_hot'].round(4)
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
