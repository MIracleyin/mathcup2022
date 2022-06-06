import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from utils.question1 import resolve_q1
from utils.paths import processed_news, processed_travel

if __name__ == '__main__':
    # load processed data
    # news = pd.read_csv(processed_news)
    # resolve_q1(news, "news_res_test.csv", n_topics=5, threshold=0)
    #
    # travel = pd.read_csv(processed_travel)
    # resolve_q1(travel, "travel_res_test.csv", n_topics=5, threshold=0)

    news = pd.read_csv("dataset/results/news_res.csv")
    news['文章ID'] = news['文章ID'].apply(lambda x: "公众号推文" + str(x))
    travel = pd.read_csv("dataset/results/travel_res.csv")
    travel['文章ID'] = travel['文章ID'].apply(lambda x: "旅游攻略" + str(x))
    res1 = pd.concat([travel, news], axis=0)
    res1['分类标签'] = res1['分类标签'].apply(lambda x:"相关" if x == 1 else "不相关")
    res1.to_csv("dataset/results/result1.csv", index=False)
