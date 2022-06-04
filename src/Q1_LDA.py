import os.path

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from utils import processed_travel, processed_news, stop_path, resolve_q1
from utils import get_topic_list, get_stop_words, get_topic_words
from utils import raw_part1_path, raw_part2_path, results_path

if __name__ == '__main__':
    # load processed data
    # news = pd.read_csv(processed_news)
    # resolve_q1(news, "news_res.csv", n_topics=8, threshold=0)

    dataset_travel_1 = pd.read_excel(raw_part1_path, sheet_name='游记攻略')
    dataset_travel_2 = pd.read_excel(raw_part2_path, sheet_name='游记攻略')
    dataset_travel = pd.concat([dataset_travel_1, dataset_travel_2], axis=0)
    dataset_news_1 = pd.read_excel(raw_part1_path, sheet_name='微信公众号新闻')
    dataset_news_2 = pd.read_excel(raw_part2_path, sheet_name='微信公众号新闻')
    dataset_news = pd.concat([dataset_news_1, dataset_news_2], axis=0)


    filter_travel = pd.read_csv(os.path.join(results_path, "travel_res.csv"))
    filter_news = pd.read_csv(os.path.join(results_path, "news_res.csv"))
    travel = pd.concat([dataset_travel, filter_travel], axis=1)
    travel = travel[travel['is_relate'] == 1]
    news = pd.concat([dataset_news, filter_news], axis=1)
    news = news[news['is_relate'] == 1]


    print(dataset_news_2)





