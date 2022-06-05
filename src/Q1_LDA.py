import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from utils.question1 import resolve_q1
from utils.paths import processed_news, processed_travel

if __name__ == '__main__':
    # load processed data
    news = pd.read_csv(processed_news)
    resolve_q1(news, "news_res_test.csv", n_topics=5, threshold=0)

    travel = pd.read_csv(processed_travel)
    resolve_q1(travel, "travel_res_test.csv", n_topics=5, threshold=0)

