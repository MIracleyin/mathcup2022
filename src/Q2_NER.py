import os.path

import numpy as np
import pandas as pd

from utils import get_class_infors, modify_infors, get_all_infors
from utils import processed_path,get_sentiment, get_frequence, get_final_socre

if __name__ == '__main__':
    # hotel, scenic, travel, food, news = get_class_infors()
    # hotel, scenic, travel, food, news = modify_infors(hotel, scenic, travel, food, news)
    # all_df = get_all_infors(hotel, scenic, travel, food, news)
    # all_df.to_csv(os.path.join(processed_path, "all.csv"), index=0)
    print("------------------save to csv-------------------------")
    all_df = pd.read_csv(os.path.join(processed_path, "all.csv"))
    result2_1 = pd.DataFrame(all_df, columns=['curpusID', 'productID', 'product'])
    result2_1.columns = ['语料ID', '产品ID','产品名称']
    result2_1.to_csv('./dataset/results/result2-1.csv', index=False, encoding='utf_8_sig')
    # text =  np.array(all_df['text'])[None,:].transpose(1,0).tolist()
    text = all_df['text'].values.tolist()
    # all_df['emotion_score'] = all_df['text'].apply(get_sentiment)
    all_df['emotion_score'] = get_sentiment(text)
    # all_df['emotion_score'] = 0.5
    year_2018_count, year_2019_count, year_2020_count, year_2021_count = get_frequence(all_df)
    result2_2 = get_final_socre(year_2018_count, year_2019_count, year_2020_count, year_2021_count)
    result2_2.to_csv('./dataset/results/result2-2.csv', index=False, encoding='utf_8_sig')


