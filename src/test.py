# -*- coding: utf-8 -*-
# @Filename: test
# @Date: 2022-06-04 21:13
# @Author: Leo Xu
# @Email: leoxc1571@163.com

import pandas as pd
import os.path
import sys

print(sys.path)


dataset_part1_path = "dataset/dataset-2018-2019.xlsx"
dataset_part2_path = "dataset/dataset-2020-2021.xlsx"
processed_path = "dataset/processed"
sys.path.append(dataset_part1_path)
sys.path.append(dataset_part2_path)
sys.path.append(processed_path)
dataset_part1_travel = pd.read_excel(dataset_part1_path, sheet_name='游记攻略')
dataset_part1_news = pd.read_excel(dataset_part1_path, sheet_name='微信公众号新闻')
dataset_part2_travel = pd.read_excel(dataset_part2_path, sheet_name='游记攻略')
dataset_part2_news = pd.read_excel(dataset_part2_path, sheet_name='微信公众号新闻')

news_res = pd.read_csv('dataset/results/news_res.csv', header=0)
travel_res = pd.read_csv('dataset/results/travel_res.csv', header=0)
news_res.rename(columns={'ID': '文章ID'}, inplace=True)
travel_res.rename(columns={'ID': '游记ID'}, inplace=True)
# concat part1 and part2
dataset_travel = pd.concat([dataset_part1_travel, dataset_part2_travel], axis=0)
dataset_news = pd.concat([dataset_part1_news, dataset_part2_news], axis=0)


dataset_travel = pd.merge(dataset_travel, travel_res, on='游记ID', how='left')
dataset_news = pd.merge(dataset_news, news_res, on='文章ID', how='left')

news_related = dataset_news[dataset_news.is_realted==1]
travel_related = dataset_travel[dataset_travel.is_realted==1]

news_related.to_csv('news_related.csv', encoding="utf_8_sig")
travel_related.to_csv('travel_related.csv', encoding="utf_8_sig")

print('Done')