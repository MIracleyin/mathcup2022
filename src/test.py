# -*- coding: utf-8 -*-
# @Filename: test
# @Date: 2022-06-04 21:13
# @Author: Leo Xu
# @Email: leoxc1571@163.com

import pandas as pd
import os.path
import sys

# print(sys.path)


before_covid = pd.read_csv('dataset/results/疫情前产品热度.csv', header=0)
after_covid = pd.read_csv('dataset/results/疫情后产品热度.csv', header=0)


emotion_b = before_covid.groupby('产品名称').agg({'情感得分': 'mean'}).reset_index()
fre_b = before_covid.groupby('产品名称').agg({'出现频次': 'mean'}).reset_index()
hot_b = before_covid.groupby('产品名称').agg({'产品热度': 'mean'}).reset_index()

before = pd.merge(emotion_b, fre_b, on='产品名称', how='left')
before = pd.merge(before, hot_b, on='产品名称', how='left')

emotion_a = after_covid.groupby('产品名称').agg({'情感得分': 'mean'}).reset_index()
fre_a = after_covid.groupby('产品名称').agg({'出现频次': 'mean'}).reset_index()
hot_a = after_covid.groupby('产品名称').agg({'产品热度': 'mean'}).reset_index()

after = pd.merge(emotion_a, fre_a, on='产品名称', how='left')
after = pd.merge(after, hot_a, on='产品名称', how='left')

print('Done')