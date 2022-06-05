import os
import pandas as pd
import torch
import numpy as np

from utils.paths import processed_path
from utils.question3 import get_entity_id_relate, get_coo_mat_feat
from transformers import AutoTokenizer, AutoModel
from scipy.sparse import save_npz, load_npz
import jsonlines
import json
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt

if __name__ == '__main__':
    all_df = pd.read_csv(os.path.join(processed_path, "all.csv")).dropna()
    # def create_one_hot(data):
    #     data['all_text'] = data['text'] + ' ' + data['product']
    #     all_feature_li = data['product']
    #     all_feature_set_li = list(set(all_feature_li))
    #     # 将所有样本编码成0-1数据形式
    #     feature_dict = defaultdict(int)
    #     for n, feat in enumerate(all_feature_set_li):
    #         feature_dict[feat] = n
    #     # print(feature_dict)
    #
    #     out_li = list()
    #     data_emb = {}
    #     for j in range(len(data)):
    #         text = data.iloc[j]['all_text']
    #         feature_num_li = []
    #         for i, f in enumerate(feature_dict):
    #             if f in text:
    #                 feature_num_li.append(feature_dict[f])
    #         inner_li = [1 if num in feature_num_li else 0 for num in range(
    #             len(all_feature_set_li))]
    #
    #         out_li.append(inner_li)
    #     out_array = np.array(out_li)
    #     return out_array, feature_dict
    #
    # data_array, feature_di = create_one_hot(all_df)

    id2e, e2id = get_entity_id_relate(all_df)
    # coo_mat, id2emb = get_coo_mat_feat(all_df, id2e, e2id)
    # save_npz(os.path.join(processed_path, "coo.npz"), coo_mat)
    #
    # with jsonlines.open(os.path.join(processed_path, "id2emb.jsonl"), 'a') as f:
    #     f.write(id2emb)
    coo_mat = load_npz(os.path.join(processed_path, "coo.npz"))
    with open(os.path.join(processed_path, "id2emb.jsonl"), 'r') as f:
        id2emb = json.load(f)
    res2_2 = pd.read_csv('dataset/results/result2-2.csv')
    p_k = res2_2['产品ID']
    p_v = res2_2['产品类型']
    p_type_dict = dict(zip(p_k, p_v))

    res = []
    for c, r in zip(coo_mat.col, coo_mat.row):
        productID1 = all_df.iloc[c]['productID']
        productID2 = all_df.iloc[r]['productID']
        type = p_type_dict[productID1] + '-' + p_type_dict[productID2]
        c, r = [id2emb[str(c)]], [id2emb[str(r)]]
        score = cosine_similarity(np.array(c), np.array(r))[0][0]
        res.append((productID1, productID2, score, type))

    res_pd = pd.DataFrame(res)
    res_pd.columns = ['产品1ID', '产品2ID', '关联度', '关联类型']
    res_pd.to_csv('./dataset/results/result3.csv', index=False, encoding='utf_8_sig')

    G = nx.from_pandas_edgelist(res_pd[res_pd['关联度'] > 0.9], "产品1ID", "产品2ID",
                                edge_attr=True, create_using=nx.MultiDiGraph())
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 替换sans-serif字体
    plt.figure(figsize=(15, 15))
    pos = nx.spring_layout(G, k=0.8)  # k为节点间距离
    nx.draw(G, with_labels=True, node_color='skyblue',
            node_size=1500, edge_cmap=plt.cm.Greens, pos=pos)
    plt.savefig('./dataset/results/vis.png', dpi=300)
