import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from transformers import AutoTokenizer, AutoModel
import jieba
import re
import networkx as nx


def create_one_hot(df):
    df['all_text'] = df['product'] + "" + df['text']
    all_feature_li = df['产品名称']
    all_feature_set_li = list(set(all_feature_li))
    # 将所有样本编码成0-1数据形式
    feature_dict = defaultdict(int)
    for n, feat in enumerate(all_feature_set_li):
        feature_dict[feat] = n
    # print(feature_dict)

    out_li = list()
    for j in range(len(df)):
        text = df.iloc[j]['all_text']
        feature_num_li = []
        for i, f in enumerate(feature_dict):
            if f in text:
                feature_num_li.append(feature_dict[f])
        inner_li = [1 if num in feature_num_li else 0 for num in range(
            len(all_feature_set_li))]

        out_li.append(inner_li)
    out_array = np.array(out_li)
    return out_array, feature_dict


def get_entity_id_relate(df):
    id2entity = list(set(df['product'].values.tolist()))
    entity2id = defaultdict(int)

    for id, feat in enumerate(id2entity):
        entity2id[feat] = id

    return id2entity, entity2id


def get_coo_mat_feat(df, id2e, e2id):
    df['all_text'] = df['product'] + "" + df['text']

    tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
    model = AutoModel.from_pretrained('bert-base-chinese').to('cuda')

    relate_list = list()
    id2emb = {}
    for i, j in enumerate(range(len(df))):
        row_text = df.iloc[j]['all_text']
        row_emb = get_embeddings(tokenizer, model, row_text)
        id2emb[i] = row_emb
        entity_id_list = []
        for i, f in enumerate(e2id):
            if f in row_text:
                entity_id_list.append(e2id[f])
        inner_list = [1 if num in entity_id_list else 0 for num in range(len(id2e))]
        relate_list.append(inner_list)
    out_array = np.array(relate_list)
    plt.matshow(out_array)
    plt.show()
    coo_mat = coo_matrix(out_array)

    return coo_mat, id2emb


def get_embeddings(tokenizer, model, text):
    input = tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=512).to('cuda')
    res = model(**input)
    emb = res.last_hidden_state[:, 0, :].detach().to('cpu')
    emb = emb.numpy().tolist()[0]
    return emb

def main():
    # 读取已经提取出产品的文件
    data = pd.read_csv('src/dataset/processed/all.csv').dropna()

    # 对样本进行one-hot编码
    def create_one_hot(data):
        data['所有文本'] = data['文本'] + ' ' + data['产品名称']
        all_feature_li = data['产品名称']
        all_feature_set_li = list(set(all_feature_li))
        # 将所有样本编码成0-1数据形式
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

    # 以one-hot形式存储数据

    # 计算相关度
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

        # 计算共同出现的特征次数
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
            support_dict[k] = v / n_samples  # 支持度
            confidence_dict[k] = v / feature_num_dict[k[0]]  # 置信度
            lift_dict[k] = v * n_samples / \
                           (feature_num_dict[k[0]] * feature_num_dict[k[1]])
            # 提升度

        return support_dict, confidence_dict, lift_dict
        # 返回支持度，置信度，提升度


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

    data_array, feature_di = create_one_hot(data)
    support_di, confidence_di, lift_di = calculate(data_array)

    support = sorted(support_di.items(), key=lambda x: x[1], reverse=True)
    confidence = sorted(confidence_di.items(),
                        key=lambda x: x[1], reverse=True)
    lift = sorted(lift_di.items(), key=lambda x: x[1], reverse=True)

    support_li, confidence_li, lift_li = convert_to_sample(feature_di, support, confidence, lift)

    support_df = pd.DataFrame(support_li, columns=['产品名称1', '产品名称2', '支持度'])
    confidence_df = pd.DataFrame(confidence_li, columns=['产品名称1', '产品名称2', '置信度'])
    lift_df = pd.DataFrame(lift_li, columns=['产品名称1', '产品名称2', '提升度'])
    submit_3 = support_df.copy()

    submit_3['关联度'] = support_df['支持度'] * 10 + confidence_df['置信度'] * 0.00004 + lift_df['提升度'] * 0.00006
    del submit_3['支持度']
    submit_3

    map_dict = {}
    for i, d in enumerate(feature_di):
        map_dict[d] = 'ID' + str(feature_di[d] + 1)
    map_dict

    submit_3['产品1'] = submit_3['产品名称1'].map(map_dict)
    submit_3['产品2'] = submit_3['产品名称2'].map(map_dict)
    result3 = submit_3[['产品1', '产品2', '关联度']]
    result3

    result2_2 = pd.read_csv('./data/result2-2.csv')
    p_k = result2_2['产品ID']
    p_v = result2_2['产品类型']
    p_type_dict = dict(zip(p_k, p_v))
    p_type_dict

    p_type = []
    for i in range(len(result3)):
        id_1 = result3.iloc[i]['产品1']
        id_2 = result3.iloc[i]['产品2']
        p_type.append(p_type_dict[id_1] + '-' + p_type_dict[id_2])
    p_type

    result3['关联类型'] = p_type
    result3.to_csv('./data/result3.csv', index=False)

    G = nx.from_pandas_edgelist(submit_3[submit_3['关联度'] > 0.05], "产品名称1", "产品名称2",
                                edge_attr=True, create_using=nx.MultiDiGraph())

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 替换sans-serif字体
    plt.figure(figsize=(15, 15))
    pos = nx.spring_layout(G, k=0.8)  # k为节点间距离
    nx.draw(G, with_labels=True, node_color='skyblue',
            node_size=1500, edge_cmap=plt.cm.Greens, pos=pos)  # 画图，定义颜色,标签，节点大小
    plt.savefig('./img/产品关联知识图谱3.png', dpi=300)


if __name__ == '__main__':
    main()