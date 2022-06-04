import re
import jieba
import pandas as pd
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from tqdm import tqdm
from utils import *

warnings.filterwarnings('ignore')
tqdm.pandas()


class Methods:

    def __init__(self, data_2018, data_2020, stopwords, baked_data):

        self.data_2018 = data_2018
        self.data_2020 = data_2020
        self.stopwords = stopwords
        self.baked_data = baked_data

    def first(self):
        print("------------------------------------first------------------------------------------------")
        stop_words = open(self.stopwords, 'r', encoding='utf-8').read()
        stop_words = stop_words.encode('utf-8').decode('utf-8-sig')  # 列表头部\ufeff处理
        stop_words = stop_words.split('\n')  # 根据分隔符分隔

        otherwords = ['茂名', '茂名市', '年', '月', '日']
        stop_words.extend(otherwords)

        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)

        print('样本预测数据加载中...')
        data_2018_wx = pd.read_excel(self.data_2018, sheet_name='微信公众号新闻', header=0)
        data_2020_wx = pd.read_excel(self.data_2020, sheet_name='微信公众号新闻', header=0)
        data_all_wx = data_2018_wx.append(data_2020_wx, ignore_index=True)
        print('加载完成')

        print('样本预测数据转换中...')
        data_all_wx['文章'] = data_all_wx['公众号标题'] + data_all_wx['正文']
        data_all_wx = data_all_wx[['文章ID', '文章']].dropna()
        data_last_wx = data_all_wx['文章'].apply(lambda x: clean_text(x)).apply(
            lambda x: " ".join([w for w in list(jieba.cut(x)) if w not in stop_words])
        )
        # 将表格做完处理后保存
        data_last_wx.to_excel('./result/fenci.xlsx', encoding='utf_8_sig')
        print('转换完成')

        print('载入训练数据....')
        data = pd.read_csv(self.baked_data, encoding='gb18030')  # 载入训练数据
        labels = data['labels']  # 设置标签
        features = data['massages'].astype(str).apply(
            lambda x: " ".join([w for w in list(jieba.cut(x)) if w not in stop_words]))  # 设置样本特征
        # 20%作为测试集，其余作为训练集
        train_x, test_x, train_y, test_y = train_test_split(features, labels, test_size=0.20, stratify=labels,
                                                            random_state=1)
        print('计算tfidf特征...')
        # 计算单词权重
        tf = TfidfVectorizer(stop_words=stop_words, max_df=0.5)

        train_features = tf.fit_transform(train_x)
        # 上面fit过了，这里transform
        test_features = tf.transform(test_x)
        predict_features = tf.transform(data_last_wx)
        print('计算完成')

        # 多项式贝叶斯分类器
        print('训练分类器...')
        clf = MultinomialNB(alpha=0.01).fit(train_features, train_y)
        predicted_labels = clf.predict(test_features)
        print('训练完成')

        # 计算准确率
        print('准确率为：', metrics.accuracy_score(test_y, predicted_labels))

        print('预测样本数据...')
        predict = clf.predict(predict_features)
        print(predict)

        print('预测完成')
        data_ = data_all_wx.values

        count = 0
        for i in data_:
            i[1] = predict[count]
            count += 1
        # print(data_)
        columes = ['文章ID', '分类标签']
        df = pd.DataFrame(data_, columns=columes)
        df.to_csv('./result/result1.csv', encoding='utf_8_sig', index=False)
        print('done')


    def second(self):
        print("------------------------------------second------------------------------------------------")
        Hotel_Infos, Scenic_Infos, Travel_Infos, Dining_Infos, Wechat_Infos = load_infos(self.data_2018, self.data_2020)
        Hotel_Infos, Scenic_Infos, Travel_Infos, Dining_Infos, Wechat_Infos = modify_infos(Hotel_Infos,
                                                                                           Scenic_Infos,
                                                                                           Travel_Infos,
                                                                                           Dining_Infos,
                                                                                           Wechat_Infos)

        all_df = prepare_all(Hotel_Infos, Scenic_Infos, Travel_Infos, Dining_Infos)
        result2 = all_df[['语料ID', '产品ID', '产品名称']]

        result2.to_csv('./result/result2-1.csv', index=False, encoding='utf_8_sig')
        all_df.to_csv('./result/result_all.csv', index=False, encoding='utf_8_sig')

        all_df['情感得分'] = all_df['文本'].progress_apply(emotion_score)

        result2_2 = hot_score(all_df)
        result2_2.to_csv('./result/result2-2.csv', index=False, encoding='utf_8_sig')

    def thrid(self):
        print("------------------------------------third------------------------------------------------")
        data = pd.read_csv('./result/result_all.csv')
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
        # submit_3

        map_dict = {}
        for i, d in enumerate(feature_di):
            map_dict[d] = 'ID' + str(feature_di[d] + 1)
        # map_dict

        submit_3['产品1'] = submit_3['产品名称1'].map(map_dict)
        submit_3['产品2'] = submit_3['产品名称2'].map(map_dict)
        result3 = submit_3[['产品1', '产品2', '关联度']]
        # result3

        result2_2 = pd.read_csv('./result/result2-2.csv')
        p_k = result2_2['产品ID']
        p_v = result2_2['产品类型']
        p_type_dict = dict(zip(p_k, p_v))
        # p_type_dict

        p_type = []
        for i in range(len(result3)):
            id_1 = result3.iloc[i]['产品1']
            id_2 = result3.iloc[i]['产品2']
            p_type.append(p_type_dict[id_1] + '-' + p_type_dict[id_2])
        # p_type

        result3['关联类型'] = p_type
        result3.to_csv('./result/result3.csv', index=False, encoding='utf_8_sig')

        G = nx.from_pandas_edgelist(submit_3[submit_3['关联度'] > 0.05], "产品名称1", "产品名称2",
                                    edge_attr=True, create_using=nx.MultiDiGraph())

        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号 #有中文出现的情况，需要u'内容'
        plt.figure(figsize=(15, 15))
        pos = nx.spring_layout(G, k=0.8)  # k为节点间距离
        nx.draw(G, with_labels=True, node_color='skyblue',
                node_size=1500, edge_cmap=plt.cm.Greens, pos=pos)  # 画图，定义颜色,标签，节点大小
        plt.savefig('./result/产品关联知识图谱3.png', dpi=300)

    def fourth(self):
        print("------------------------------------fourth------------------------------------------------")
        all_df = pd.read_csv('./result/result_all.csv')
        all_df['情感得分'] = all_df['文本'].progress_apply(emotion_score)
        # 计算产品热度
        pre_data, after_data = calculate_frequence(all_df)
        # 计算综合得分
        pre_data_sort, after_data_sort = calculate_score(pre_data, after_data)
        # 判断产品类型
        pre_data_sort, after_data_sort = judge(pre_data_sort, after_data_sort)
        pre_data_sort.to_csv('./result/before.csv', index=False, encoding='utf_8_sig')
        after_data_sort.to_csv('./result/after.csv', index=False, encoding='utf_8_sig')


if __name__ == '__main__':
    data_2018 = './datasets/dataset_2018-2019.xlsx'
    data_2020 = './datasets/dataset_2020-2021.xlsx'
    stopwords = './datasets/stopwords.txt'
    baked_data = './datasets/baked_data.csv'
    methods = Methods( data_2018, data_2020, stopwords, baked_data)
    methods.first()
    methods.second()
    methods.thrid()
    methods.fourth()