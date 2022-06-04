import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from utils import processed_travel, processed_news, stop_path, resolve_q1
from utils import get_topic_list, get_stop_words, get_topic_words

if __name__ == '__main__':
    # load processed data
    news = pd.read_csv(processed_travel)
    resolve_q1(news, "travel_res.csv", n_topics=5, threshold=0)
    # topic_list = get_topic_list()
    # stop_list = get_stop_words(stop_path)
    # travel_corpus = travel['text'].tolist()
    #
    # # word 2 vecter (count tf)
    # n_feature = 250
    # cntVector = CountVectorizer(strip_accents='unicode',
    #                             max_features=n_feature,
    #                             stop_words=stop_list,
    #                             max_df=0.5,
    #                             min_df=10)
    # cntTf = cntVector.fit_transform(travel_corpus)
    # print(cntTf)
    #
    # # build LDA model
    # n_topics = 6
    # lda = LatentDirichletAllocation(n_components=n_topics,
    #                                 max_iter=50,
    #                                 learning_method='batch',
    #                                 learning_offset=50.,
    #                                 # doc_topic_prior=0.1,
    #                                 # topic_word_prior=0.01,
    #                                 random_state=0)
    # docres = lda.fit_transform(cntTf)
    #
    # # get each topic words
    # n_top_words = 30
    # tf_feature_names = cntVector.get_feature_names_out()
    # topic_word = get_topic_words(lda, tf_feature_names, n_top_words)
    # print(topic_word)
    # topic_dict = {}
    # for t_id, topic in enumerate(topic_word):
    #     t_list = topic.split()
    #     t_count = 0
    #     for w in t_list:
    #         if w in topic_list:
    #             t_count += 1
    #     topic_dict[t_id] = t_count
    #
    # print(topic_dict)
    #
    #
    # # get each text topic (postprocess)
    # topics = lda.transform(cntTf)
    # topic = []
    # is_related = []
    # for t in topics:
    #     topic.append("Topic #"+str(list(t).index(np.max(t))))
    #     related_number = topic_dict[list(t).index(np.max(t))]
    #     is_related.append(1 if related_number > 0 else 0)
    # travel['is_realted'] = is_related
    # travel_res = pd.DataFrame(travel, columns=['ID', 'is_realted'])
    # travel_res.to_csv("dataset/results/travel_realte.csv", index=0)
    #
    #
    #
    #
    # import pyLDAvis
    # import pyLDAvis.sklearn
    #
    # pic = pyLDAvis.sklearn.prepare(lda, cntTf, cntVector)
    # pyLDAvis.display(pic)
    # pyLDAvis.save_html(pic, 'lda_pass' + str(8) + '.html')
    # pyLDAvis.display(pic)

