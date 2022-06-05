import os
import numpy as np
import pandas as pd
import pyLDAvis
import pyLDAvis.sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from .base import get_stop_words, get_topic_words, get_topic_list
from .paths import stop_path

def resolve_q1(process_data: pd.DataFrame, save_path: str,
               is_vis=True,
               n_topics=5,
               n_topic_words=30,
               threshold=0,
               n_feature=1000,
               ):
    assert 'ID' in process_data.columns
    assert 'text' in process_data.columns
    process_data = process_data.dropna()
    # get data
    corpus = process_data['text'].tolist()

    # transform to vector by cnt
    cnt_vector = CountVectorizer(strip_accents='unicode',
                                 # max_features=n_feature,
                                 stop_words=get_stop_words(stop_path),
                                 max_df=0.5,
                                 min_df=10)
    cnt_tf = cnt_vector.fit_transform(corpus)

    # build LDA model
    lda = LatentDirichletAllocation(n_components=n_topics,
                                    max_iter=50,
                                    learning_method='batch',
                                    learning_offset=50.,
                                    # doc_topic_prior=0.1,
                                    # topic_word_prior=0.01,
                                    random_state=0)
    docs_res = lda.fit_transform(cnt_tf)

    # get each topic words
    tf_feature_names = cnt_vector.get_feature_names_out()
    topic_word = get_topic_words(lda, tf_feature_names, n_topic_words)
    print(topic_word)

    # get topic count(appare in topic_list) map
    topic_list = get_topic_list()
    topic_count_dict = {}
    for t_id, topic in enumerate(topic_word):
        t_list = topic.split()
        t_count = 0
        for w in t_list:
            if w in topic_list:
                t_count += 1
        topic_count_dict[t_id] = t_count
    print(topic_count_dict)

    # get results
    topics = lda.transform(cnt_tf)  # 200 [(200,6)]
    is_related = []
    for t in topics:  # [1, 6]
        related_number = topic_count_dict[list(t).index(np.max(t))]
        is_related.append(1 if related_number > threshold else 0)  #
    process_data['is_realted'] = is_related
    process_data = pd.DataFrame(process_data, columns=['ID', 'is_realted'])
    process_data.to_csv(os.path.join("dataset/results", save_path), index=0)

    if is_vis:
        pic = pyLDAvis.sklearn.prepare(lda, cnt_tf, cnt_vector)
        pyLDAvis.display(pic)
        pyLDAvis.save_html(pic, 'lda_pass' + str(n_topics) + '.html')
        pyLDAvis.display(pic)
