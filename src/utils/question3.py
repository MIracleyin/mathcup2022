import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from transformers import AutoTokenizer, AutoModel


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
