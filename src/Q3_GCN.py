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

if __name__ == '__main__':
    all_df = pd.read_csv(os.path.join(processed_path, "all.csv")).dropna()
    id2e, e2id = get_entity_id_relate(all_df)
    # coo_mat, id2emb = get_coo_mat_feat(all_df, id2e, e2id)
    # save_npz(os.path.join(processed_path, "coo.npz"), coo_mat)
    #
    # with jsonlines.open(os.path.join(processed_path, "id2emb.jsonl"), 'a') as f:
    #     f.write(id2emb)
    coo_mat = load_npz(os.path.join(processed_path, "coo.npz"))
    with open(os.path.join(processed_path, "id2emb.jsonl"), 'r') as f:
        id2emb = json.load(f)

    score = []
    for c, r in zip(coo_mat.col, coo_mat.row):
        c, r = [id2emb[str(c)]], [id2emb[str(r)]]

        score.append(cosine_similarity(np.array(c),np.array(r))[0][0])
    print(score)



