import numpy as np 
import pickle 
import torch
import random
from utils import evaluate_ranking_list, eval_metrics
from gnn_reranking import gnn_reranking


dataset = 'Market'
k1, k2 = 27,6 

with open(f'datasets/{dataset}/feat.pkl', 'rb') as f :
    features = pickle.load(f)  # (19281, 2048) [:100,:]
with open(f'datasets/{dataset}/ids.pkl', 'rb') as f :
    pids = pickle.load(f) # 751 unique id [:100]
with open(f'datasets/{dataset}/camids.pkl', 'rb') as f :
    camids = pickle.load(f)  # 6 unique camids [:100]


print('number of unique ids: ', len(np.unique(pids)))
number_ids = 350
selected_ids = np.unique(pids)[:number_ids]
print(f'Using a subsample of {number_ids} ids')
def select(id):
    return id in selected_ids

selector = list(map(select, pids))
print('samples', sum(selector))
features = features[selector,:]
pids = pids[selector]
camids = camids[selector]

features = torch.from_numpy(features)
pids = torch.from_numpy(pids)
camids = torch.from_numpy(camids)

random.seed(10)
gallery_mask = random.sample(range(len(pids)), k=int(round(len(pids)*0.7)))
query_mask = list(set(range(len(pids))) - set(gallery_mask))

query_pids = pids[query_mask]
query_features = features[query_mask, :]
query_camids = camids[query_mask]

gallery_pids = pids[gallery_mask]
gallery_features = features[gallery_mask, :]
gallery_camids = camids[gallery_mask]

query_features = query_features.cuda()
gallery_features = gallery_features.cuda()
print(query_features.shape, gallery_features.shape)
indices = gnn_reranking(query_features, gallery_features, k1, k2)
#evaluate_ranking_list(indices, query_pids, query_camids, gallery_pids, gallery_camids)
query_pids, gallery_pids, query_camids, gallery_camids = \
    query_pids.cpu().numpy(), gallery_pids.cpu().numpy(),\
    query_camids.cpu().numpy(), gallery_camids.cpu().numpy()
eval_metrics(dataset, indices, query_pids, gallery_pids, query_camids, gallery_camids, max_rank=10)