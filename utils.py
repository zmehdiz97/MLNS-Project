"""
    Understanding Image Retrieval Re-Ranking: A Graph Neural Network Perspective

    Xuanmeng Zhang, Minyue Jiang, Zhedong Zheng, Xiao Tan, Errui Ding, Yi Yang

    Project Page : https://github.com/Xuanmeng-Zhang/gnn-re-ranking

    Paper: https://arxiv.org/abs/2012.07620v2

    ======================================================================
   
    On the Market-1501 dataset, we accelerate the re-ranking processing from 89.2s to 9.4ms
    with one K40m GPU, facilitating the real-time post-processing. Similarly, we observe 
    that our method achieves comparable or even better retrieval results on the other four 
    image retrieval benchmarks, i.e., VeRi-776, Oxford-5k, Paris-6k and University-1652, 
    with limited time cost.
"""

import pickle
import numpy as np
import torch
from tabulate import tabulate
from collections import OrderedDict
import logging
from termcolor import colored


def load_pickle(pickle_path):
   with open(pickle_path, 'rb') as f:
       data = pickle.load(f)
       return data

def save_pickle(pickle_path, data):
    with open(pickle_path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

def pairwise_squared_distance(x):
    '''
    x : (n_samples, n_points, dims)
    return : (n_samples, n_points, n_points)
    '''
    x2s = (x * x).sum(-1, keepdim=True)
    return x2s + x2s.transpose(-1, -2) - 2 * x @ x.transpose(-1, -2)
    
def pairwise_distance(x, y):
    m, n = x.size(0), y.size(0)
    
    x = x.view(m, -1)
    y = y.view(n, -1)

    dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n,m).t()
    dist.addmm_(1, -2, x, y.t())

    return dist

def cosine_similarity(x, y):
    m, n = x.size(0), y.size(0)

    x = x.view(m, -1)
    y = y.view(n, -1)

    y = y.t()
    score = torch.mm(x, y)

    return score

def evaluate_ranking_list(indices, query_label, query_cam, gallery_label, gallery_cam):   
    CMC = np.zeros((len(gallery_label)), dtype=np.int)
    ap = 0.0

    for i in range(len(query_label)):
        ap_tmp, CMC_tmp = evaluate(indices[i],query_label[i], query_cam[i], gallery_label, gallery_cam)
        if CMC_tmp[0]==-1:
            continue
        CMC = CMC + CMC_tmp
        ap += ap_tmp      

    CMC = CMC.astype(np.float32)
    CMC = CMC/len(query_label) #average CMC
    print('Rank@1:%f Rank@5:%f Rank@10:%f mAP:%f'%(CMC[0],CMC[4],CMC[9],ap/len(query_label)))

def evaluate(index, ql,qc,gl,gc):
    query_index = np.argwhere(gl==ql)
    camera_index = np.argwhere(gc==qc)

    good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    junk_index1 = np.argwhere(gl==-1)
    junk_index2 = np.intersect1d(query_index, camera_index)
    junk_index = np.append(junk_index2, junk_index1) #.flatten())
    
    CMC_tmp = compute_mAP(index, good_index, junk_index)
    return CMC_tmp


def compute_mAP(index, good_index, junk_index):
    ap = 0
    cmc = np.zeros((len(index)), dtype=np.int)
    if good_index.size==0:   # if empty
        cmc[0] = -1
        return ap,cmc

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask==True)
    rows_good = rows_good.flatten()
    
    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0/ngood
        precision = (i+1)*1.0/(rows_good[i]+1)
        if rows_good[i]!=0:
            old_precision = i*1.0/rows_good[i]
        else:
            old_precision=1.0
        ap = ap + d_recall*(old_precision + precision)/2

    return ap, cmc

def print_csv_format(results):
    """
    Print main metrics in a format similar to Detectron2,
    so that they are easy to copypaste into a spreadsheet.
    Args:
        results (OrderedDict): {metric -> score}
    """
    # unordered results cannot be properly printed

    logger = logging.getLogger(__name__)

    dataset_name = results.pop('dataset')
    metrics = ["Dataset"] + [k for k in results]
    csv_results = [(dataset_name, *list(results.values()))]

    # tabulate it
    table = tabulate(
        csv_results,
        tablefmt="pipe",
        floatfmt=".2f",
        headers=metrics,
        numalign="left",
    )

    print("Evaluation results in csv format: \n" + colored(table, "cyan"))
def eval_metrics(dataset, indices, q_ids, g_ids, q_camids, g_camids, max_rank=10):
    """Evaluation of the queries output 
    """
    num_q, num_g = indices.shape

    if num_g < max_rank:
        max_rank = num_g
        print('Note: number of gallery samples is quite small, got {}'.format(num_g))

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    all_INP = []
    num_valid_q = 0  # number of valid query

    i=0
    for q_idx in range(num_q):
        # get query pid and camid
        q_id = q_ids[q_idx]
        q_camid = q_camids[q_idx]

        order = indices[q_idx]
        remove = (g_ids[order] == q_id) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        matches = (g_ids[order] == q_id).astype(np.int32)
        raw_cmc = matches[keep]  # binary vector, positions with value 1 are correct matches

        if not np.any(raw_cmc):
            # this condition is true when query identity does not appear in gallery
            i+=1
            continue

        cmc = raw_cmc.cumsum()

        pos_idx = np.where(raw_cmc == 1)
        max_pos_idx = np.max(pos_idx)
        inp = cmc[max_pos_idx] / (max_pos_idx + 1.0)
        all_INP.append(inp)

        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    print('number of queries that do not exist in the gallery :', i)
    assert num_valid_q > 0, 'Error: all query identities do not appear in gallery'

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q

    results = OrderedDict()
    results['dataset'] = dataset
    mAP = np.mean(all_AP)
    mINP = np.mean(all_INP)
    for r in [1, 5, 10]:
        results['Rank-{}'.format(r)] = all_cmc[r - 1] * 100
    results['mAP'] = mAP * 100
    results['mINP'] = mINP * 100
    results["metric"] = (mAP + all_cmc[0]) / 2 * 100

    print_csv_format(results)
    return all_cmc, all_AP, all_INP