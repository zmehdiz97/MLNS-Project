# MLNS-Final-Project

In this project, we explore several approaches to optimise target re-identification (re-ID) as a re-ranking problem. Our work consisted in trying different methods to re-rank the re-ID results. 

# Datasets

We will be testing our work on Market-1501 and VeriWild

# Requirements 

* dgl

# GNN based reranking
The code has been included in `/extension`. To compile it:

```shell
cd extension
sh make.sh
```
To run reranking evaluation:
1. Place dataset files under 'dataset/' folder:
The dataset structure should be like:

```bash
datasets/
    Market/
        camids.pkl
        feat.pkl
        ids.pkl
``` 
2. python run.py
Ps: Reranking runs only on GPU