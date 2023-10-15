import networkx as nx
import numpy as np
import pickle as pkl
import scipy.sparse as sp
import sys
import pandas as pd
import torch
import anndata as ad
import scanpy as sc


def create_adjacency(feature="./data/Adult-Brain_Lake_dge.txt",preprocessing=True):
    g = nx.read_edgelist('./data/adjacency_edgelist.txt',delimiter='\t')
    nodeset = sorted(set(g.nodes()))
    adj = nx.adjacency_matrix(g,nodelist=nodeset)
    prefix=feature.split("/")[-1].split(".")[-1]
    if prefix=="h5ad":
        print("Start reading h5ad file......")
        raw_fea=sc.read_h5ad(feature)
        if preprocessing:
            raw_fea = preprocessing_features(raw_fea,normalize=True,log=True,scale=True)
        else:
            raw_fea = preprocessing_features(raw_fea,normalize=False,log=False,scale=False)
    elif prefix=="csv" or prefix =="txt":
        print("Start reading csv or txt files......")
        raw_fea = pd.read_csv(feature,sep=",",header=0,index_col=0)
        if preprocessing:
            df_t = raw_fea.T
            var = pd.DataFrame(index=df_t.columns)
            obs = pd.DataFrame(index=df_t.index)
            adata=ad.AnnData(df_t.values,obs=obs,var=var,dtype='float32')
            raw_fea = preprocessing_features(adata,normalize=True,log=True,scale=True)
    else:
        print("Start reading 10x_mtx files......")
        raw_fea=sc.read_10x_mtx(feature)
        if preprocessing:
            raw_fea = preprocessing_features(raw_fea,normalize=True,log=True,scale=True)
        else:
            print("preprocessing is False.")
            raw_fea = preprocessing_features(raw_fea,normalize=False,log=False,scale=False)

    if(raw_fea.index.duplicated().sum()>0):
        print("features exist duplication!!!")
        for i in raw_fea.index[raw_fea.index.duplicated()]:
            new=raw_fea.loc[[i]].apply(sum)
            raw_fea = raw_fea.drop(index=i)
            raw_fea.loc[i]=new.values

    raw_fea = raw_fea.drop(set(raw_fea.index)-set(g.nodes()))
    nfea = len(set(g.nodes())-set(raw_fea.index))
    addfeature = pd.DataFrame(np.zeros((nfea,raw_fea.shape[1])),index=g.nodes-set(raw_fea.index),columns=raw_fea.columns)
    raw_fea = raw_fea.append(addfeature)
    raw_fea = raw_fea.sort_index(axis=0) 

    features = raw_fea.values
    features = sp.csr_matrix(features.astype(int))
    print(features)

    return adj,features

def preprocessing_features(raw_fea,normalize=True,log=True,scale=True):
    adata = raw_fea
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    print(adata)
    if normalize:
        sc.pp.normalize_total(adata, exclude_highly_expressed=True, target_sum=1e6) #High-count filtering CPM normalization
    if log:
        sc.pp.log1p(adata)
    if scale:
        sc.pp.scale(adata)
    adata.var_names_make_unique()

    if sp.isspmatrix_csr(adata.X):
        features = pd.DataFrame.sparse.from_spmatrix(adata.X.T,index=adata.var.index,columns=adata.obs.index)
    else:
        features = pd.DataFrame(adata.X.T,index=adata.var.index,columns=adata.obs.index)

    return features
