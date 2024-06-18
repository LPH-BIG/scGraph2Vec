import scvi
import networkx as nx
import numpy as np
import pickle as pkl
import scipy.sparse as sp
import sys
import pandas as pd
import torch
import anndata as ad
import scanpy as sc

feature="Adult-Brain_Lake_dge.txt"
raw_fea = pd.read_csv(feature,sep=",",header=0,index_col=0)
df_t = raw_fea.T
var = pd.DataFrame(index=df_t.columns)
obs = pd.DataFrame(index=df_t.index)
adata=ad.AnnData(df_t.values,obs=obs,var=var,dtype='float32')

sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.normalize_total(adata, exclude_highly_expressed=True, target_sum=1e6)
sc.pp.log1p(adata)

g = nx.read_edgelist('adjacency_edgelist.txt',delimiter='\t')
nodeset = sorted(set(g.nodes()))
adj = nx.adjacency_matrix(g,nodelist=nodeset)
raw_fea = pd.DataFrame(adata.X.T,index=adata.var.index,columns=adata.obs.index)
raw_fea = raw_fea.drop(set(raw_fea.index)-set(g.nodes()))
nfea = len(set(g.nodes())-set(raw_fea.index))
addfeature = pd.DataFrame(np.zeros((nfea,raw_fea.shape[1])),index=list(g.nodes-set(raw_fea.index)),columns=raw_fea.columns)
raw_fea = raw_fea._append(addfeature)
raw_fea = raw_fea.sort_index(axis=0)
features = raw_fea.values

scvi.settings.seed = 0
adata1=ad.AnnData(features,obs=pd.DataFrame(index=raw_fea.index),var=pd.DataFrame(index=raw_fea.columns),dtype='float32')
scvi.model.SCVI.setup_anndata(adata1)
vae = scvi.model.SCVI(adata1, n_hidden=256, n_layers=3,n_latent=16,latent_distribution="normal")
vae.train(max_epochs=600)
adata1.obsm["X_scVI"] = vae.get_latent_representation()

np.savetxt('brain_lake_scVI_epoch600_embedding.txt',vae.get_latent_representation(), fmt='%.6f', delimiter='\t')
adata1.write('brain_lake_scVI_epoch600_embedding.h5ad')

# Extract training and test losses
train_losses = vae.history["elbo_train_set"]
test_losses = vae.history["elbo_test_set"]
np.savetxt('brain_lake_scVI_epoch600_trainloss.txt',train_losses, fmt='%.6f', delimiter='\t')
np.savetxt('brain_lake_scVI_epoch600_testloss.txt',test_losses, fmt='%.6f', delimiter='\t')
