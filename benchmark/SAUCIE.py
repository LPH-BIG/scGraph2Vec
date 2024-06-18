import SAUCIE
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import sklearn.decomposition
import scprep
import sys

data_raw=pd.read_csv('Adult-Brain_Lake_dge.txt',sep=",",header=0,index_col=0)

#pca_op = sklearn.decomposition.PCA(100)
#data = pca_op.fit_transform(data_raw)
data=data_raw

load = SAUCIE.Loader(data, shuffle=False)

tf.reset_default_graph()
# build the SAUCIE model
saucie = SAUCIE.SAUCIE(data.shape[1], lambda_c=.1, lambda_d=.9)

saucie.train(load, steps=600)
embedding = saucie.get_embedding(load)
num_clusters, clusters = saucie.get_clusters(load)

res = pd.DataFrame(embedding,index=data_raw.index)
res['cluster']=clusters

res.to_csv("brain_embedding_lambda_c1_lambda_d9_epoch600.txt")
