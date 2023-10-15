from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score,calinski_harabasz_score,davies_bouldin_score
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, average_precision_score, roc_auc_score
import numpy as np
import community as cm
import networkx as nx
#import tensorflow as tf
import tensorflow.compat.v1 as tf
import os
import psutil


flags = tf.app.flags
FLAGS = flags.FLAGS



def sigmoid(x):

    """
    Sigmoid activation function
    :param x: scalar value
    :return: sigmoid activation
    """

    return 1 / (1 + np.exp(-x))


def link_prediction(edges_pos, edges_neg, emb):

    """
    Link prediction: computes AUC ROC and AP scores from embeddings vectors,
    and from ground-truth lists of positive and negative node pairs
    :param edges_pos: list of positive node pairs
    :param edges_neg: list of negative node pairs
    :param emb: n*d matrix of embedding vectors for all nodes
    :return: Area Under ROC Curve (AUC ROC) and Average Precision (AP) scores
    """

    preds = []
    preds_neg = []
    for e in edges_pos:
        # Link prediction on positive pairs
        preds.append(sigmoid(emb[e[0], :].dot(emb[e[1], :].T)))
    for e in edges_neg:
        # Link prediction on negative pairs
        preds_neg.append(sigmoid(emb[e[0], :].dot(emb[e[1], :].T)))

    # Stack all predictions and labels
    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])

    # Computes metrics
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score


def community_detection(emb, label, nb_clusters = None):

    """
    Community detection: computes AMI and ARI scores
    from a K-Means clustering of nodes in an embedding space
    :param emb: n*d matrix of embedding vectors for all nodes
    :param label: ground-truth node labels
    :param nb_clusters: int number of ground-truth communities in the graph
    :return: Adjusted Mutual Information (AMI) and Adjusted Rand Index (ARI)
    """

    if nb_clusters is None:
        nb_clusters = len(np.unique(label))

    # K-Means clustering
    #clustering_pred = KMeans(n_clusters = nb_clusters, init = 'k-means++',n_init = 200, max_iter = 500).fit(emb).labels_
    clustering_pred = KMeans(init = 'k-means++',n_init = 200, max_iter = 500).fit(emb).labels_

    # Compute metrics
    ami = adjusted_mutual_info_score(label, clustering_pred,average_method = 'arithmetic')
    #ami = adjusted_mutual_info_score(label, clustering_pred)
    ari = adjusted_rand_score(label, clustering_pred)

    #return clustering_pred,ami, ari
    silhouette=silhouette_score(emb,clustering_pred)
    ch_score = calinski_harabasz_score(emb,clustering_pred)
    dbi = davies_bouldin_score(emb,clustering_pred)

    #return clustering_pred,ami, ari
    return clustering_pred,ami,ari,silhouette,ch_score,dbi

def modularity(emb):
    adjacency=sigmoid(emb.dot(emb.T))
    partition = cm.best_partition(nx.from_scipy_sparse_matrix(adjacency)) #使用louvain寻找聚类的基因簇
    clusters = list(partition.values())

    degrees = adjacency.sum(axis=0).A1
    m = degrees.sum()
    result = 0
    for cluster_id in np.unique(clusters):
        cluster_indices = np.where(clusters == cluster_id)[0]
        adj_submatrix = adjacency[cluster_indices, :][:, cluster_indices]
        degrees_submatrix = degrees[cluster_indices]
        result += np.sum(adj_submatrix) - (np.sum(degrees_submatrix)**2) / m
    return result / m

def show_info(self):
    pid = os.getpid()
    p = psutil.Process(pid)
    info = p.memory_full_info()
    memory = info.uss / 1024 / 1024
    return memory

