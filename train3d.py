from evaluation import community_detection, link_prediction
from input_data import create_adjacency
from louvain import *
from model import  GCNModelAE, GCNModelVAE,GCNModelVAE3D
from optimizer import OptimizerAE, OptimizerVAE
from preprocessing import *
from sampling import get_distribution, node_sampling
import numpy as np
import os
import math
import scipy.sparse as sp
import tensorflow as tf
import time


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


flags = tf.app.flags
FLAGS = flags.FLAGS


# Parameters related to the input data
flags.DEFINE_string('dataset', 'cora', 'Graph dataset, among: cora, citeseer, \
                                        pubmed, blogs, cora-large, sbm, brain_lake')
flags.DEFINE_boolean('features', False, 'Whether to include node features')
flags.DEFINE_string('output', '/xtdisk/liufan_group/linshq/work/2022project/cellphe/code/modularity_aware_gae-main/final_code/test', 'output directory')

# Parameters related to the Modularity-Aware GAE/VGAE model to train

# 1/3 - General parameters associated with GAE/VGAE
flags.DEFINE_string('model', 'linear_vae', 'Model to train, among: gcn_ae, gcn_vae, \
                                            linear_ae, linear_vae')
flags.DEFINE_float('dropout', 0., 'Dropout rate')
flags.DEFINE_integer('iterations', 200, 'Number of iterations in training')
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate (Adam)')
flags.DEFINE_integer('hidden2', 256, 'Dimension of the GCN hidden layer')
flags.DEFINE_integer('hidden', 64, 'Dimension of the GCN hidden layer')
flags.DEFINE_integer('dimension', 16, 'Dimension of the output layer, i.e., \
                                       dimension of the embedding space')

# 2/3 - Additional parameters, specific to Modularity-Aware models
flags.DEFINE_float('beta', 0.0, 'Beta hyperparameter of Mod.-Aware models')
flags.DEFINE_float('lamb', 0.0, 'Lambda hyperparameter of Mod.-Aware models')
flags.DEFINE_float('gamma', 1.0, 'Gamma hyperparameter of Mod.-Aware models')
flags.DEFINE_integer('s_reg', 2, 's hyperparameter of Mod.-Aware models')

# 3/3 - Additional parameters, aiming to improve scalability
#flags.DEFINE_boolean('fastgae', False, 'Whether to use the FastGAE framework')
flags.DEFINE_integer('nb_node_samples', 1000, 'In FastGAE, number of nodes to \
                                               sample at each iteration, i.e., \
                                               size of decoded subgraph')
flags.DEFINE_string('measure', 'degree', 'In FastGAE, node importance measure used \
                                          for sampling: degree, core, or uniform')
flags.DEFINE_float('alpha', 1.0, 'alpha hyperparameter associated with the FastGAE sampling')
flags.DEFINE_boolean('replace', False, 'Sample nodes with or without replacement')
flags.DEFINE_boolean('simple', True, 'Use simpler (and faster) modularity in optimizers')


# Parameters related to the experimental setup
flags.DEFINE_string('task', 'task_2', 'task_1: pure community detection \
                                       task_2: joint link prediction and \
                                               community detection')
flags.DEFINE_integer('nb_run', 1, 'Number of model run + test')
flags.DEFINE_float('prop_val', 5., 'Proportion of edges in validation set \
                                    for the link prediction task')
flags.DEFINE_float('prop_test', 5., 'Proportion of edges in test set \
                                      for the link prediction task')
flags.DEFINE_boolean('validation', False, 'Whether to compute validation \
                                           results at each iteration, for \
                                           the link prediction task')
flags.DEFINE_boolean('verbose', True, 'Whether to print all comments details')


# Introductory message
if FLAGS.verbose:
    introductory_message()

if not os.path.exists(FLAGS.output):
    os.makedirs(FLAGS.output)

# Initialize lists to collect final results
mean_ami = []
mean_ari = []
clustering_preds = []
cost_modularity = []
modularity_Q = []
recode_loss = []
if FLAGS.task == 'task_2':
    mean_roc = []
    mean_ap = []
    record_auc = []


# Check that the evaluation task in properly defined
if FLAGS.task not in ('task_1', 'task_2'):
    raise ValueError('Error: undefined task!')


# Load data
if FLAGS.verbose:
    print("LOADING DATA\n")
    print("Loading the", FLAGS.dataset, "graph")

if FLAGS.dataset=="brain_lake":
    adj_init, features_init = create_adjacency()
    labels = np.loadtxt("/xtdisk/liufan_group/linshq/work/2022project/cellphe/code/modularity_aware_gae-main/final_code/data/HSIAO_HOUSEKEEPING_GENES_405_new_label.txt", delimiter = ' ', dtype = str)
else:
    adj_init, features_init = create_adjacency(FLAGS.dataset)
    labels = np.loadtxt(FLAGS.dataset)

if FLAGS.verbose:
    print("- Number of nodes:", adj_init.shape[0])
    print("- Number of communities:", len(np.unique(labels)))
    print("- Use of node features:", FLAGS.features)
    print("Done! \n \n \n \n")


# We repeat the entire training+test process FLAGS.nb_run times
for i in range(FLAGS.nb_run):
    # Flag to compute running time for each iteration
    t = time.time()

    if FLAGS.verbose:
        print("EXPERIMENTS ON MODEL", i + 1, "/", FLAGS.nb_run, "\n")
        print("STEP 1/3 - PREPROCESSING STEPS \n")


    # Edge masking for Link Prediction:
    if FLAGS.task == 'task_2':
        # Compute Train/Validation/Test sets
        if FLAGS.verbose:
            print("Masking some edges from the training graph, for link prediction")
            print("(validation set:", FLAGS.prop_val, "% of edges - test set:",
                  FLAGS.prop_test, "% of edges)")
        adj, val_edges, val_edges_false, test_edges, test_edges_false = \
        mask_test_edges(adj_init, FLAGS.prop_test, FLAGS.prop_val)
        if FLAGS.verbose:
            #print("val_edges:"+str(val_edges.shape)+"val_edges_false:"+str(val_edges_false.shape)+"test_edges:"+str(test_edges_false.shape))
            print("Done! \n")
    else:
        adj = adj_init


    # Compute the number of nodes
    num_nodes = adj.shape[0]


    # Preprocessing on node features
    if FLAGS.verbose:
        print("Preprocessing node features")
    if FLAGS.features:
        features = features_init
    else:
        # If features are not used, replace feature matrix by identity matrix
        features = sp.identity(num_nodes)
    features = sparse_to_tuple(features)
    num_features = features[2][1]
    features_nonzero = features[1].shape[0]
    if FLAGS.verbose:
        print("Done! \n")


    # Community detection using Louvain, as a preprocessing step
    if FLAGS.verbose:
        print("Running the Louvain algorithm for community detection")
        print("as a preprocessing step for the encoder")
    # Get binary community matrix (adj_louvain_init[i,j] = 1 if nodes i and j are
    # in the same community) as well as number of communities found by Louvain
    adj_louvain_init, nb_communities_louvain, partition = louvain_adj_clustering(adj, FLAGS.s_reg)
    if FLAGS.verbose:
        print("Done! Louvain has found", nb_communities_louvain, "communities \n")


    sampled_nodes = np.array(range(FLAGS.nb_node_samples))


    # Placeholders
    if FLAGS.verbose:
        print('Setting up the model and the optimizer')
    placeholders = {
        'features': tf.sparse_placeholder(tf.float32),
        'adj': tf.sparse_placeholder(tf.float32),
        'adj_layer2': tf.sparse_placeholder(tf.float32), # Only used for 2-layer GCN encoders
        'degree_matrix': tf.sparse_placeholder(tf.float32),
        'adj_orig': tf.sparse_placeholder(tf.float32),
        'dropout': tf.placeholder_with_default(0., shape = ()),
        'sampled_nodes': tf.placeholder_with_default(sampled_nodes, shape = [FLAGS.nb_node_samples])
    }


    # Create model
    if FLAGS.model == 'gcn_ae':
        # 2-layer GCN Graph Autoencoder
        model = GCNModelAE(placeholders, num_features, features_nonzero)
    elif FLAGS.model == 'gcn_vae':
        # 2-layer GCN Graph Variational Autoencoder
        model = GCNModelVAE(placeholders, num_features, num_nodes, features_nonzero)
    elif FLAGS.model == 'gcn_vae3d':
        # 3-layer GCN Graph Variational Autoencoder
        model = GCNModelVAE3D(placeholders, num_features, num_nodes, features_nonzero)
    else:
        raise ValueError('Undefined model!')


    # Optimizer
    pos_weight = float(num_nodes * num_nodes - adj.sum()) / adj.sum()
    norm = num_nodes * num_nodes / float((num_nodes * num_nodes
                                                    - adj.sum()) * 2)

    if FLAGS.model in ('gcn_ae'):
        opt = OptimizerAE(preds = model.reconstructions,
                          labels = tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                        validate_indices = False), [-1]),
                          degree_matrix = tf.reshape(tf.sparse_tensor_to_dense(placeholders['degree_matrix'],
                                                                        validate_indices = False), [-1]),
                          num_nodes = num_nodes,
                          pos_weight = pos_weight,
                          norm = norm,
                          clusters_distance = model.clusters)
    else:
        opt = OptimizerVAE(preds = model.reconstructions,
                           labels = tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                         validate_indices = False), [-1]),
                           degree_matrix = tf.reshape(tf.sparse_tensor_to_dense(placeholders['degree_matrix'],
                                                                         validate_indices = False), [-1]),
                           model = model,
                           num_nodes = num_nodes,
                           pos_weight = pos_weight,
                           norm = norm,
                           clusters_distance = model.clusters)
    if FLAGS.verbose:
        print("Done! \n")


    # Symmetrically normalized "message passing" matrices
    if FLAGS.verbose:
        print("Preprocessing on message passing matrices")
    adj_norm = preprocess_graph(adj + FLAGS.lamb*adj_louvain_init)
    adj_norm_layer2 = preprocess_graph(adj)
    adj_label = sparse_to_tuple(adj + sp.eye(num_nodes))
    if FLAGS.verbose:
        print("Done! \n")


    # Degree matrices
    deg_matrix, deg_matrix_init = preprocess_degree(adj, FLAGS.simple)


    # Initialize TF session
    if FLAGS.verbose:
        print("Initializing TF session")
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    if FLAGS.verbose:
        print("Done! \n")


    # Model training
    if FLAGS.verbose:
        print("STEP 2/3 - MODEL TRAINING \n")
        print("Starting training")
    
    early_stopping = EarlyStopping(patience=50, verbose=True)  # EarlyStopping 初始化

    for iter in range(FLAGS.iterations):

        # Flag to compute running time for each iteration
        t = time.time()

        # Construct feed dictionary
        feed_dict = construct_feed_dict(adj_norm, adj_norm_layer2, adj_label, \
                                        features, deg_matrix, placeholders)

        # Weights update
        outs = sess.run([opt.opt_op, opt.cost, opt.cost_adj, opt.cost_mod,opt.mod_Q,opt.labels_sub,opt.clusters_distance],feed_dict = feed_dict)

        # Compute average loss
        avg_cost = outs[1]
        recode_loss.append(avg_cost)
        avg_mod = outs[3]
        avg_cost_adj = outs[2]
        mod_q = outs[4]
        cost_modularity.append(avg_mod)
        modularity_Q.append(mod_q)
        if FLAGS.verbose:
            # Display information on the iteration
            print("Iteration:", '%04d' % (iter + 1), "Loss:", "{:.5f}".format(avg_cost),"Loss_linkpred:", "{:.5f}".format(avg_cost_adj),"Loss_Modularity:", "{:.5f}".format(avg_mod),"Modularity_Q:", "{:.5f}".format(mod_q))
            if FLAGS.validation and FLAGS.task == 'task_2':
                feed_dict.update({placeholders['dropout']: 0})
                val_out = sess.run([model.z_mean,opt.cost, opt.cost_adj, opt.cost_mod,opt.mod_Q], feed_dict = feed_dict)
                emb = val_out[0]
                val_loss = val_out[1]
                val_mod = val_out[3]
                val_cost_adj = val_out[2]
                val_mod_q = val_out[4]
                print("Iteration:", '%04d' % (iter + 1), "Validation Loss:", "{:.5f}".format(val_loss),"Loss_linkpred:", "{:.5f}".format(val_cost_adj),"Loss_Modularity:", "{:.5f}".format(val_mod),"Modularity_Q:", "{:.5f}".format(val_mod_q))
                feed_dict.update({placeholders['dropout']: FLAGS.dropout})
                if math.isnan(avg_cost):
                    print("The value is Nan,break...")
                    break
                val_roc, val_ap = link_prediction(val_edges, val_edges_false, emb)
                record_auc.append(val_roc)
                print("Validation AUC:", "{:.5f}".format(val_roc),
                      "Validation AP:", "{:.5f}".format(val_ap))

        early_stopping(avg_cost, sess,FLAGS.output+"/"+str(i))
        if early_stopping.early_stop:
            print("Early stopping")
            break

    if FLAGS.verbose:
        print("Running Time:", "{:.5f}".format(time.time() - t))
        np.savetxt(FLAGS.output+'/run'+str(i)+'_recodeloss.txt',recode_loss, fmt='%.6f', delimiter='\t')
        print("Done! \n")


    # Compute embedding vectors, for evaluation
    if FLAGS.verbose:
        print("STEP 3/3 - MODEL EVALUATION \n")
        print("Computing the final embedding vectors, for evaluation")
    emb = sess.run(model.z_mean, feed_dict = feed_dict)
    np.savetxt(FLAGS.output+'/run'+str(i)+'_embedding.txt',emb, fmt='%.6f', delimiter='\t')
    if FLAGS.verbose:
        print("Done! \n")


    # Test model: link prediction (classification edges/non-edges)
    if FLAGS.task == 'task_2':
        if FLAGS.verbose:
            print("Testing: link prediction")
        # Get ROC and AP scores
        if math.isnan(avg_cost):
            print("The loss is Nan,go to next run...")
            continue
        roc_score, ap_score = link_prediction(test_edges, test_edges_false, emb)
        mean_roc.append(roc_score)
        mean_ap.append(ap_score)
        if FLAGS.verbose:
            print("Done! \n")


    # Test model: community detection
    # K-Means clustering in the embedding space
    if FLAGS.verbose:
        print("Testing: community detection")
    # Get AMI and ARI scores
    clustering_pred,mi_score,ari_score,silhouette,ch_score,dbi = community_detection(emb, labels)
    mean_ami.append(mi_score)
    mean_ari.append(ari_score)
    clustering_preds.append(clustering_pred)
    np.savetxt(FLAGS.output+'/run'+str(i)+'_clustering_preds.txt',clustering_preds, fmt='%.6f', delimiter='\t')
    np.savetxt(FLAGS.output+'/run'+str(i)+'_cost_modularity.txt',cost_modularity, fmt='%.6f', delimiter='\t')
    np.savetxt(FLAGS.output+'/run'+str(i)+'_modularity_Q.txt',modularity_Q, fmt='%.6f', delimiter='\t')
    if FLAGS.verbose:
        print("Done! \n \n \n \n")


# Report final results
print("FINAL RESULTS \n")

if FLAGS.task == 'task_1':
    print('Recall: the selected task was "Task 1", i.e., pure community detection, on', FLAGS.dataset)
else:
    np.savetxt(FLAGS.output+'/record_auc.txt',record_auc, fmt='%.6f', delimiter='\t')
    print('Recall: the selected task was "Task 2", i.e., joint community detection and link prediction, on', FLAGS.dataset)
print("All scores reported below are computed over the", FLAGS.nb_run, "run(s) \n")

print("Community detection:\n")
print("Mean AMI score:", np.mean(mean_ami))
print("Std of AMI scores:", np.std(mean_ami), "\n")

print("Mean ARI score: ", np.mean(mean_ari))
print("Std of ARI scores: ", np.std(mean_ari), "\n")

if FLAGS.task == 'task_2':
    print("Link prediction:\n")
    print("Mean AUC score: ", np.mean(mean_roc))
    print("Std of AUC scores: ", np.std(mean_roc), "\n")

    print("Mean AP score: ", np.mean(mean_ap))
    print("Std of AP scores: ", np.std(mean_ap), "\n \n")
    np.savetxt(FLAGS.output+'/mean_roc.txt',mean_roc, fmt='%.6f', delimiter='\t')
    np.savetxt(FLAGS.output+'/mean_ap.txt',mean_ap, fmt='%.6f', delimiter='\t')
