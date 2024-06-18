import siVAE
import logging
import os
import tensorflow as tf
import scanpy as sc
import pandas as pd
import anndata as ad
import numpy as np

## System
os.environ['TF_CPP_MIN_LOG_LEVEL']  = '3'  # no debugging from TF
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'

logging.getLogger('tensorflow').disabled = True
logging.getLogger().setLevel(logging.INFO)

## Tensorflow
tf.get_logger().setLevel('INFO')
tf.logging.set_verbosity(tf.logging.ERROR)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

#data load
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

adata1=ad.AnnData(adata.X.T,obs=adata.var,var=adata.obs,dtype='float32')

from siVAE.data.data_handler import data2handler
datah_sample, datah_feature, plot_args = data2handler(adata1)

#### Setup the train/test/validation split
k_split=0.99
datah_sample.create_split_index_list(k_split=k_split,random_seed=0)

#### Training Parameters
iter          = 400
mb_size       = 0.2
l2_scale      = 1e-3
keep_prob     = 1
learning_rate = 1e-4
early_stop    = False
decay_rate    = 0.9

#### Model parameters
# Architecture should be a string with a specific format
# architecture: "Encoder-LE-Decoder-Output (0)-Index of LE"
architecture = '256-64-16-LE-16-64-256-0-3'
decoder_activation = 'NA'
zv_recon_scale = 0.1
LE_dim = 16
datah_sample.create_dataset(kfold_idx=0)

#### Set up tf config
#gpu_device = '0'
#os.environ["CUDA_VISIBLE_DEVICES"]  = gpu_device
config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#config.log_device_placement = True
#config.allow_soft_placement = True
#config.intra_op_parallelism_threads = 5
#config.inter_op_parallelism_threads = 5

#### Set parameters

graph_args = {'LE_dim'       : LE_dim,
              'architecture' : architecture,
              'config'       : config,
              'iter'         : iter,
              'mb_size'      : mb_size,
              'l2_scale'     : l2_scale,
              'tensorboard'  : True,
              'batch_norm'   : False,
              'keep_prob'    : keep_prob,
              'log_frequency': 50,
              'learning_rate': learning_rate,
              "early_stopping"   : early_stop,
              "validation_split" : 0,
              "decay_rate"       : decay_rate,
              "decay_steps"      : 1000,
              'var_dependency'   : True,
              'activation_fun'   : tf.nn.relu,
              'activation_fun_decoder': tf.nn.relu,
              'output_distribution': 'normal',
              'beta'               : 1,
              'l2_scale_final'     : 5e-3,
              'log_variational'    : False,
              'beta_warmup'        : 1000,
              'max_patience_count' : 100}

logdir='out_test'
graph_args['logdir_tf'] = logdir
os.makedirs(logdir,exist_ok=True)

# Running siVAE will output the results as siVAE_output which includes the cell and feature embeddings as well as train information.
from siVAE.run_model import run_VAE
siVAE_output = run_VAE(graph_args_sample=graph_args,
                        LE_method='siVAE',
                        datah_sample=datah_sample,
                        datah_feature=datah_feature)

## Create data frame where each rows are genes and columns are metadata/stat
gene_embeddings = siVAE_output.get_feature_embeddings()
gene_names      = siVAE_output.get_model().get_value('var_names')
np.savetxt('brain_lake_siVAE_geneembedding_epoch400.txt',gene_embeddings, fmt='%.6f', delimiter='\t')
np.savetxt('brain_lake_siVAE_genenames_epoch400.txt',gene_names, fmt='%.6f', delimiter='\t')

# Save the result into pickle.
siVAE_output.save(filename=os.path.join(logdir,'brain_result_epoch50.pickle'))

# Check model trained properly through metrics from tensorboard
from siVAE.model.output.plot import plot_scalars
plot_scalars(siVAE_output, os.path.join(logdir,'plot'))

from IPython.core.display import SVG
SVG(filename=os.path.join(logdir,'plot','scalars','combinedModel_1-total_loss3.svg'))
SVG(filename=os.path.join(logdir,'plot','scalars','VAE_sample-Total_loss_13.svg'))
SVG(filename=os.path.join(logdir,'plot','scalars','VAE_feature-Total_loss_13.svg'))

# Plot cell embeddings for visualization
from siVAE.model.output import plot
kwargs={'s':5,'edgecolor':"none"}

plot.plot_latent_embeddings(siVAE_output,logdir=logdir,
                            filename='CellEmbeddings3.svg',
                            show_legend=True,
                            **kwargs)

SVG(filename=os.path.join(logdir,'GeneEmbeddings3.svg'))
