# scGraph2Vec: a deep generative model for gene embedding augmented by Graph Neural Network and single-cell omics data

# Introduction
Based on the variational graph autoencoder (VGAE) framework, scGraph2Vec extends the framework's ability to perform link prediction and module detection simultaneously, and enhances the high-dimensional information of gene embedding. scGraph2Vec can help effectively identify gene modules in specific tissue contexts, providing new ideas for studying gene regulatory functions in tissues. In addition, scGraph2Vec can also help elucidate the influence of regulatory genes on biological processes and infer more disease-related genes to explain disease risk.

# Requirements
python == 3.7  
tensorflow == 1.15.0

# Installation
Requirements: networkx, numpy, python-louvain, scikit-learn, scipy, tensorflow (1.15).
```
conda create -n scgraph2vec=3.7
source activate scgraph2vec
git clone https://github.com/LPH-BIG/scGraph2Vec
```

# Run Experiments
In model training, we provided a three-layer neural network (default: 256-64-16) encoder and a two-layer neural network (default: 64-16) encoder. The Adam optimizer strategy and learning rate = 1×10^(-4) were adopted. After a hyperparameter sweep, we recommended using β=10,λ=1,γ=0.1,s=10 as the hyperparameter combination.
A first quick test (10 training iterations only):
```
python3.7 train3d.py --dataset=brain_lake --features=True --output=./test --task=task_2 --model=gcn_vae3d --iterations=10 --learning_rate=0.0001 --hidden2=256 --hidden=64 --dimension=16 --beta=10 --lamb=0.1 --gamma=0.1 --s_reg=10 --nb_run=2 --simple=False --validation=True
python3.7 train3d.py --dataset=brain_lake --features=True --output=./test --task=task_2 --model=gcn_vae --iterations=10 --learning_rate=0.0001 --hidden=64 --dimension=16 --beta=10 --lamb=0.1 --gamma=0.1 --s_reg=10 --nb_run=10 --simple=False --validation=True
```
# Parameter Explanation
The core function of the scGraph2Vec method provided in train3d.py, and the detailed structure of the model has been given in the model.py.
## Parameters related to the input data
  - features: whether to include node features
  - output: output directory
  - model: model to train, among: gcn_ae, gcn_vae, gcn_vae3d
## General parameters associated with GAE/VGAE
  - dropout: 0., dropout rate
  - iterations: 200, number of iterations in training
  - learning_rate: 0.01, initial learning rate (Adam)
  - hidden2: 256, dimension of the first GCN hidden layer
  - hidden: 64, dimension of the second GCN hidden layer
  - dimension: 16, dimension of the output layer, i.e., dimension of the embedding space
## Additional parameters, specific to modularity calculation
  - beta: 0.0, beta hyperparameter
  - lamb: 0.0, Lambda hyperparameter
  - gamma: 1.0, Gamma hyperparameter
  - s_reg: 2, s hyperparameter
## Parameters related to the experimental setup
  - task: task_2', 'task_1: pure community detection; task_2: joint link prediction and community detection
  - nb_run: 1, number of model run + test
  - prop_val: 5., proportion of edges in validation set for the link prediction task
  - prop_test: 5., proportion of edges in test set for the link prediction task
  - validation: False, whether to compute validation results at each iteration, for the link prediction task
  - verbose: True, whether to print all comments details

The default output is 16-dimensional gene embedding, which can be simplified into a two-dimensional vector for visualization by the t-SNE algorithm.
