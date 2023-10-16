# scGraph2Vec

# Requirements
python=3.7,
tensorflow=1.X,
networkx, numpy, python-louvain, scikit-learn, scipy.

# Run Experiments
A first quick test (10 training iterations only):

python3.7 train3d.py --dataset=brain_lake --features=True --output=./test --task=task_2 --model=gcn_vae3d --iterations=10 --learning_rate=0.0001 --hidden=256 --hidden=64 --dimension=16 --beta=10 --lamb=0.1 --gamma=0.1 --s_reg=10 --nb_run=2 --simple=False --validation=True
