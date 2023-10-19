# scGraph2Vec

# Requirements
python == 3.7  
tensorflow == 1.15.0

# Installation
```
git clone https://github.com/LPH-BIG/scGraph2Vec
```

# Run Experiments
A first quick test (10 training iterations only):
```
python3.7 train3d.py --dataset=brain_lake --features=True --output=./test --task=task_2 --model=gcn_vae3d --iterations=10 --learning_rate=0.0001 --hidden2=256 --hidden=64 --dimension=16 --beta=10 --lamb=0.1 --gamma=0.1 --s_reg=10 --nb_run=2 --simple=False --validation=True
python3.7 train3d.py --dataset=brain_lake --features=True --output=./test --task=task_2 --model=gcn_vae --iterations=10 --learning_rate=0.0001 --hidden=64 --dimension=16 --beta=10 --lamb=0.1 --gamma=0.1 --s_reg=10 --nb_run=10 --simple=False --validation=True
```
