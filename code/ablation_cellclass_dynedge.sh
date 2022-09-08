NEPOCHS=2001

python3 gnn.py DynEdge.GCN --runs 5 --epochs $NEPOCHS --num_layers 1
python3 gnn.py DynEdge.GCN --runs 5 --epochs $NEPOCHS --num_layers 2
