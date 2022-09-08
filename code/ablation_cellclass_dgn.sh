NEPOCHS=2001

python3 gnn.py DGN.GCN --runs 5 --epochs $NEPOCHS --num_layers 1
python3 gnn.py GraphSAGE --runs 5 --epochs $NEPOCHS --num_layers 1
python3 gnn.py DGN.GAT --runs 5 --epochs $NEPOCHS --num_layers 1 --gat_heads 8
python3 gnn.py DGN.GAT --runs 5 --epochs $NEPOCHS --num_layers 1 --gat_heads 32

python3 gnn.py DGN.GCN --runs 2 --epochs $NEPOCHS --num_layers 2
python3 gnn.py GraphSAGE --runs 2 --epochs $NEPOCHS --num_layers 2
python3 gnn.py DGN.GAT --runs 2 --epochs $NEPOCHS --num_layers 2 --gat_heads 8
python3 gnn.py DGN.GAT --runs 2 --epochs $NEPOCHS --num_layers 2 --gat_heads 32
