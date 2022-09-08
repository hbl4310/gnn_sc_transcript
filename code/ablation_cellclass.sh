NRUNS=5
NEPOCHS=2001
NLAYERS=1

python3 gnn.py GCN --runs $NRUNS --epochs $NEPOCHS --num_layers $NLAYERS
python3 gnn.py GraphSAGE --runs $NRUNS --epochs $NEPOCHS --num_layers $NLAYERS
python3 gnn.py GAT --runs $NRUNS --epochs $NEPOCHS --num_layers $NLAYERS --gat_heads 8
python3 gnn.py GAT --runs $NRUNS --epochs $NEPOCHS --num_layers $NLAYERS --gat_heads 32
# 	python3 gnn.py GAT --runs $NRUNS --epochs $NEPOCHS --num_layers $NLAYERS --gat_heads 8 --gat_concat
# 	python3 gnn.py GAT --runs $NRUNS --epochs $NEPOCHS --num_layers $NLAYERS --gat_heads 32 --gat_concat
python3 gnn.py Cheb --runs $NRUNS --epochs $NEPOCHS --num_layers $NLAYERS --cheb_k 1
python3 gnn.py Cheb --runs $NRUNS --epochs $NEPOCHS --num_layers $NLAYERS --cheb_k 3
python3 gnn.py Cheb --runs $NRUNS --epochs $NEPOCHS --num_layers $NLAYERS --cheb_k 10
python3 gnn.py GIN --runs $NRUNS --epochs $NEPOCHS --num_layers $NLAYERS --gin_eps 0.1
python3 gnn.py DGCNN --runs $NRUNS --epochs $NEPOCHS --num_layers $NLAYERS 
if [[NLAYERS -gt 1]]; then 
    python3 gnn.py DGCNN --runs $NRUNS --epochs $NEPOCHS --num_layers $NLAYERS --dgcnn_dynamic --dgcnn_k 1
    python3 gnn.py DGCNN --runs $NRUNS --epochs $NEPOCHS --num_layers $NLAYERS --dgcnn_dynamic --dgcnn_k 10
fi
