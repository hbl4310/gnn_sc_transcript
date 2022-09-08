NEPOCHS=2001
NGENES=100
MODEL=DGNGCN

#python3 grape.py --epochs $NEPOCHS --embed_dim 16
#python3 grape.py --epochs $NEPOCHS --embed_dim 32
#python3 grape.py --epochs $NEPOCHS --embed_dim 64
python3 grape.py --n_genes $NGENES --epochs $NEPOCHS --embed_dim 16 --predict_model_type $MODEL
python3 grape.py --n_genes $NGENES --epochs $NEPOCHS --embed_dim 32 --predict_model_type $MODEL
python3 grape.py --n_genes $NGENES --epochs $NEPOCHS --embed_dim 64 --predict_model_type $MODEL
python3 grape.py --n_genes $NGENES --epochs $NEPOCHS --embed_dim 16 --predict_model_type $MODEL --recon_loss
python3 grape.py --n_genes $NGENES --epochs $NEPOCHS --embed_dim 32 --predict_model_type $MODEL --recon_loss
python3 grape.py --n_genes $NGENES --epochs $NEPOCHS --embed_dim 64 --predict_model_type $MODEL --recon_loss
