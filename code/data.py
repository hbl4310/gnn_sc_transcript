import pandas as pd 
import scanpy as sc
from scanpy import AnnData
import numpy as np  
import torch 
import torch.nn.functional as F
import scipy

from torch_geometric.data import Data, Batch, DataLoader


import os 


# Scanpy config
sc.settings.verbosity = 3             # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.logging.print_header()
sc.settings.set_figure_params(dpi=80, facecolor='white')



data_path = '../scanpy-tutorials/data/'

# TODO put into `def download_pbmc3k``
# !mkdir data
# !wget http://cf.10xgenomics.com/samples/cell-exp/1.1.0/pbmc3k/pbmc3k_filtered_gene_bc_matrices.tar.gz -O data/pbmc3k_filtered_gene_bc_matrices.tar.gz
# !cd data; tar -xzf pbmc3k_filtered_gene_bc_matrices.tar.gz
# !mkdir write


# https://www.10xgenomics.com/resources/datasets/3-k-pbm-cs-from-a-healthy-donor-1-standard-1-1-0
def get_pbmc3k():
    return sc.read_10x_mtx(
        os.path.join(data_path, 'filtered_gene_bc_matrices/hg19/'),  # the directory with the `.mtx` file
        var_names='gene_symbols',                # use gene symbols for the variable names (variables-axis index)
        cache=True                               # write a cache file for faster subsequent reading
    )

def get_paul15():
    return sc.datasets.paul15()

def preprocess_basic(adata:AnnData, 
    highly_variable_min_mean=0.0125, highly_variable_max_mean=3, highly_variable_min_disp=0.5,
    flavor='seurat', n_top_genes=None): 
    sc.pp.calculate_qc_metrics(adata, inplace=True)
    sc.pp.normalize_total(adata, target_sum=1e4)
    if flavor != 'seurat_v3':
        sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, 
        flavor=flavor, 
        min_mean=highly_variable_min_mean,  
        max_mean=highly_variable_max_mean, min_disp=highly_variable_min_disp, 
        n_top_genes=n_top_genes)
    if flavor == 'seurat_v3':
        sc.pp.log1p(adata)
    adata = adata[:, adata.var.highly_variable]
    return adata

def preprocess_leiden(adata:AnnData, 
    n_neighbors=10, n_pcs=40):
    sc.tl.pca(adata, svd_solver='arpack')
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)
    sc.tl.leiden(adata)
    return adata 


def preprocess_pbmc3k(adata: AnnData, 
        min_genes=200, min_cells=3, 
        n_genes_by_counts=2500, pct_counts_mt=5, 
        highly_variable_min_mean=0.0125, highly_variable_max_mean=3, highly_variable_min_disp=0.5,
        n_neighbors=10, n_neighbors_pcs=40
    ): 
    print('Data before preprocessing:')
    print(adata)
    # this is unnecessary if using `var_names='gene_ids'` in `sc.read_10x_mtx`
    adata.var_names_make_unique() 
    # filter out cells with less than min_genes
    sc.pp.filter_cells(adata, min_genes=min_genes) 
    # filter out genes which appear in less than min_cells 
    sc.pp.filter_genes(adata, min_cells=min_cells)
    # probably not necessary
    adata.var['mt'] = adata.var_names.str.startswith('MT-')  # annotate the group of mitochondrial genes as 'mt'
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    # filter out cells with more than n_genes_by_counts unique genes
    adata = adata[adata.obs.n_genes_by_counts < n_genes_by_counts, :]
    # filter out cells with more than pct_counts_mt% mt 
    adata = adata[adata.obs.pct_counts_mt < pct_counts_mt, :] 
    # normalise total count to 10,000 per cell 
    sc.pp.normalize_total(adata, target_sum=1e4)
    # log transform counts: log(X + 1)
    sc.pp.log1p(adata)
    # identify genes whose expression varies a lot between cells
    sc.pp.highly_variable_genes(adata, min_mean=highly_variable_min_mean, max_mean=highly_variable_max_mean, min_disp=highly_variable_min_disp)
    # keep highly variable genes
    adata = adata[:, adata.var.highly_variable]
    # Regress out effects of total counts per cell and the percentage of mitochondrial genes expressed. 
    # https://scanpy.readthedocs.io/en/latest/generated/scanpy.pp.regress_out.html?highlight=regress_out
    sc.pp.regress_out(adata, ['total_counts', 'pct_counts_mt'])
    # Scale the data to unit variance.
    sc.pp.scale(adata, max_value=10)  # TODO get rid of this since mixing test/train
    # compute neighbourhood graph 
    sc.tl.pca(adata, svd_solver='arpack')
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_neighbors_pcs)
    # Leiden clustering 
    sc.tl.leiden(adata)

    # find marker genes
    sc.tl.rank_genes_groups(adata, 'leiden', method='t-test')
    # sc.tl.rank_genes_groups(adata, 'leiden', method='wilcoxon')
    # sc.tl.rank_genes_groups(adata, 'leiden', method='logreg')

    print('Data after preprocessing:')
    print(adata)
    return adata 


def preprocess_paul15(adata: AnnData, 
    **kwargs
    ): 
    # adata.X = adata.X.astype('float64')  # this is not required and results will be comparable without it
    # sc.pp.recipe_zheng17(adata)
    # # run PCA 
    # sc.tl.pca(adata, svd_solver='arpack')
    # # get neighborhood graph
    # sc.pp.neighbors(adata, n_neighbors=4, n_pcs=20)
    # sc.pp.normalize_total(adata, target_sum=1e4)
    # sc.pp.log1p(adata)
    # flavor='seurat_v3', n_top_genes=X
    # sc.pp.highly_variable_genes(adata, min_mean=highly_variable_min_mean, max_mean=highly_variable_max_mean, min_disp=highly_variable_min_disp)
    adata = preprocess_basic(adata, **kwargs)
    cats = adata.obs.paul15_clusters.dtype.categories
    adata.obs['paul15_clusters_ind'] = adata.obs.paul15_clusters.replace(cats.to_list(), range(len(cats)))

    return adata 

def get_pbmc3k_preprocessed():
    return sc.datasets.pbmc3k_processed()
    
def preprocess_pbmc3k_preprocessed(adata: AnnData):
    cats = adata.obs.louvain.dtype.categories
    adata.obs['louvain_ind'] = adata.obs.louvain.replace(cats.to_list(), range(len(cats)))
    return adata

def connectivities2adjacency(adata: sc.AnnData, binary=True): 
    if binary: 
        return torch.tensor((adata.obsp['connectivities'].A > 0.)).float()
    return  torch.tensor(adata.obsp['connectivities'].A).float()    

# convert ann connectivity matrix into sparse adjacency
def connectivities2sparse(adata: sc.AnnData, binary=True):
    coo = scipy.sparse.coo_matrix(adata.obsp['connectivities'])
    if binary:
        values = np.ones_like(coo.data)
    else: 
        values = coo.data 
    indices = np.vstack((coo.row, coo.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape
    if binary:
        return torch.sparse.LongTensor(i, v, torch.Size(shape))
    return torch.sparse.FloatTensor(i, v, torch.Size(shape))

def connectivities2edge_index(adata: sc.AnnData):
    coo = scipy.sparse.coo_matrix(adata.obsp['connectivities'])
    indices = np.vstack((coo.row, coo.col))
    return torch.tensor(indices).type(torch.int64)
    # return A_sparse.coalesce().indices()

def target2onehot(adata: sc.AnnData, target_col='leiden'): 
    return F.one_hot(torch.tensor(adata.obs[target_col].values.astype(int))).float()

def num_categories(adata: sc.AnnData, target_col='leiden'):
    return len(adata.obs[target_col].dtype.categories)

def implied_edge_index(adata, threshold=None, threshold_quantile=0.5, samples_for_statistic=10_000): 
    nrow, _ = adata.X.shape
    norms = []
    if threshold is None:
        ii, jj = np.random.randint(0, nrow, samples_for_statistic), np.random.randint(0, nrow, samples_for_statistic)
        for i, j in zip(ii, jj):
            norms.append(np.linalg.norm(adata.X[i] - adata.X[j]))
        threshold = np.quantile(np.array(norms), threshold_quantile)
        print(f'Distance threshold {threshold_quantile} quantile: {threshold:.2f}')
    edge_index_implied = [[], []]
    for i in range(nrow - 1): 
        for j in range(i, nrow): 
            dist = np.linalg.norm(adata.X[i] - adata.X[j])
            # norms.append(dist)
            if dist < threshold: 
                edge_index_implied[0].append(i)
                edge_index_implied[1].append(j)
    edge_index_implied = torch.tensor(edge_index_implied)
    edge_index_implied = torch.cat([edge_index_implied, edge_index_implied[[1, 0], :]], axis=1)
    print(f'Adjacency sparsity: {edge_index_implied.size(1) / nrow**2:.4f}')
    return edge_index_implied 


def data_heterogeneous(data, emb_dim = 16):
    nrow, ncol = data.df_X.shape 
    data_het = data.to_heterogeneous(
        node_type=torch.tensor([0]*nrow + [1]*ncol), node_type_names=['cell', 'gene'], 
        edge_type=torch.tensor([0]*int(data.num_edges/2) + [1]*int(data.num_edges/2)), 
        edge_type_names=[('cell', 'has-expression', 'gene'), ('gene', 'is-expressed', 'cell')])

    data_het.get_node_store('cell')['x'] = data_het.get_node_store('cell')['x'].repeat(1, emb_dim)
    return data_het
