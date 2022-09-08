import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim

import torch_geometric as pyg
import torch_geometric.nn as pyg_nn

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(MLP, self).__init__()
        self.layer_1 = nn.Linear(input_dim, hidden_dim)
        self.layer_2 = nn.Linear(hidden_dim, output_dim)
        self.seq = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), 
            nn.ReLU(), 
            nn.Linear(hidden_dim, output_dim), 
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, x):
        return self.seq(x)


# output_builder = lambda input_dim, output_dim: MLP(input_dim, output_dim, input_dim)
output_builder = lambda input_dim, output_dim: nn.Sequential(
            nn.Linear(input_dim, output_dim), 
            nn.Softmax(dim=-1)
        )

class GNN(torch.nn.Module):
    conv_layer_map = {
        'GCN': pyg_nn.GCNConv, 
        'GraphSAGE': pyg_nn.SAGEConv, 
        'GAT': pyg_nn.GATConv, 
        'Cheb': pyg_nn.ChebConv, 
    }
    def __init__(self, conv_layer, input_dim, output_dim, *layer_args, num_layers = 1, hidden_dim = 64, output_builder=output_builder, **layer_kwargs):
        super(GNN, self).__init__()
        self.convs = nn.ModuleList() 

        if num_layers == 1: 
            hidden_dim = output_dim
        
        conv_layer_builder = GNN.conv_layer_map[conv_layer]
        self.convs.append(conv_layer_builder(input_dim, hidden_dim, *layer_args, **layer_kwargs))
        for _ in range(num_layers - 1): 
            self.convs.append(conv_layer_builder(hidden_dim, hidden_dim, *layer_args, **layer_kwargs))

        self.output = output_builder(hidden_dim, output_dim)

    def forward(self, x, edge_index, **kwargs): 
        for conv in self.convs:  
            x = conv(x, edge_index, **kwargs) 
        return self.output(x)

class GNN_GIN(torch.nn.Module):
    def __init__(self, input_dim, output_dim, num_layers = 1, hidden_dim = 64, eps = 0.0, output_builder=output_builder):
        super(GNN_GIN, self).__init__()
        self.convs = nn.ModuleList() 

        if num_layers == 1: 
            hidden_dim = output_dim
        
        self.convs.append(pyg_nn.GINConv(MLP(input_dim, hidden_dim, hidden_dim), eps))
        for _ in range(num_layers - 1): 
            self.convs.append(pyg_nn.GINConv(MLP(hidden_dim, hidden_dim, hidden_dim), eps))

        self.output = output_builder(hidden_dim, output_dim)

    def forward(self, x, edge_index, **kwargs): 
        for conv in self.convs:  
            x = conv(x, edge_index, **kwargs) 
        return self.output(x)

class GNN_DGCNN(torch.nn.Module):
    def __init__(self, input_dim, output_dim, *layer_args, num_layers = 1, hidden_dim = 64, dynamic=True, output_builder=output_builder, **layer_kwargs):
        super(GNN_DGCNN, self).__init__()
        self.convs = nn.ModuleList() 

        if num_layers == 1: 
            hidden_dim = output_dim
        
        self.dynamic = dynamic
        if dynamic:            
            conv_layer_builder = pyg_nn.DynamicEdgeConv
        else: 
            conv_layer_builder = pyg_nn.EdgeConv

        self.convs.append(conv_layer_builder(MLP(2*input_dim, hidden_dim, hidden_dim), *layer_args, **layer_kwargs))
        for _ in range(num_layers - 1): 
            self.convs.append(conv_layer_builder(MLP(2*hidden_dim, hidden_dim, hidden_dim), *layer_args, **layer_kwargs))

        self.output = output_builder(hidden_dim, output_dim)

    def forward(self, x, edge_index, **kwargs): 
        for conv in self.convs:  
            if self.dynamic: 
                x = conv(x, **kwargs) 
            else: 
                x = conv(x, edge_index, **kwargs) 
        return self.output(x)

class GNN_Combo(torch.nn.Module):
    def __init__(self, GNN, input_dim, output_dim, k=10, hidden_dim = 64, batch_norm = False, pooling = False):
        super(GNN_Combo, self).__init__()
        self.convs = nn.ModuleList() 

        self.dgcnn = EDynamicEdgeConv(MLP(2*input_dim, hidden_dim, hidden_dim), k)
        self.conv = GNN
        
        # https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.norm.DiffGroupNorm
        self.batch_norm = batch_norm
        if batch_norm: 
            self.dgn = pyg_nn.DiffGroupNorm(hidden_dim, k, lamda=0.01, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        else: 
            self.dgn = None
        
        # TODO  DMoN pooling layer - note this shrinks the output dimension; might be better for some other downstream task 
        # https://github.com/pyg-team/pytorch_geometric/blob/master/examples/proteins_dmon_pool.py
        self.pooling = pooling 
        if pooling: 
            self.dmon = pyg_nn.DMoNPooling([hidden_dim, hidden_dim], k)
        else: 
            self.dmon = None

        self.output = output_builder(hidden_dim, output_dim)

    def forward(self, x, edge_index, **kwargs): 
        x, edge_index_knn = self.dgcnn(x)
        x = self.conv(x, edge_index_knn, **kwargs)
        if self.batch_norm: 
            x = self.dgn(x)
        if self.pooling: 
            if len(x.shape) == 2: 
                x = x.unsqueeze(0)
            _, x, adj, sp2, o2, c2 = self.dmon(x, pyg.utils.to_dense_adj(edge_index))
            # x.shape == (k, hidden_dim)
            x.squeeze()
        return self.output(x)

class GNN_DGN_wrapper(torch.nn.Module): 
    def __init__(self, GNN, output_dim, k, hidden_dim = 64, output_builder=output_builder):
        super(GNN_DGN_wrapper, self).__init__()
        self.GNN = GNN
        # https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.norm.DiffGroupNorm
        self.dgn = pyg_nn.DiffGroupNorm(hidden_dim, k, lamda=0.01, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.output = output_builder(hidden_dim, output_dim)

    def forward(self, x, edge_index, **kwargs): 
        x = self.GNN(x, edge_index, **kwargs)
        x = self.dgn(x)
        return self.output(x)

from data import connectivities2edge_index
import scanpy as sc 
class GNN_DynEdge_wrapper(torch.nn.Module): 
    def __init__(self, GNN, k, npcs=None, cache=True):
        super(GNN_DynEdge_wrapper, self).__init__()
        self.GNN = GNN
        self.k = k 
        self.npcs = npcs 
        self.cache = cache
        self.x = None

    def forward(self, x, _edge_index, **kwargs): 
        if self.cache and self.x is None: 
            self.x = x.clone().detach().cpu().numpy()
        if self.npcs: 
            if self.x is not None: 
                _adata = sc.AnnData(self.x)
            else: 
                _adata = sc.AnnData(x)
            sc.pp.pca(_adata, svd_solver='arpack')
            sc.pp.neighbors(_adata, n_neighbors=self.k, n_pcs=self.npcs)
            edge_index = connectivities2edge_index(_adata).to(x.device)
        else: 
            edge_index = knn(x[0], x[1], self.k, None, None).flip([0])
        return self.GNN(x, edge_index, **kwargs)



"""
https://pytorch-geometric.readthedocs.io/en/latest/notes/cheatsheet.html#heterogeneous-graph-neural-network-operators

DCGNN
https://medium.com/@sanketgujar95/dynamic-graph-cnn-edge-conv-2582c3eb18d8
https://github.com/WangYueFt/dgcnn/blob/e96a7e26555c3212dbc5df8d8875f07228e1ccc2/pytorch/model.py
https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/conv/edge_conv.html
https://github.com/WangYueFt/dgcnn/blob/e96a7e26555c3212dbc5df8d8875f07228e1ccc2/pytorch/model.py
https://github.com/lcosmo/DGM_pytorch/blob/main/DGMlib/model_dDGM.py#L53

DGM 
https://github.com/lcosmo/DGM_pytorch/tree/main/DGMlib

Heterogeneous graphs 
https://pytorch-geometric.readthedocs.io/en/latest/notes/heterogeneous.html#heterogeneous-graph-transformations
https://pytorch-geometric.readthedocs.io/en/latest/notes/cheatsheet.html#heterogeneous-graph-neural-network-operators
"""


# TODO support DataLoader and mini-batches
# implicitly transductive

# TODO add batch_norm https://github.com/Kaixiong-Zhou/DGN/blob/main/models/GCN.py#L35-L42
# https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.norm.DiffGroupNorm

# https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.dense.dense_mincut_pool
# https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.dense.DMoNPooling


from torch.utils.data import DataLoader, TensorDataset

def train(model, x, edge_index, y, n_train,
        lr = 0.001, num_epochs = 1, batch_size=None, record_interval=10, log_interval=100
    ):
    # if batch_size is None: 
    #     batch_size = x.size(0)
    # train_loader = DataLoader(TensorDataset(x[:n_train], y[:n_train]), batch_size=batch_size, shuffle=True)

    optimiser = optim.Adam(model.parameters(), lr=lr)
    epochs = []
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
    for epoch in range(num_epochs):
        model.train()
        optimiser.zero_grad()
        y_hat = model(x, edge_index)[:n_train]
        loss = F.cross_entropy(y_hat, y[:n_train])
        loss.backward()
        optimiser.step()
        # epoch_loss = 0
        # for i, (bx,by) in enumerate(train_loader): 
        #     y_hat = model(bx, edge_index)
        #     loss = F.cross_entropy(y_hat, by)
        #     loss.backward()
        #     optimiser.step()
        #     epoch_loss += loss.item()
        # train_losses.append(epoch_loss/(i+1))
        if epoch % record_interval == 0: 
            epochs.append(epoch)
            train_losses.append(loss.item())
            train_accuracies.append(calc_accuracy(y_hat, y[:n_train]))
            test_loss, test_acc = eval(model, x, edge_index, y, n_train, _print=False)
            test_losses.append(test_loss)
            test_accuracies.append(test_acc)
        if epoch % log_interval == 0: 
            print(f'[EPOCH {epoch}] train loss: {loss.item()}')
    obj = {
        'epochs': epochs, 
        'train_losses': train_losses, 
        'train_accs': train_accuracies, 
        'test_losses': test_losses, 
        'test_accs': test_accuracies, 
    }
    return obj 

def calc_accuracy(y_hat, y):
    num_correct = y_hat.max(axis=1).indices.eq(y.max(axis=1).indices).sum()
    num_total = len(y_hat)
    return 100.0 * (num_correct/num_total).item()

@torch.no_grad()
def eval(model, x, edge_index, y, n_train, _print=True): 
    model.eval()
    y_hat = model(x, edge_index)[n_train:]
    loss = F.cross_entropy(y_hat, y[n_train:])
    accuracy = calc_accuracy(y_hat, y[n_train:])
    if _print: 
        print('accuracy:', accuracy)
    return loss.item(), accuracy



from typing import Callable, Optional, Union

import torch
from torch import Tensor

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptTensor, PairOptTensor, PairTensor

# from ..inits import reset
def reset(value):
    if hasattr(value, 'reset_parameters'):
        value.reset_parameters()
    else:
        for child in value.children() if hasattr(value, 'children') else []:
            reset(child)

try:
    from torch_cluster import knn
except ImportError:
    knn = None
class EDynamicEdgeConv(MessagePassing):
    r"""The dynamic edge convolutional operator from the `"Dynamic Graph CNN
    for Learning on Point Clouds" <https://arxiv.org/abs/1801.07829>`_ paper
    (see :class:`torch_geometric.nn.conv.EdgeConv`), where the graph is
    dynamically constructed using nearest neighbors in the feature space.

    Args:
        nn (torch.nn.Module): A neural network :math:`h_{\mathbf{\Theta}}` that
            maps pair-wise concatenated node features :obj:`x` of shape
            `:obj:`[-1, 2 * in_channels]` to shape :obj:`[-1, out_channels]`,
            *e.g.* defined by :class:`torch.nn.Sequential`.
        k (int): Number of nearest neighbors.
        aggr (string): The aggregation operator to use (:obj:`"add"`,
            :obj:`"mean"`, :obj:`"max"`). (default: :obj:`"max"`)
        num_workers (int): Number of workers to use for k-NN computation.
            Has no effect in case :obj:`batch` is not :obj:`None`, or the input
            lies on the GPU. (default: :obj:`1`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})` or
          :math:`((|\mathcal{V}|, F_{in}), (|\mathcal{V}|, F_{in}))`
          if bipartite,
          batch vector :math:`(|\mathcal{V}|)` or
          :math:`((|\mathcal{V}|), (|\mathcal{V}|))`
          if bipartite *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})` or
          :math:`(|\mathcal{V}_t|, F_{out})` if bipartite
    """
    def __init__(self, nn: Callable, k: int, aggr: str = 'max',
                 num_workers: int = 1, **kwargs):
        super().__init__(aggr=aggr, flow='source_to_target', **kwargs)

        if knn is None:
            raise ImportError('`DynamicEdgeConv` requires `torch-cluster`.')

        self.nn = nn
        self.k = k
        self.num_workers = num_workers
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)

    def forward(
            self, x: Union[Tensor, PairTensor],
            batch: Union[OptTensor, Optional[PairTensor]] = None) -> Tensor:
        # type: (Tensor, OptTensor) -> Tensor  # noqa
        # type: (PairTensor, Optional[PairTensor]) -> Tensor  # noqa
        """"""
        if isinstance(x, Tensor):
            x: PairTensor = (x, x)

        if x[0].dim() != 2:
            raise ValueError("Static graphs not supported in DynamicEdgeConv")

        b: PairOptTensor = (None, None)
        if isinstance(batch, Tensor):
            b = (batch, batch)
        elif isinstance(batch, tuple):
            assert batch is not None
            b = (batch[0], batch[1])

        edge_index = knn(x[0], x[1], self.k, b[0], b[1]).flip([0])

        # propagate_type: (x: PairTensor)
        return self.propagate(edge_index, x=x, size=None), edge_index


    def message(self, x_i: Tensor, x_j: Tensor) -> Tensor:
        return self.nn(torch.cat([x_i, x_j - x_i], dim=-1))

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(nn={self.nn}, k={self.k})'


if __name__=='__main__':
    import argparse, os, sys
    parser = argparse.ArgumentParser()
    parser.add_argument('model_type', type=str)
    parser.add_argument('--output_dir', type=str, default='./output')

    parser.add_argument('--n_genes', type=int, default=None)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--hidden_dim', type=int, default=64)

    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=1)

    parser.add_argument('--gat_heads', type=int, default=1)
    parser.add_argument('--gat_concat', action='store_true', default=False)
    parser.add_argument('--cheb_k', type=int, default=1)
    parser.add_argument('--gin_eps', type=float, default=0.0)
    parser.add_argument('--dgcnn_dynamic', action='store_true', default=False)
    parser.add_argument('--dgcnn_k', type=int, default=1)

    parser.add_argument('--dgn_k', type=int, default=10)
    parser.add_argument('--dynedge_k', type=int, default=10)
    parser.add_argument('--dynedge_npcs', type=int, default=None)

    parser.add_argument('--dryrun', action='store_true', default=False)
    
    args = parser.parse_args()
    print(args)

    # get data 
    import data 
    data.sc.settings.verbosity = 0

    if args.n_genes is None: 
        adata = data.preprocess_paul15(data.get_paul15()) 
    else: 
        adata = data.preprocess_paul15(data.get_paul15(), n_top_genes=args.n_genes) 

    target_col_categories = 'paul15_clusters'
    target_col = 'paul15_clusters_ind'

    n_train_perc = 0.7
    n_train = int(n_train_perc * adata.X.shape[0])

    if args.model_type == 'baseline': 
        # SVM baseline 
        import svm 
        x_train, x_test, y_train, y_test = svm.split_data_cellclass(adata, n_train, target_col=target_col)
        clf = svm.svm_cellclass(x_train, y_train, x_test, y_test, kernel='linear')
        # MLP baseline 
        import mlp 
        NUM_EPOCHS = 1000
        LR = 0.001 
        x = torch.tensor(adata.X)
        y = data.target2onehot(adata, target_col=target_col)
        input_dim = x.shape[1]
        output_dim = data.num_categories(adata, target_col=target_col_categories)
        hidden_dim = 64
        x_train, y_train = x[:n_train], y[:n_train]
        x_test, y_test = x[n_train:], y[n_train:]

        for layers in [
            [(input_dim, output_dim, hidden_dim),], 
            [(input_dim, hidden_dim, hidden_dim),(hidden_dim, output_dim, hidden_dim),], 
            [(input_dim, hidden_dim, hidden_dim),(hidden_dim, hidden_dim, hidden_dim),(hidden_dim, output_dim, hidden_dim),], 
        ]: 
            model = nn.Sequential(*[mlp.SimpleMLP(a,b,c) for a,b,c in layers])
            losses = mlp.train(model, x_train, y_train, lr=LR, num_epochs=NUM_EPOCHS)
            print(f'MLP-{len(layers)}', end=' ')
            test_acc = mlp.eval(model, x_test, y_test)
        sys.exit()

    # move data to gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.tensor(adata.X).to(device)
    y = data.target2onehot(adata, target_col=target_col).to(device)

    # get edge index
    import torch_cluster
    import utils
    utils.seed(0) 
    edge_indices = []
    edge_indices.append(('empty', torch.empty(2, 0, dtype=torch.long)))
    adata = data.preprocess_leiden(adata)
    edge_indices.append(('pca+knn10', data.connectivities2edge_index(adata)))
    edge_indices.append(('knn20', torch_cluster.knn(x, x, 20)))
    edge_indices.append(('euc0.5', data.implied_edge_index(adata, threshold_quantile=0.005)))
    adata = data.preprocess_leiden(adata, n_neighbors=30)
    edge_indices.append(('pca+knn30', data.connectivities2edge_index(adata)))
    edge_indices.append(('knn50', torch_cluster.knn(x, x, 50)))
    edge_indices.append(('euc2', data.implied_edge_index(adata, threshold_quantile=0.02)))
    
    for edge_index_name,edge_index in edge_indices: 
        print(edge_index_name, 'edges:', edge_index.shape[1], 'edge density:', edge_index.shape[1]/adata.X.shape[0]**2)

    import json, time
    utils.seed(0) 

    RUNS = args.runs
    NUM_EPOCHS = args.epochs
    LR = 0.001

    input_dim = x.shape[1]
    output_dim = data.num_categories(adata, target_col=target_col_categories)

    def _get_model(model_type, input_dim, output_dim, args):
        model_flavor = ''
        if model_type == 'GCN':
            model = GNN('GCN', input_dim, output_dim, num_layers=args.num_layers, hidden_dim=args.hidden_dim)
        elif model_type == 'GraphSAGE':
            model = GNN('GraphSAGE', input_dim, output_dim, num_layers=args.num_layers, hidden_dim=args.hidden_dim)
        elif model_type == 'GAT':
            model = GNN('GAT', input_dim, output_dim, num_layers=args.num_layers, hidden_dim=args.hidden_dim, heads=args.gat_heads, concat=args.gat_concat)
            model_flavor += f'{args.gat_heads}.{args.gat_concat}'
        elif model_type == 'Cheb':
            model = GNN('Cheb', input_dim, output_dim, args.cheb_k, num_layers=args.num_layers, hidden_dim=args.hidden_dim)
            model_flavor += f'{args.cheb_k}'
        elif model_type == 'GIN':
            model = GNN_GIN(input_dim, output_dim, num_layers=args.num_layers, hidden_dim=args.hidden_dim, eps=args.gin_eps)
            model_flavor += f'{args.gin_eps}'
        elif model_type == 'DGCNN': 
            if args.dgcnn_dynamic:
                model = GNN_DGCNN(input_dim, output_dim, args.dgcnn_k, num_layers=args.num_layers, hidden_dim=args.hidden_dim, dynamic=args.dgcnn_dynamic)
                model_flavor += f'dyn.{args.dgcnn_k}'
            else: 
                model = GNN_DGCNN(input_dim, output_dim, num_layers=args.num_layers, hidden_dim=args.hidden_dim, dynamic=args.dgcnn_dynamic)
        # elif model_type == 'combo':
        #     model = GNN_Combo(input_dim, output_dim, hidden_dim=args.hidden_dim, batch_norm=False)
        else: 
            raise(NotImplementedError())
        return model, model_flavor

    if args.model_type.startswith('DGN.'): 
        model_type = args.model_type.split('.')[1]
        gnn, model_flavor = _get_model(model_type, input_dim, args.hidden_dim, args)
        model = GNN_DGN_wrapper(gnn, output_dim, k=args.dgn_k, hidden_dim=args.hidden_dim)
        model_flavor = model_flavor[:-1] + f'-{args.dgn_k}'
    elif args.model_type.startswith('DynEdge.'):
        model_type = args.model_type.split('.')[1]
        gnn, model_flavor = _get_model(model_type, args.hidden_dim, args.hidden_dim, args)
        model = GNN_Combo(gnn, input_dim, output_dim, args.dynedge_k, hidden_dim=args.hidden_dim)
        model_flavor = model_flavor[:-1] + f'-{args.dynedge_k}.{args.dynedge_npcs}'
    elif args.model_type.startswith('DynEdgeDGN.'): 
        model_type = args.model_type.split('.')[1]
        gnn, model_flavor = _get_model(model_type, args.hidden_dim, args.hidden_dim, args)
        model = GNN_DGN_wrapper(gnn, args.hidden_dim, k=args.dgn_k, hidden_dim=args.hidden_dim)
        model = GNN_Combo(model, input_dim, output_dim, args.dynedge_k, hidden_dim=args.hidden_dim)
        model_flavor = model_flavor[:-1] + f'-{args.dynedge_k}.{args.dynedge_npcs}-{args.dgn_k}'
    else: 
        model, model_flavor = _get_model(args.model_type, input_dim, output_dim, args)

    if model_flavor: 
        model_flavor = '.' + model_flavor 
    model_name = f'{args.model_type}_{args.num_layers}.{args.hidden_dim}{model_flavor}'
    print('model:', model_name)
    nparams = sum(p.numel() for p in model.parameters())
    print('model params:', nparams)
    
    genestr = ''
    if args.n_genes is not None: 
        genestr += f'_{args.n_genes}genes_'
    output_dir = os.path.abspath(args.output_dir)
    model = model.to(device)
    for run in range(RUNS):
        print('beginning run:', run)
        for edge_index_name,edge_index in edge_indices: 
            print('adjacency mode:', edge_index_name)
            edge_index = edge_index.to(device)
            stime = time.time()
            obj = train(model, x, edge_index, y, n_train, lr=LR, num_epochs=NUM_EPOCHS, log_interval=1000)

            obj['time'] = time.time() - stime
            obj['nparams'] = nparams
            fn = f'{str(run).zfill(3)}{genestr}_{model_name}_{edge_index_name}'
            if not args.dryrun:
                with open(os.path.join(output_dir, fn+'.json'), 'w') as fp:
                    json.dump(obj, fp)

            torch.cuda.empty_cache()
            if args.model_type == 'DGCNN' and args.dgcnn_dynamic: 
                break 

    # fig, ax = plt.subplots(2, figsize=(8, 10))
    # ax[0].plot(obj['train_losses'], label='train')
    # ax[0].plot(obj['test_losses'], label='test')
    # ax[0].set_yscale('log')
    # ax[0].legend()
    # ax[1].plot(obj['train_accs'], label='train')
    # ax[1].plot(obj['test_accs'], label='test')
    # ax[1].legend()
    # fig.show()
