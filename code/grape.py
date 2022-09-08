import torch 
import torch.nn.functional as F

import GRAPE.models.gnn_model as gnn_model
from GRAPE.models.gnn_model import is_data_homoegenous
from GRAPE.models.prediction_model import MLPNet
from GRAPE.utils.utils import build_optimizer, get_known_mask, mask_edge
from gnn import calc_accuracy
import os 
from data import preprocess_leiden, connectivities2edge_index

DEBUG = False

def process_data(data, device=torch.device('cpu')):
    if is_data_homoegenous(data): 
        x = data.x.clone().detach().to(device)
        edge_index = data.edge_index.clone().detach().to(device)
    else: 
        x = {k: v.clone().detach().to(device) for k,v in data.collect('x').items()}
        edge_index = torch.cat([data.get_edge_store(*t)['edge_index'] for t in data.edge_types], axis=1).clone().detach().to(device)
    y = data.y.clone().detach().to(device)
    train_edge_index = data.train_edge_index.clone().detach().to(device)
    train_edge_attr = data.train_edge_attr.clone().detach().to(device)
    train_labels = data.train_labels.clone().detach().to(device)  # needed if reconstruction loss used
    all_train_y_mask = data.train_y_mask.clone().detach().to(device)
    test_y_mask = data.test_y_mask.clone().detach().to(device)

    train_y_mask = all_train_y_mask.clone().detach()
    print("all y num is {}, train num is {}, test num is {}"\
            .format(
            all_train_y_mask.shape[0],torch.sum(train_y_mask),
            torch.sum(test_y_mask)))

    n_row, n_col = data.df_X.shape
    return x, edge_index, y, train_labels, train_edge_index, train_edge_attr, train_y_mask, test_y_mask, n_row, n_col

def impute_features(model, impute_model, x, 
        train_edge_attr, train_edge_index, edge_index, n_samples, n_features
    ): 
    x_embd = model(x, train_edge_attr, train_edge_index)
    X = impute_model([x_embd[edge_index[0, :int(n_samples * n_features)]], x_embd[edge_index[1, :int(n_samples * n_features)]]])
    X = torch.reshape(X, [n_samples, n_features])
    return x_embd, X


DEFAULT_PREDICT_LOSS = F.mse_loss  # F.cross_entropy
def train_y(args, model, impute_model, predict_model, 
        x, edge_index, y, 
        train_labels, train_edge_index, train_edge_attr, train_y_mask, test_y_mask,
        recon_loss_func = None, 
        predict_loss_func = DEFAULT_PREDICT_LOSS, edge_index_builder=None, 
        record_interval=10, log_interval=100, device=torch.device('cpu'), log_path=''
    ):
    trainable_parameters = list(model.parameters()) \
                            + list(impute_model.parameters()) \
                            + list(predict_model.parameters())
    print("total trainable_parameters: ",len(trainable_parameters), 
        'total model parameters:', sum(p.numel() for p in model.parameters()) + \
                                    sum(p.numel() for p in impute_model.parameters()) + \
                                    sum(p.numel() for p in predict_model.parameters())
                                    )
    # build optimizer
    scheduler, opt = build_optimizer(args.optimizer, trainable_parameters)

    if type(x) == dict: 
        n_row, n_col = x['cell'].shape[0], x['gene'].shape[0]
    else: 
        n_row, n_col = y.shape[0], x.shape[0] - y.shape[0]
    
    # train
    epochs = []
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
    Lr = []

    for epoch in range(args.optimizer.epochs):
        model.train()
        impute_model.train()
        predict_model.train()

        known_mask = get_known_mask(args.known, int(train_edge_attr.shape[0] / 2)).to(device)
        double_known_mask = torch.cat((known_mask, known_mask), dim=0)
        # known_edge_index, known_edge_attr = mask_edge(train_edge_index, train_edge_attr, double_known_mask, True)

        opt.zero_grad()
        x_emb, X = impute_features(model, impute_model, x, 
                                    train_edge_attr, train_edge_index, edge_index, n_row, n_col
                                )
        if edge_index_builder: 
            pred = predict_model(X, edge_index_builder(X).to(device)).squeeze()
        else: 
            pred = predict_model(X).squeeze()
        pred_train = pred[train_y_mask]
        label_train = y[train_y_mask]

        loss = predict_loss_func(pred_train, label_train)
        # TODO augment loss with imputation loss from scGNN??
        if recon_loss_func is not None: 
            reconX_train = X.flatten()[:int(train_edge_attr.shape[0] / 2)]
            recon_loss = recon_loss_func(reconX_train[known_mask], train_labels[known_mask])
            if DEBUG: 
                print(recon_loss)
            loss += recon_loss
        # TODO add other downstream loss? like clustering losses

        loss.backward()
        opt.step()
        if scheduler is not None:
            scheduler.step(epoch)
        for param_group in opt.param_groups:
            Lr.append(param_group['lr'])

        if epoch % record_interval == 0: 
            pred_test, label_test, test_loss = eval_y(model, impute_model, predict_model, 
                                                x, y, train_edge_attr, train_edge_index, edge_index, 
                                                n_row, n_col, test_y_mask, 
                                                recon_loss_func=recon_loss_func, train_labels=train_labels, known_mask=known_mask,
                                                edge_index_builder=edge_index_builder
                                                )
            epochs.append(epoch)
            train_losses.append(loss.item())
            test_losses.append(test_loss.item())
            if len(pred_test.shape) > 1: 
                train_accuracies.append(calc_accuracy(pred_train, label_train))
                test_accuracies.append(calc_accuracy(pred_test, label_test))

        if epoch % log_interval == 0: 
            str_output = f'[Epoch {epoch}] train (loss={loss.item()}) '
            str_output += f'test (loss={test_loss})'
            print(str_output)

    pred_test, label_test, _ = eval_y(model, impute_model, predict_model, 
                                    x, y, train_edge_attr, train_edge_index, edge_index, 
                                    n_row, n_col, test_y_mask, 
                                    recon_loss_func=recon_loss_func, train_labels=train_labels, known_mask=known_mask,
                                    edge_index_builder=edge_index_builder
                                    )
    pred_train = pred_train.detach().cpu().numpy()
    label_train = label_train.detach().cpu().numpy()
    pred_test = pred_test.detach().cpu().numpy()
    label_test = label_test.detach().cpu().numpy()

    obj = dict()
    obj['args'] = args
    obj['curves'] = dict()
    obj['curves']['epoch'] = epochs
    obj['curves']['train_loss'] = train_losses
    obj['curves']['test_loss'] = test_losses
    if train_accuracies: 
        obj['curves']['train_acc'] = train_accuracies
        obj['curves']['test_acc'] = test_accuracies
    obj['lr'] = Lr
    obj['outputs'] = dict()
    obj['outputs']['pred_train'] = pred_train.tolist()
    obj['outputs']['label_train'] = label_train.tolist()
    obj['outputs']['pred_test'] = pred_test.tolist()
    obj['outputs']['label_test'] = label_test.tolist()

    if log_path: 
        print('saving models')
        torch.save(model, os.path.join(log_path, 'model.pt'))
        torch.save(impute_model, os.path.join(log_path, 'impute_model.pt'))
        torch.save(predict_model, os.path.join(log_path, 'predict_model.pt'))

    return obj 

@torch.no_grad()
def eval_y(model, impute_model, predict_model, 
        x, y, train_edge_attr, train_edge_index, edge_index, 
        n_row, n_col, test_y_mask, 
        recon_loss_func = None, train_labels = None, known_mask = None, 
        predict_loss_func = DEFAULT_PREDICT_LOSS, edge_index_builder = None
    ):
    model.eval()
    impute_model.eval()
    predict_model.eval()

    x_emb, X = impute_features(model, impute_model, x, 
                            train_edge_attr, train_edge_index, edge_index, n_row, n_col
                        )
    if edge_index_builder: 
        pred = predict_model(X, edge_index_builder(X).to(y.device)).squeeze()
    else:
        pred = predict_model(X).squeeze()
    pred_test = pred[test_y_mask]
    label_test = y[test_y_mask]

    loss = predict_loss_func(pred_test, label_test)
    if recon_loss_func is not None: 
        reconX_train = X.flatten()[:int(train_edge_attr.shape[0] / 2)]
        recon_loss = recon_loss_func(reconX_train[known_mask], train_labels[known_mask])
        loss += recon_loss

    return pred_test, label_test, loss

from dataclasses import dataclass
@dataclass 
class ArgsGnnModel: 
    aggr: str = 'mean'
    concat_states: bool = False
    dropout: float = 0. 
    edge_dim: int = 64 
    edge_mode: int = 1  # 0: use it as weight; 1: as input to mlp
    gnn_activation: str = 'relu'
    model_types: str = 'EGSAGE_EGSAGE_EGSAGE'
    node_dim: int = 64 
    norm_embs: str = None  # default to be all true
    post_hiddens: str = None  # default to be 1 hidden of node_dim
    embedding_input_dim: int = None  # for gene node embedding input; num of genes used 

@dataclass
class ArgsOptimizer: 
    epochs: int = 1000
    lr: float = 0.001
    opt: str = 'adam'
    opt_decay_rate: float = 0.9
    opt_decay_step: int = 1000
    opt_scheduler: str = None
    weight_decay: float = 0.
@dataclass 
class Args: 
    model: ArgsGnnModel = ArgsGnnModel()
    optimizer: ArgsOptimizer = ArgsOptimizer() 
    auto_known: bool = False
    ce_loss: bool = False 
    concat_states: bool = False
    dropout: float = 0.
    impute_activation: str = 'relu'
    impute_hiddens: str = '64'
    known: float = 0.7
    loss_mode: int = 0 
    mode: str = 'train'  # debug
    # norm_label
    save_model: bool = False
    save_prediction: bool = False
    split_sample: float = 0.
    split_test: bool = False 
    split_train: bool = False 
    transfer_dir: str = None
    transfer_extra: str = ''
    valid: float = 0.
    log_path: str = ''

    predict_hiddens: str = ''
    output_activation: str = None 


from torch_cluster import knn
import scanpy as sc 
def pcaknn_edge_index(X, k, npcs=40): 
    X = torch.tensor(adata.X, requires_grad=True)
    U, S, V = torch.pca_lowrank(X, q=npcs)
    edge_index = knn(U, U, k, None, None)
    edge_index = edge_index[:, ~edge_index[0].eq(edge_index[1])]
    return torch.cat([edge_index, edge_index.flip(0)], axis=1)


if __name__ == '__main__': 
    import argparse, os
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='./output_grape')
    # data
    parser.add_argument('--n_genes', type=int, default=100)
    parser.add_argument('--embed_dim', type=int, default=None)
    parser.add_argument('--edge_index_k', type=int, default=30)
    # optimiser
    parser.add_argument('--epochs', type=int, default=1)
    # model
    parser.add_argument('--model_types', type=str, default='EGSAGE_EGSAGE_EGSAGE')
    parser.add_argument('--norm_embs', type=str, default=None)
    parser.add_argument('--impute_hiddens', type=str, default='64')
    parser.add_argument('--predict_hiddens', type=str, default='')
    parser.add_argument('--predict_model_type', type=str, default='MLP')
    # recon loss
    parser.add_argument('--recon_loss', action='store_true', default=False)
    parser.add_argument('--recon_loss_lambda', type=float, default=0.000001)
    # misc
    parser.add_argument('--dryrun', action='store_true', default=False)

    args = parser.parse_args()
    print(args)
    print(args.predict_model_type)

    import GRAPE.models.gnn_model as gnn_model
    from GRAPE.models.prediction_model import MLPNet
    import GRAPE.uci.uci_data as uci_data

    import data
    data.sc.settings.verbosity = 0
    from data import preprocess_paul15, preprocess_basic, get_paul15, get_pbmc3k, data_heterogeneous
    import gnn 
    import utils


    adata = preprocess_paul15(get_paul15(), n_top_genes=args.n_genes) 
    target_col='paul15_clusters_ind'
    n_genes = args.n_genes if args.n_genes else adata.X.shape[1]

    # get data
    split_sample = 0. 
    node_mode = 2 
    train_edge = 0.7 
    split_by = 'y'
    train_y_frac = 0.7 
    seed = 0
    data = uci_data.load_data(adata, target_col=target_col, node_mode=node_mode, 
            train_edge=train_edge, split_sample=split_sample, split_by=split_by, train_y=train_y_frac, seed=seed,
            one_hot=True)

    # embed data into heterogeneous graph 
    if args.embed_dim: 
        data = data_heterogeneous(data, args.embed_dim)

    # optimise 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    args_opt = ArgsOptimizer(epochs=args.epochs)
    args_model = ArgsGnnModel(
        model_types=args.model_types,
        norm_embs=args.norm_embs,  # normalise output of EGSAGE layer (true if not None)
        embedding_input_dim=n_genes, 
    )

    args2 = Args(model = args_model, optimizer = args_opt, 
        impute_hiddens=args.impute_hiddens, 
        predict_hiddens=args.predict_hiddens,
        output_activation='softmax', 
        log_path=args.output_dir, 
        )

    # GNN model 
    model = gnn_model.get_gnn(data, args2.model).to(device)

    # imputation model 
    # TODO: use AE/VAE for this? or just explore reconstruction loss functions
    impute_hiddens = list(map(int, args.impute_hiddens.split('_')))
    input_dim = args2.model.node_dim * 2

    output_dim = 1  # imputation output is 1
    impute_model = MLPNet(input_dim, output_dim,
                            hidden_layer_sizes=impute_hiddens,
                            hidden_activation=args2.impute_activation,
                            dropout=args2.dropout).to(device)


    # cell classification model 
    output_dim = 1 if len(data.y.shape) == 1 else data.y.shape[1]
    edge_index_builder = None
    if args.predict_model_type == 'MLP':
        predict_model = MLPNet(n_genes, output_dim,
                            output_activation=args2.output_activation, 
                            hidden_layer_sizes=args2.predict_hiddens,
                            dropout=args2.dropout).to(device)
    elif args.predict_model_type == 'GCN': 
        predict_model = gnn.GNN('GCN', n_genes, output_dim, num_layers=2, hidden_dim=64).to(device)
        edge_index_builder = lambda x: pcaknn_edge_index(x, args.edge_index_k)
    elif args.predict_model_type == 'DGNGCN': 
        from gnn import GNN_DGN_wrapper
        hidden_dim = 64
        k = 10
        gnn = gnn.GNN('GCN', n_genes, hidden_dim, num_layers=1, hidden_dim=hidden_dim)
        predict_model = GNN_DGN_wrapper(gnn, output_dim, k=k, hidden_dim=hidden_dim).to(device)
        edge_index_builder = lambda x: pcaknn_edge_index(x, args.edge_index_k)

    # reconstruction loss
    recon_loss_func = None
    if args.recon_loss: 
        from scgnn import loss_function_graph_celltype
        recon_loss_func = lambda a,b: args.recon_loss_lambda * loss_function_graph_celltype(a, b, '', '')

    # training 
    utils.seed(0)
    x, edge_index, y, train_labels, train_edge_index, train_edge_attr, train_y_mask, test_y_mask, n_row, n_col = process_data(data, device=device)

    fn = f'grape_{n_genes}genes_{args.embed_dim}emb_model{args.model_types}_predict{args.predict_model_type}'
    if args.recon_loss:
        fn += '_recon'
    output_dir = os.path.join(args.output_dir, fn)
    os.makedirs(output_dir, exist_ok=True)

    obj = train_y(args2, model, impute_model, predict_model, 
        x, edge_index, y, 
        train_labels, train_edge_index, train_edge_attr, train_y_mask, test_y_mask,
        recon_loss_func=recon_loss_func, 
        edge_index_builder=edge_index_builder, 
        record_interval=10, log_interval=100, device=device, log_path=output_dir
    )
    print('final test acc:', obj['curves']['test_acc'][-1])

    if not args.dryrun: 
        import os, json, dataclasses
        print('saving:', obj.keys())
        obj['args'] = dataclasses.asdict(obj['args'])
        with open(os.path.join(output_dir, 'obj.json'), 'w') as fp:
            json.dump(obj, fp)
