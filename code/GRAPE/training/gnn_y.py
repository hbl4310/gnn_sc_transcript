import numpy as np
import torch
import torch.nn.functional as F
import pickle

from ..models.gnn_model import get_gnn, is_data_homoegenous
from ..models.prediction_model import MLPNet
from ..utils.plot_utils import plot_curve, plot_sample
from ..utils.utils import build_optimizer, objectview, get_known_mask, mask_edge

def train_gnn_y(data, args, log_path, device=torch.device('cpu')):
    model = get_gnn(data, args.model).to(device)

    if args.impute_hiddens == '':
        impute_hiddens = []
    else:
        impute_hiddens = list(map(int, args.impute_hiddens.split('_')))
    if args.concat_states:
        input_dim = args.model.node_dim * len(model.convs) * 2
    else:
        input_dim = args.model.node_dim * 2
    output_dim = 1  # imputation output is 1
    impute_model = MLPNet(input_dim, output_dim,
                            hidden_layer_sizes=impute_hiddens,
                            hidden_activation=args.impute_activation,
                            dropout=args.dropout).to(device)

    if args.predict_hiddens == '':
        predict_hiddens = []
    else:
        predict_hiddens = list(map(int, args.predict_hiddens.split('_')))
    n_row, n_col = data.df_X.shape
    output_dim = 1 if len(data.y.shape) == 1 else data.y.shape[1]
    predict_model = MLPNet(n_col, output_dim,
                           output_activation=args.output_activation, 
                           hidden_layer_sizes=predict_hiddens,
                           dropout=args.dropout).to(device)

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

    # train
    Train_loss = []
    Test_loss = []
    Test_l1 = []
    Lr = []
    Train_acc, Test_acc = [], []

    if is_data_homoegenous(data): 
        x = data.x.clone().detach().to(device)
        edge_index = data.edge_index.clone().detach().to(device)
    else: 
        x = {k: v.clone().detach().to(device) for k,v in data.collect('x').items()}
        edge_index = torch.cat([data.get_edge_store(*t)['edge_index'] for t in data.edge_types], axis=1).clone().detach().to(device)
    y = data.y.clone().detach().to(device)
    train_edge_index = data.train_edge_index.clone().detach().to(device)
    train_edge_attr = data.train_edge_attr.clone().detach().to(device)
    all_train_y_mask = data.train_y_mask.clone().detach().to(device)
    test_y_mask = data.test_y_mask.clone().detach().to(device)

    if args.valid > 0.:
        valid_mask = get_known_mask(args.valid, all_train_y_mask.shape[0]).to(device)
        valid_mask = valid_mask*all_train_y_mask
        train_y_mask = all_train_y_mask.clone().detach()
        train_y_mask[valid_mask] = False
        valid_y_mask = all_train_y_mask.clone().detach()
        valid_y_mask[~valid_mask] = False
        print("all y num is {}, train num is {}, valid num is {}, test num is {}"\
                .format(
                all_train_y_mask.shape[0],torch.sum(train_y_mask),
                torch.sum(valid_y_mask),torch.sum(test_y_mask)))
        Valid_loss = []
        Valid_l1 = []
        best_valid_loss = np.inf
        best_valid_loss_epoch = 0
        best_valid_l1 = np.inf
        best_valid_l1_epoch = 0
    else:
        train_y_mask = all_train_y_mask.clone().detach()
        print("all y num is {}, train num is {}, test num is {}"\
                .format(
                all_train_y_mask.shape[0],torch.sum(train_y_mask),
                torch.sum(test_y_mask)))

    for epoch in range(args.optimizer.epochs):
        model.train()
        impute_model.train()
        predict_model.train()

        known_mask = get_known_mask(args.known, int(train_edge_attr.shape[0] / 2)).to(device)
        double_known_mask = torch.cat((known_mask, known_mask), dim=0)
        known_edge_index, known_edge_attr = mask_edge(train_edge_index, train_edge_attr, double_known_mask, True)

        opt.zero_grad()
        x_embd = model(x, known_edge_attr, known_edge_index)
        X = impute_model([x_embd[edge_index[0, :int(n_row * n_col)]], x_embd[edge_index[1, :int(n_row * n_col)]]])
        X = torch.reshape(X, [n_row, n_col])
        pred = predict_model(X).squeeze()
        pred_train = pred[train_y_mask]
        label_train = y[train_y_mask]

        loss = F.mse_loss(pred_train, label_train)
        loss.backward()
        opt.step()
        train_loss = loss.item()
        if scheduler is not None:
            scheduler.step(epoch)
        for param_group in opt.param_groups:
            Lr.append(param_group['lr'])
        if output_dim > 1: 
            acc = pred_train.max(axis=1).indices.eq(label_train.max(axis=1).indices).float().mean()
            train_acc = acc.item() 

        model.eval()
        impute_model.eval()
        predict_model.eval()
        with torch.no_grad():
            if args.valid > 0.:
                x_embd = model(x, train_edge_attr, train_edge_index)
                X = impute_model([x_embd[edge_index[0, :int(n_row * n_col)]], x_embd[edge_index[1, :int(n_row * n_col)]]])
                X = torch.reshape(X, [n_row, n_col])
                pred = predict_model(X).squeeze()
                pred_valid = pred[valid_y_mask]
                label_valid = y[valid_y_mask]
                mse = F.mse_loss(pred_valid, label_valid)
                valid_loss = mse.item()
                l1 = F.l1_loss(pred_valid, label_valid)
                valid_l1 = l1.item()
                if valid_l1 < best_valid_l1:
                    best_valid_l1 = valid_l1
                    best_valid_l1_epoch = epoch
                    torch.save(model, log_path + 'model_best_valid_l1.pt')
                    torch.save(impute_model, log_path + 'impute_model_best_valid_l1.pt')
                    torch.save(predict_model, log_path + 'predict_model_best_valid_l1.pt')
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    best_valid_loss_epoch = epoch
                    torch.save(model, log_path + 'model_best_valid_loss.pt')
                    torch.save(impute_model, log_path + 'impute_model_best_valid_loss.pt')
                    torch.save(predict_model, log_path + 'predict_model_best_valid_loss.pt')
                Valid_loss.append(valid_loss)
                Valid_l1.append(valid_l1)

            x_embd = model(x, train_edge_attr, train_edge_index)
            X = impute_model([x_embd[edge_index[0, :int(n_row * n_col)]], x_embd[edge_index[1, :int(n_row * n_col)]]])
            X = torch.reshape(X, [n_row, n_col])
            pred = predict_model(X).squeeze()
            pred_test = pred[test_y_mask]
            label_test = y[test_y_mask]
            mse = F.mse_loss(pred_test, label_test)
            test_loss = mse.item()
            l1 = F.l1_loss(pred_test, label_test)
            test_l1 = l1.item()

            Train_loss.append(train_loss)
            Test_loss.append(test_loss)
            Test_l1.append(test_l1)
            if output_dim > 1: 
                Train_acc.append(train_acc)
                test_acc = pred_test.max(axis=1).indices.eq(label_test.max(axis=1).indices).float().mean().item()
                Test_acc.append(test_acc)

            if epoch % 100 == 0: 
                str_output = f'[Epoch {epoch}] train (loss={train_loss}) '
                if args.valid > 0.:
                    str_output += f'valid (rms={valid_loss}, l1={valid_l1})'
                str_output += f'test (rms={test_loss}, l1={test_l1})'
                print(str_output)

    pred_train = pred_train.detach().cpu().numpy()
    label_train = label_train.detach().cpu().numpy()
    pred_test = pred_test.detach().cpu().numpy()
    label_test = label_test.detach().cpu().numpy()

    obj = dict()
    obj['args'] = args
    obj['curves'] = dict()
    obj['curves']['train_loss'] = Train_loss
    if args.valid > 0.:
        obj['curves']['valid_loss'] = Valid_loss
        obj['curves']['valid_l1'] = Valid_l1
    obj['curves']['test_loss'] = Test_loss
    obj['curves']['test_l1'] = Test_l1
    if output_dim > 1: 
        obj['curves']['train_acc'] = Train_acc
        obj['curves']['test_acc'] = Test_acc
    obj['lr'] = Lr
    obj['outputs'] = dict()
    obj['outputs']['pred_train'] = pred_train
    obj['outputs']['label_train'] = label_train
    obj['outputs']['pred_test'] = pred_test
    obj['outputs']['label_test'] = label_test
    pickle.dump(obj, open(log_path + 'result.pkl', "wb"))

    torch.save(model, log_path + 'model.pt')
    torch.save(impute_model, log_path + 'impute_model.pt')
    torch.save(predict_model, log_path + 'predict_model.pt')

    # obj = objectview(obj)
    # plot_curve(obj['curves'], log_path+'curves.png',keys=None, 
    #             clip=True, label_min=True, label_end=True)
    # plot_curve(obj, log_path+'lr.png',keys=['lr'], 
    #             clip=False, label_min=False, label_end=False)
    # plot_sample(obj['outputs'], log_path+'outputs.png', 
    #             groups=[['pred_train','label_train'],
    #                     ['pred_test','label_test']
    #                     ], 
    #             num_points=20)
    if args.valid > 0.:
        print("best valid loss is {:.3g} at epoch {}".format(best_valid_loss,best_valid_loss_epoch))
        print("best valid l1 is {:.3g} at epoch {}".format(best_valid_l1,best_valid_l1_epoch))
    
    return model, impute_model, predict_model, obj
