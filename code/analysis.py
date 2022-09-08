import torch 
import torch.nn.functional as F 

import matplotlib 
import matplotlib.pyplot as plt 
from matplotlib.ticker import MaxNLocator

import pandas as pd 

plt.rcParams.update({'font.size': 22})  # 14 is default
plt.rc('figure', titlesize=14)

# SMALL_SIZE = 8
# MEDIUM_SIZE = 10
# BIGGER_SIZE = 12

# plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
# plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
# plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
# plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
# plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
# plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
# plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

def plot_embeddings(embd, figsize=(8,4)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(range(embd.shape[1]), 
        embd.mean(axis=0), 
        yerr=embd.std(axis=0))
    ax.set_xlabel('embedding')
    ax.set_title('Mean Embeddings and Std')
    return fig, ax 

def similarity_metrics(x, y): 
    cs = torch.nn.CosineSimilarity()(x, y)
    l1 = torch.nn.functional.l1_loss(x, y)
    mse = torch.nn.functional.mse_loss(x, y)
    return {
        'cosine_median': cs.median().item(), 
        'cosine_mean': cs.mean().item(), 
        'l1_median': l1.median().item(), 
        'l1_mean': l1.mean().item(), 
        'mse_median': mse.median().item(), 
        'mse_mean': mse.mean().item(), 
    }

def class_accuracies(labels, predictions, unique_labels): 
    df = pd.DataFrame({'label': labels, 'count': torch.zeros_like(labels)})
    df2 = pd.concat([df[['label']], pd.DataFrame(F.one_hot(predictions, num_classes=len(unique_labels)).numpy())], axis=1)
    df_pred_count = pd.merge(df.groupby('label').count(), df2.groupby('label').sum(), on='label')
    df_pred_acc = df_pred_count.copy()
    for l in unique_labels: 
        df_pred_acc[l] /= df_pred_acc['count']
    df_pred_acc.drop('count', axis=1, inplace=True)

    fig, ax = plt.subplot_mosaic([['counts'], ['accs']], constrained_layout=True, sharex=True, 
        gridspec_kw={'height_ratios':(1,2)}, figsize=(6,9))
    ax['counts'].bar(df_pred_count.index, df_pred_count['count'])
    ax['counts'].set_ylabel(('count'))
    ax['counts'].set_xlabel(('label'))
    ax['accs'].imshow(df_pred_acc.transpose(), cmap='hot')
    # ax['accs'].grid(False)
    ax['accs'].set_ylabel('prediction')
    ax['accs'].set_xlabel('label')
    ax['accs'].sharex(ax['counts'])
    fig.suptitle('Class label counts and prediction percentages')

    return fig, ax

def plot_loss(obj):
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(obj['curves']['train_loss'], label='train')
    ax.plot(obj['curves']['test_loss'], label='test')
    ax.legend()
    ax.set_yscale('log')
    return fig, ax

def plot_acc(obj):
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(obj['curves']['train_acc'], label='train')
    ax.plot(obj['curves']['test_acc'], label='test')
    ax.set_ylabel('acc')
    ax.legend()
    return fig, ax 

def plot_loss_acc(obj):
    fig, ax = plt.subplots(figsize=(8,4))
    epochs = obj['curves']['epoch']
    ln1 = ax.plot(epochs, obj['curves']['train_loss'], '--', label='train loss')
    ln2 = ax.plot(epochs, obj['curves']['test_loss'], '--', label='test loss')
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    ax.set_yscale('log')
    ax2 = ax.twinx()
    ln3 = ax2.plot(epochs, obj['curves']['train_acc'], label='train acc.')
    ln4 = ax2.plot(epochs, obj['curves']['test_acc'], label='test acc.')
    ax2.set_ylabel('accuracy %')
    lns = ln1 + ln2 + ln3 + ln4
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc=0, bbox_to_anchor=(0.9, -0.2), ncol=4)
    return fig, ax



def simulate_dropout(X, p): 
    return X * (torch.rand_like(X) < p).float()



from grape import impute_features
@torch.no_grad()
def simulated_dropout_analysis(model, impute_model, target_X, x, train_edge_attr, train_edge_index, edge_index, n_row, n_col, 
        ps = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    ): 
    metrics = []
    n = train_edge_attr.shape[0]//2
    for p in ps: 
        sim_edge_attr = simulate_dropout(train_edge_attr[:n], p) 
        sim_edge_attr = torch.cat([sim_edge_attr, sim_edge_attr])
        x_emb, X = impute_features(model, impute_model, x, 
                                sim_edge_attr, train_edge_index, edge_index, n_row, n_col
                            )
        metrics.append(similarity_metrics(target_X, X))
    return ps, metrics
    fig, ax = plt.subplots()
    for m in ['cosine', 'l1', 'mse']: 
        ax.plot(ps, [ms[m+'_median'] for ms in metrics], label=m)
    ax.set_xlabel('dropout rate')
    ax.legend() 
    return fig, ax