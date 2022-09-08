import numpy as np
import math
import itertools
import scipy as sp
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_geometric
from torch_scatter import scatter_mean, scatter_max, scatter_sum
from torch_geometric.utils import to_dense_adj
from torch.nn import Embedding

import pdb

# Is this stuff relevant? GIN, MPNN, CoordMPNN
# GIN model doesnt seem to work... seems like the node embeddings converge to each other and prediction will collapse onto one class
#     the more layers, the worse it becomes 

class Graph(object):
    def __init__(self, edge_index, x, y):
        """ Graph structure 
            for a mini-batch it will store a big (sparse) graph 
            representing the entire batch
        Args:
            x: node features  [num_nodes x num_feats]
            y: graph labels   [num_graphs]
            edge_index: list of edges [2 x num_edges]
        """
        self.edge_index = edge_index
        self.x = x.to(torch.float32)
        self.y = y
        self.num_nodes = self.x.shape[0]

    #ignore this for now, it will be useful for batching
    def set_batch(self, batch):
        """ list of ints that maps each node to the graph it belongs to
            e.g. for batch = [0,0,0,1,1,1,1]: the first 3 nodes belong to graph_0 while
            the last 4 belong to graph_1
        """
        self.batch = batch

    # this function return a sparse tensor
    def get_adjacency_matrix(self):
        """ from the list of edges create 
        a num_nodes x num_nodes sparse adjacency matrix
        """
        return torch.sparse.LongTensor(self.edge_index, 
                              # we work with a binary adj containing 1 if an edge exist
                              torch.ones((self.edge_index.shape[1])), 
                              torch.Size((self.num_nodes, self.num_nodes))
                              )

class GINLayer(nn.Module):
    """A single GIN layer, implementing MLP(AX + (1+eps)X)"""
    def __init__(self, in_feats: int, out_feats: int, hidden_dim: int, eps: float=0.0):
        super(GINLayer, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        # ============ YOUR CODE HERE =============
        # epsilon should be a learnable parameter
        self.eps = nn.Parameter(torch.Tensor([eps]))
        # =========================================
        self.linear1 = nn.Linear(self.in_feats, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, self.out_feats)

    def forward(self, x, adj_sparse): 
        # ============ YOUR CODE HERE =============
        # aggregate the neighbours as in GIN: (AX + (1+eps)X)
        x = torch.sparse.mm(adj_sparse, x) + (1 + self.eps) * x
        
        # project the features (MLP_k)
        x = self.linear1(x)
        x = self.linear2(x)
        # =========================================
        return x

class GIN(nn.Module):
    """ 
    A Graph Neural Network containing GIN layers 
    as in https://arxiv.org/abs/1810.00826 
    The readout function used to obtain graph-lvl representations
    aggregate pred from multiple layers (as in JK-Net)

    Args:
    input_dim (int): Dimensionality of the input feature vectors
    output_dim (int): Dimensionality of the output softmax distribution
    num_layers (int): Number of layers
    """
    def __init__(self, input_dim, hidden_dim, predmodel, num_layers=2, eps=0.0, \
                 molecular=True):
        super(GIN, self).__init__()
        self.num_layers = num_layers 
        self.molecular = molecular
        # nodes in ZINC dataset are characterised by one integer (atom category)
        # we will create embeddings from the categorical features using nn.Embedding
        if self.molecular:
            self.embed_x = Embedding(28, hidden_dim)
        else:
            self.embed_x = nn.Linear(input_dim, hidden_dim)
            # self.embed_x = nn.Linear(input_dim, input_dim)

        self.layers = [GINLayer(hidden_dim, hidden_dim, hidden_dim, eps) for _ in range(num_layers)]
        self.layers = nn.ModuleList(self.layers)
        
        self.pred = predmodel 
        # self.pred = SimpleMLP(hidden_dim, output_dim, hidden_dim)
        # could try inputting all intermediate layers, num_layers * hidden_dim -> output_dim
        total_dim = num_layers*hidden_dim
        # self.pred = SimpleMLP(total_dim, output_dim, total_dim)
        # self.pred = nn.Sequential(
        #     SimpleMLP(total_dim, total_dim, total_dim), 
        #     SimpleMLP(total_dim, total_dim, total_dim),
        #     SimpleMLP(total_dim, total_dim, hidden_dim),
        # )
        # =========================================

    def forward(self, graph):
        adj_sparse = graph.get_adjacency_matrix()
        if self.molecular:
            x = self.embed_x(graph.x.long()).squeeze(1)
        else:
            x = self.embed_x(graph.x)

        # ============ YOUR CODE HERE ============= 
        # perform the forward pass with the new readout function  
        hk = []
        for i in range(self.num_layers):
            x = self.layers[i](x, adj_sparse)
            x = F.relu(x)
            hk.append(x)

        y_hat = self.pred(x)
        return y_hat, x

        # x_concat = torch.cat(hk, axis=1)
        # y_hat = self.pred(x_concat)
        # return y_hat, x

        y_hat = torch.stack([self.pred(h) for h in hk])
        # print(y_hat.mean(axis=0).shape)
        return y_hat.mean(axis=0), x

def train(model, x, y, edge_index, n_train,
    lr = 0.001, num_epochs = 1, device='cpu'):
    # TODO get rid of dependency on Graph
    graph = Graph(edge_index, x, y)
    # full batch training for now 
    graph.set_batch(torch.zeros((x.shape[0]), dtype=torch.int64))

    optimiser = optim.Adam(model.parameters(), lr=lr)
    losses = []
    for epoch in range(num_epochs):
        model.train()
        # TODO minibatch data
        # num_iter = int(len(dataset)/BATCH_SIZE)
        # for i in range(num_iter):
        #     batch_list = dataset[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        #     batch = create_mini_batch(batch_list)
        #     optimiser.zero_grad()
        #     y_hat, _ = model(batch)
        #     loss = loss_fct(y_hat, batch.y)
        #     metric = metric_fct(y_hat, batch.y)
        #     loss.backward()
        #     optimiser.step() 
        #     if (i+1) % print_every == 0:
        #       print(f"Epoch {epoch} Iter {i}/{num_iter}",
        #                 f"Loss train {loss.data}; Metric train {metric.data}")
        y_hat, _ = model(graph)
        train_loss = F.cross_entropy(y_hat[:n_train], graph.y[:n_train])
        train_loss.backward()
        optimiser.step() 
        losses.append(train_loss.item())
    return losses

def eval(model, x, y, edge_index, n_train): 
    # TODO get rid of dependency on Graph
    graph = Graph(edge_index, x, y)
    # full batch training for now 
    graph.set_batch(torch.zeros((x.shape[0]), dtype=torch.int64))
    model.eval()
    with torch.no_grad():
        y_hat, _ = model(graph)
        y_hat = y_hat[n_train:]
        num_correct = y_hat.max(axis=1).indices.eq(graph.y[n_train:].max(axis=1).indices).sum()
        num_total = y_hat.shape[0]
        accuracy = 100.0 * (num_correct/num_total)
        print('accuracy:', accuracy)
    return accuracy

# TODO put data into PyG Data object: data.x, data.edge_index, data.edge_attr, data.pos, data.y

# knn -> GNN -> cell prediction
# data -> adjacency + cell prediction 
# neural relational inference, knn graphs, dynamic graph CNN, differentiable graph module (RL), pointer graph networks (supervised learning), SLAPS
