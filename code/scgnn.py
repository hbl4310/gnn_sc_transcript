# https://github.com/juexinwang/scGNN

import scanpy as sc
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import Dataset 
import scipy

class scDataset(Dataset):
    def __init__(self, data: sc.AnnData, transform=None):
        """
        Args:
            data : AnnData
            transform (callable, optional):
        """
        # Now lines are cells, and cols are genes
        # self.features = data.transpose()
        self.features = data.X

        # save nonzero
        # self.nz_i,self.nz_j = self.features.nonzero()
        self.transform = transform

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.features[idx, :]
        if type(sample) == scipy.sparse.lil_matrix:
            sample = torch.from_numpy(sample.toarray())
        else:
            sample = torch.from_numpy(sample)

        # transform after get the data
        if self.transform:
            sample = self.transform(sample)

        return sample, idx


# https://github.com/juexinwang/scGNN/blob/09ac50ba0a2bbf87535613a2d284872074804d02/model.py
class AE(nn.Module):
    ''' Autoencoder for dimensional reduction'''
    def __init__(self,dim):
        super(AE, self).__init__()
        self.dim = dim
        self.fc1 = nn.Linear(dim, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 512)
        self.fc4 = nn.Linear(512, dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return F.relu(self.fc2(h1))

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.relu(self.fc4(h3))

    def forward(self, x):
        z = self.encode(x.view(-1, self.dim))
        return self.decode(z), z

class VAE(nn.Module):
    ''' Variational Autoencoder for dimensional reduction'''
    def __init__(self,dim):
        super(VAE, self).__init__()
        self.dim = dim
        self.fc1 = nn.Linear(dim, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.dim))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z


def get_enum(reduction):
    # type: (str) -> int
    if reduction == 'none':
        ret = 0
    elif reduction == 'mean':
        ret = 1
    elif reduction == 'elementwise_mean':
        print("reduction='elementwise_mean' is deprecated, please use reduction='mean' instead.")
        ret = 1
    elif reduction == 'sum':
        ret = 2
    else:
        ret = -1  # TODO: remove once JIT exceptions support control flow
        raise ValueError(
            "{} is not a valid value for reduction".format(reduction))
    return ret

def legacy_get_string(size_average, reduce, emit_warning=True):
    # type: (Optional[bool], Optional[bool], bool) -> str
    warning = "size_average and reduce args will be deprecated, please use reduction='{}' instead."

    if size_average is None:
        size_average = True
    if reduce is None:
        reduce = True

    if size_average and reduce:
        ret = 'mean'
    elif reduce:
        ret = 'sum'
    else:
        ret = 'none'
    if emit_warning:
        print(warning.format(ret))
    return ret

def regulation_mse_loss_function(input, target, regulationMatrix, size_average=None, reduce=None, reduction='mean'):
    # type: (Tensor, Tensor, str, Optional[bool], Optional[bool], str) -> Tensor
    r"""regulation_mse_loss_function(input, target, regulationMatrix, regularizer_type, size_average=None, reduce=None, reduction='mean') -> Tensor
    Measures the element-wise mean squared error for regulation input, now only support LTMG.
    See :revised from pytorch class:`~torch.nn.MSELoss` for details.
    """
    if not (target.size() == input.size()):
        print("Using a target size ({}) that is different to the input size ({}). "
              "This will likely lead to incorrect results due to broadcasting. "
              "Please ensure they have the same size.".format(target.size(), input.size()))
    if size_average is not None or reduce is not None:
        reduction = legacy_get_string(size_average, reduce)
    # Now it use regulariz type to distinguish, it can be imporved later
    ret = (input - target) ** 2
    # ret = (0.001*input - 0.001*target) ** 2
    ret = torch.mul(ret, regulationMatrix)
    if reduction != 'none':
        ret = torch.mean(ret) if reduction == 'mean' else torch.sum(ret)
    return ret

def regulation01_mse_loss_function(input, target, regulationMatrix, size_average=None, reduce=None, reduction='mean'):
    # type: (Tensor, Tensor, str, Optional[bool], Optional[bool], str) -> Tensor
    r"""regulation_mse_loss_function(input, target, regulationMatrix, regularizer_type, size_average=None, reduce=None, reduction='mean') -> Tensor
    Measures the element-wise mean squared error for regulation input, now only support LTMG.
    See :revised from pytorch class:`~torch.nn.MSELoss` for details.
    """
    if not (target.size() == input.size()):
        print("Using a target size ({}) that is different to the input size ({}). "
              "This will likely lead to incorrect results due to broadcasting. "
              "Please ensure they have the same size.".format(target.size(), input.size()))
    if size_average is not None or reduce is not None:
        reduction = legacy_get_string(size_average, reduce)
    # Now it use regulariz type to distinguish, it can be imporved later
    ret = (input - target) ** 2
    # ret = (0.001*input - 0.001*target) ** 2
    regulationMatrix[regulationMatrix > 0] = 1
    ret = torch.mul(ret, regulationMatrix)
    if reduction != 'none':
        ret = torch.mean(ret) if reduction == 'mean' else torch.sum(ret)
    return ret

def graph_mse_loss_function(input, target, graphregu, size_average=None, reduce=None, reduction='mean'):
    # type: (Tensor, Tensor, Optional[bool], Optional[bool], str) -> Tensor
    r"""graph_mse_loss_function(input, target, adj, regularizer_type, size_average=None, reduce=None, reduction='mean') -> Tensor
    Measures the element-wise mean squared error in graph regularizor.
    See:revised from pytorch class:`~torch.nn.MSELoss` for details.
    """
    if not (target.size() == input.size()):
        print("Using a target size ({}) that is different to the input size ({}). "
              "This will likely lead to incorrect results due to broadcasting. "
              "Please ensure they have the same size.".format(target.size(), input.size()))
    if size_average is not None or reduce is not None:
        reduction = legacy_get_string(size_average, reduce)
    # Now it use regulariz type to distinguish, it can be imporved later
    ret = (input - target) ** 2
    # ret = (0.001*input - 0.001*target) ** 2
    # if graphregu != None:
    # print(graphregu.type())
    # print(ret.type())
    ret = torch.matmul(graphregu, ret)
    if reduction != 'none':
        ret = torch.mean(ret) if reduction == 'mean' else torch.sum(ret)
    return ret

def loss_function_graph_celltype(recon_x, x, mu, logvar, graphregu=None, celltyperegu=None, gammaPara=1.0, regulationMatrix=None, regularizer_type='noregu', reguPara=0.001, reguParaCelltype=0.001, modelusage='AE', reduction='sum'):
    '''
    Regularized by the graph information
    Reconstruction + KL divergence losses summed over all elements and batch
    '''
    # Original
    # BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    # Graph
    target = x
    if regularizer_type == 'Graph' or regularizer_type == 'LTMG' or regularizer_type == 'LTMG01' or regularizer_type == 'Celltype':
        target.requires_grad = True
    # Euclidean
    # BCE = gammaPara * vallina_mse_loss_function(recon_x, target, reduction='sum')
    BCE = gammaPara * \
        vallina_mse_loss_function(recon_x, target, reduction=reduction)
    if regularizer_type == 'noregu':
        loss = BCE
    elif regularizer_type == 'LTMG':
        loss = BCE + reguPara * \
            regulation_mse_loss_function(
                recon_x, target, regulationMatrix, reduction=reduction)
    elif regularizer_type == 'LTMG01':
        loss = BCE + reguPara * \
            regulation01_mse_loss_function(
                recon_x, target, regulationMatrix, reduction=reduction)
    elif regularizer_type == 'Graph':
        loss = BCE + reguPara * \
            graph_mse_loss_function(
                recon_x, target, graphregu=graphregu, reduction=reduction)
    elif regularizer_type == 'Celltype':
        loss = BCE + reguPara * graph_mse_loss_function(recon_x, target, graphregu=graphregu, reduction=reduction) + \
            reguParaCelltype * \
            graph_mse_loss_function(
                recon_x, target, graphregu=celltyperegu, reduction=reduction)
    elif regularizer_type == 'CelltypeR':
        loss = BCE + (1-gammaPara) * regulation01_mse_loss_function(recon_x, target, regulationMatrix, reduction=reduction) + reguPara * graph_mse_loss_function(recon_x,
                                                                                                                                                                 target, graphregu=graphregu, reduction=reduction) + reguParaCelltype * graph_mse_loss_function(recon_x, target, graphregu=celltyperegu, reduction=reduction)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    if modelusage == 'VAE':
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = loss + KLD

    return loss

def vallina_mse_loss_function(input, target, size_average=None, reduce=None, reduction='mean'):
    # type: (Tensor, Tensor, Optional[bool], Optional[bool], str) -> Tensor
    r"""vallina_mse_loss_function(input, target, size_average=None, reduce=None, reduction='mean') -> Tensor
    Original: Measures the element-wise mean squared error.
    See :revised from pytorch class:`~torch.nn.MSELoss` for details.
    """
    if not (target.size() == input.size()):
        print("Using a target size ({}) that is different to the input size ({}). "
              "This will likely lead to incorrect results due to broadcasting. "
              "Please ensure they have the same size.".format(target.size(), input.size()))
    if size_average is not None or reduce is not None:
        reduction = legacy_get_string(size_average, reduce)
    # Now it use regulariz type to distinguish, it can be imporved later
    # Original, for not require grads, using c++ version
    # However, it has bugs there, different number of cpu cause different results because of MKL parallel library
    # Not known yet whether GPU has same problem.
    # Solution 1: set same number of cpu when running, it works for reproduce everything but not applicable for other users
    # https://pytorch.org/docs/stable/torch.html#torch.set_num_threads
    # https://pytorch.org/docs/stable/notes/cpu_threading_torchscript_inference.html
    # Solution 2: not use C++ codes, as we did here.
    # https://github.com/pytorch/pytorch/issues/8710

    if target.requires_grad:
        ret = (input - target) ** 2
        # 0.001 to reduce float loss
        # ret = (0.001*input - 0.001*target) ** 2
        if reduction != 'none':
            ret = torch.mean(ret) if reduction == 'mean' else torch.sum(ret)
    else:
        expanded_input, expanded_target = torch.broadcast_tensors(
            input, target)
        ret = torch._C._nn.mse_loss(
            expanded_input, expanded_target, get_enum(reduction))

    # ret = (input - target) ** 2
    # # 0.001 to reduce float loss
    # # ret = (0.001*input - 0.001*target) ** 2
    # if reduction != 'none':
    #     ret = torch.mean(ret) if reduction == 'mean' else torch.sum(ret)
    return ret

def loss_function_graph(recon_x, x, mu, logvar, graphregu=None, gammaPara=1.0, regulationMatrix=None, regularizer_type='noregu', reguPara=0.001, modelusage='AE', reduction='sum'):
    '''
    Regularized by the graph information
    Reconstruction + KL divergence losses summed over all elements and batch
    '''
    # Original
    # BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    # Graph
    target = x
    if regularizer_type == 'Graph' or regularizer_type == 'LTMG' or regularizer_type == 'LTMG01':
        target.requires_grad = True
    # Euclidean
    # BCE = gammaPara * vallina_mse_loss_function(recon_x, target, reduction='sum')
    BCE = gammaPara * vallina_mse_loss_function(recon_x, target, reduction=reduction)
    if regularizer_type == 'noregu':
        loss = BCE
    elif regularizer_type == 'LTMG':
        loss = BCE + reguPara * \
            regulation_mse_loss_function(
                recon_x, target, regulationMatrix, reduction=reduction)
    elif regularizer_type == 'LTMG01':
        loss = BCE + reguPara * \
            regulation01_mse_loss_function(
                recon_x, target, regulationMatrix, reduction=reduction)
    elif regularizer_type == 'Graph':
        loss = BCE + reguPara * \
            graph_mse_loss_function(
                recon_x, target, graphregu=graphregu, reduction=reduction)
    elif regularizer_type == 'GraphR':
        loss = BCE + reguPara * \
            graph_mse_loss_function(
                recon_x, target, graphregu=1-graphregu, reduction=reduction)
    elif regularizer_type == 'LTMG-Graph':
        loss = BCE + reguPara * regulation_mse_loss_function(recon_x, target, regulationMatrix, reduction=reduction) + \
            reguPara * \
            graph_mse_loss_function(
                recon_x, target, graphregu=graphregu, reduction=reduction)
    elif regularizer_type == 'LTMG-GraphR':
        loss = BCE + reguPara * regulation_mse_loss_function(recon_x, target, regulationMatrix, reduction=reduction) + \
            reguPara * \
            graph_mse_loss_function(
                recon_x, target, graphregu=1-graphregu, reduction=reduction)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    if modelusage == 'VAE':
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = loss + KLD

    return loss


def train(model, epoch, train_loader, EMFlag=False, 
    taskType='celltype', 
    sparseImputation='nonsparse', 
    l1reg = 0., l2reg = 0., 
    regulized_type = 'noregu',          # regulized type (default: LTMG) in EM, otherwise: noregu/LTMG/LTMG01
    gammaPara=0.1,                      # regulized intensity 
    alphaRegularizePara=0.9,            # regulized parameter 
    reduction='sum',                    # reduction type: mean/sum
    EMreguTag = False,                  # whether regu in EM process
    log_interval=100, epoch_log_interval=100, device='cpu', 
    precisionModel='Float'
    ):
    '''
    EMFlag indicates whether in EM processes. 
        If in EM, use regulized-type parsed from program entrance,
        Otherwise, noregu
        taskType: celltype or imputation
    '''
    model.train()
    optimiser = optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    train_loss = 0
    for batch_idx, (data, dataindex) in enumerate(train_loader):
        if precisionModel == 'Double':
            data = data.type(torch.DoubleTensor)
        elif precisionModel == 'Float':
            data = data.type(torch.FloatTensor)
        data = data.to(device)
        if not regulized_type == 'noregu':
            regulationMatrixBatch = regulationMatrix[dataindex, :]
            regulationMatrixBatch = regulationMatrixBatch.to(device)
        else:
            regulationMatrixBatch = None
        if taskType == 'imputation':
            if sparseImputation == 'nonsparse':
                celltypesampleBatch = celltypesample[dataindex,
                                                     :][:, dataindex]
                adjsampleBatch = adjsample[dataindex, :][:, dataindex]
            elif sparseImputation == 'sparse':
                celltypesampleBatch = generateCelltypeRegu(
                    listResult[dataindex])
                celltypesampleBatch = torch.from_numpy(celltypesampleBatch)
                if precisionModel == 'Float':
                    celltypesampleBatch = celltypesampleBatch.float()
                elif precisionModel == 'Double':
                    celltypesampleBatch = celltypesampleBatch.type(
                        torch.DoubleTensor)
                # mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                # print('celltype Mem consumption: '+str(mem))

                adjsampleBatch = adj[dataindex, :][:, dataindex]
                adjsampleBatch = sp.csr_matrix.todense(adjsampleBatch)
                adjsampleBatch = torch.from_numpy(adjsampleBatch)
                if precisionModel == 'Float':
                    adjsampleBatch = adjsampleBatch.float()
                elif precisionModel == 'Double':
                    adjsampleBatch = adjsampleBatch.type(torch.DoubleTensor)
                # mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                # print('adj Mem consumption: '+str(mem))

        optimiser.zero_grad()
        if type(model).__name__ == 'VAE':
            recon_batch, mu, logvar, z = model(data)
            if taskType == 'celltype':
                if EMFlag and (not EMreguTag):
                    loss = loss_function_graph(recon_batch, data.view(-1, recon_batch.shape[1]), mu, logvar, 
                                               gammaPara=gammaPara, regulationMatrix=regulationMatrixBatch,
                                               regularizer_type='noregu', reguPara=alphaRegularizePara, modelusage='VAE', reduction=reduction)
                else:
                    loss = loss_function_graph(recon_batch, data.view(-1, recon_batch.shape[1]), mu, logvar, 
                                               gammaPara=gammaPara, regulationMatrix=regulationMatrixBatch,
                                               regularizer_type=regulized_type, reguPara=alphaRegularizePara, modelusage='VAE', reduction=reduction)
            elif taskType == 'imputation':
                if EMFlag and (not EMreguTag):
                    loss = loss_function_graph_celltype(recon_batch, data.view(-1, recon_batch.shape[1]), mu, logvar, graphregu=adjsampleBatch, celltyperegu=celltypesampleBatch, gammaPara=args.gammaImputePara,
                                                        regulationMatrix=regulationMatrixBatch, regularizer_type=args.EMregulized_type, reguPara=args.graphImputePara, reguParaCelltype=args.celltypeImputePara, modelusage=args.model, reduction=reduction)
                else:
                    loss = loss_function_graph_celltype(recon_batch, data.view(-1, recon_batch.shape[1]), mu, logvar, graphregu=adjsampleBatch, celltyperegu=celltypesampleBatch, gammaPara=args.gammaImputePara,
                                                        regulationMatrix=regulationMatrixBatch, regularizer_type=regulized_type, reguPara=args.graphImputePara, reguParaCelltype=args.celltypeImputePara, modelusage=args.model, reduction=reduction)

        elif type(model).__name__ == 'AE':
            recon_batch, z = model(data)
            mu_dummy = ''
            logvar_dummy = ''
            if taskType == 'celltype':
                if EMFlag and (not EMreguTag):
                    loss = loss_function_graph(recon_batch, data.view(-1, recon_batch.shape[1]), mu_dummy, logvar_dummy, gammaPara=gammaPara,
                                               regulationMatrix=regulationMatrixBatch, regularizer_type='noregu', reguPara=alphaRegularizePara, modelusage='AE', reduction=reduction)
                else:
                    loss = loss_function_graph(recon_batch, data.view(-1, recon_batch.shape[1]), mu_dummy, logvar_dummy, gammaPara=gammaPara, regulationMatrix=regulationMatrixBatch,
                                               regularizer_type=regulized_type, reguPara=alphaRegularizePara, modelusage='AE', reduction=reduction)
            elif taskType == 'imputation':
                if EMFlag and (not EMreguTag):
                    loss = loss_function_graph_celltype(recon_batch, data.view(-1, recon_batch.shape[1]), mu_dummy, logvar_dummy, graphregu=adjsampleBatch, celltyperegu=celltypesampleBatch, gammaPara=args.gammaImputePara,
                                                        regulationMatrix=regulationMatrixBatch, regularizer_type=args.EMregulized_type, reguPara=args.graphImputePara, reguParaCelltype=args.celltypeImputePara, modelusage=args.model, reduction=reduction)
                else:
                    loss = loss_function_graph_celltype(recon_batch, data.view(-1, recon_batch.shape[1]), mu_dummy, logvar_dummy, graphregu=adjsampleBatch, celltyperegu=celltypesampleBatch, gammaPara=args.gammaImputePara,
                                                        regulationMatrix=regulationMatrixBatch, regularizer_type=regulized_type, reguPara=args.graphImputePara, reguParaCelltype=args.celltypeImputePara, modelusage=args.model, reduction=reduction)

        # L1 and L2 regularization
        # 0.0 for no regularization
        l1 = 0.0
        l2 = 0.0
        for p in model.parameters():
            l1 = l1 + p.abs().sum()
            l2 = l2 + p.pow(2).sum()
        loss = loss + l1reg * l1 + l2reg * l2

        loss.backward()
        train_loss += loss.item()
        optimiser.step()
        if (batch_idx+1) % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

        # for batch
        if batch_idx == 0:
            recon_batch_all = recon_batch
            data_all = data
            z_all = z
        else:
            recon_batch_all = torch.cat((recon_batch_all, recon_batch), 0)
            data_all = torch.cat((data_all, data), 0)
            z_all = torch.cat((z_all, z), 0)

    if epoch % epoch_log_interval == 0:
        print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, train_loss / len(train_loader.dataset)))

    return recon_batch_all, data_all, z_all


        # adj, edgeList = generateAdj(zOut, graphType=args.prunetype, para=args.knn_distance+':'+str(
        #     args.k), adjTag=(args.useGAEembedding or args.useBothembedding))


import time 
import numpy as np 
from scipy.spatial import distance
import networkx as nx

# https://github.com/juexinwang/scGNN/blob/09ac50ba0a2bbf87535613a2d284872074804d02/graph_function.py#L257
#para: measuareName:k:threshold
def calculateKNNgraphDistanceMatrixStatsSingleThread(featureMatrix, distanceType='euclidean', k=10, param=None):
    r"""
    Thresholdgraph: KNN Graph with stats one-std based methods, SingleThread version
    """       

    edgeList=[]
    # Version 1: cost memory, precalculate all dist

    ## distMat = distance.cdist(featureMatrix,featureMatrix, distanceType)
    ## parallel
    # distMat = pairwise_distances(featureMatrix,featureMatrix, distanceType, n_jobs=-1)
    
    # for i in np.arange(distMat.shape[0]):
    #     res = distMat[:,i].argsort()[:k+1]
    #     tmpdist = distMat[res[1:k+1],i]
    #     mean = np.mean(tmpdist)
    #     std = np.std(tmpdist)
    #     for j in np.arange(1,k+1):
    #         if (distMat[i,res[j]]<=mean+std) and (distMat[i,res[j]]>=mean-std):
    #             weight = 1.0
    #         else:
    #             weight = 0.0
    #         edgeList.append((i,res[j],weight))

    ## Version 2: for each of the cell, calculate dist, save memory 
    p_time = time.time()
    for i in np.arange(featureMatrix.shape[0]):
        if i%10000==0:
            print('Start pruning '+str(i)+'th cell, cost '+str(time.time()-p_time)+'s')
        tmp=featureMatrix[i,:].reshape(1,-1)
        distMat = distance.cdist(tmp,featureMatrix, distanceType)
        res = distMat.argsort()[:k+1]
        tmpdist = distMat[0,res[0][1:k+1]]
        boundary = np.mean(tmpdist)+np.std(tmpdist)
        for j in np.arange(1,k+1):
            # TODO: check, only exclude large outliners
            # if (distMat[0,res[0][j]]<=mean+std) and (distMat[0,res[0][j]]>=mean-std):
            if distMat[0,res[0][j]]<=boundary:
                weight = 1.0
            else:
                weight = 0.0
            edgeList.append((i,res[0][j],weight))

    # Version 3: for each of the cell, calculate dist, use heapq to accelerate
    # However, it cannot defeat sort
    # Get same results as this article
    # https://stackoverflow.com/questions/12787650/finding-the-index-of-n-biggest-elements-in-python-array-list-efficiently
    #
    # p_time = time.time()
    # for i in np.arange(featureMatrix.shape[0]):
    #     if i%10000==0:
    #         print('Start pruning '+str(i)+'th cell, cost '+str(time.time()-p_time)+'s')
    #     tmp=featureMatrix[i,:].reshape(1,-1)
    #     distMat = distance.cdist(tmp,featureMatrix, distanceType)[0]
    #     # res = distMat.argsort()[:k+1]
    #     res = heapq.nsmallest(k+1, range(len(distMat)), distMat.take)[1:k+1]
    #     tmpdist = distMat[res]
    #     boundary = np.mean(tmpdist)+np.std(tmpdist)
    #     for j in np.arange(k):
    #         # TODO: check, only exclude large outliners
    #         # if (distMat[0,res[0][j]]<=mean+std) and (distMat[0,res[0][j]]>=mean-std):
    #         if distMat[res[j]]<=boundary:
    #             weight = 1.0
    #         else:
    #             weight = 0.0
    #         edgeList.append((i,res[j],weight))
    
    return edgeList

# edgeList to edgeDict
def edgeList2edgeDict(edgeList, nodesize):
    graphdict={}
    tdict={}

    for edge in edgeList:
        end1 = edge[0]
        end2 = edge[1]
        tdict[end1]=""
        tdict[end2]=""
        if end1 in graphdict:
            tmplist = graphdict[end1]
        else:
            tmplist = []
        tmplist.append(end2)
        graphdict[end1]= tmplist

    #check and get full matrix
    for i in range(nodesize):
        if i not in tdict:
            graphdict[i]=[]

    return graphdict