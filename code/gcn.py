import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class GCNLayer(nn.Module):
    """GCN layer, but with a larger receptive field

    Args:
        input_dim (int): Dimensionality of the input feature vectors
        output_dim (int): Dimensionality of the output softmax distribution
        A (torch.Tensor): 2-D adjacency matrix
    """
    def __init__(self, input_dim, output_dim, A, k=1):
        super(GCNLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.A = A
        self.k = k 

        # Compute symmetric norm
        Atilde = torch.matrix_power(A + torch.eye(A.shape[0]), k)
        Dtilde_invsqrt = torch.diag(Atilde.sum(0).pow(-1/2))
        self.Anorm = (torch.matmul(torch.matmul(Dtilde_invsqrt, Atilde), Dtilde_invsqrt))
        # + Simple linear transformation and non-linear activation
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.linear(torch.matmul(self.Anorm, x))
        x = F.softmax(x, dim=1)
        return x

class SimpleGNN(nn.Module):
    """Simple GNN model using the GCNLayer implemented by students

    Args:
        input_dim (int): Dimensionality of the input feature vectors
        output_dim (int): Dimensionality of the output softmax distribution
        A (torch.Tensor): 2-D adjacency matrix
    """
    def __init__(self, input_dim, output_dim, A, gcnlayer = GCNLayer, **kwargs):
        super(SimpleGNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.A = A
        self.gcn_layer = gcnlayer(input_dim, output_dim, A, **kwargs)

    def forward(self, x):
        x = self.gcn_layer(x)
        # y_hat = F.log_softmax(x, dim=1) <- old version
        # y_hat = x
        y_hat = F.softmax(x, dim=1)
        return y_hat

# TODO support DataLoader and mini-batches
# implicitly transductive
def train(model, x, y, n_train,
    lr = 0.001, num_epochs = 1, log_interval=100, device='cpu'):
    optimiser = optim.Adam(model.parameters(), lr=lr)
    losses = []
    for epoch in range(num_epochs):
        model.train()
        optimiser.zero_grad()
        y_hat = model(x)[:n_train]
        loss = F.cross_entropy(y_hat, y[:n_train])
        loss.backward()
        optimiser.step()
        losses.append(loss.item())
        if epoch % log_interval == 0: 
            print(f'[EPOCH {epoch}] train loss: {loss.item()}')
    return losses

def eval(model, x, y, n_train): 
    model.eval()
    with torch.no_grad():
        y_hat = model(x)[n_train:]
        num_correct = y_hat.max(axis=1).indices.eq(y[n_train:].max(axis=1).indices).sum()
        num_total = len(y_hat)
        accuracy = 100.0 * (num_correct/num_total)
        print('accuracy:', accuracy.item())
    return accuracy
