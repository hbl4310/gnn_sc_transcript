import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class SimpleMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(SimpleMLP, self).__init__()
        self.layer_1 = nn.Linear(input_dim, hidden_dim)
        self.layer_2 = nn.Linear(hidden_dim, output_dim)
        self.seq = nn.Sequential(
            self.layer_1, nn.ReLU(), self.layer_2
        )

    def forward(self, x):
        x = self.seq(x)
        y_hat = F.log_softmax(x, dim=1)
        # y_hat = F.softmax(x, dim=1)
        return y_hat


# TODO support DataLoader and mini-batches
def train(model, x_train, y_train, 
    lr = 0.001, num_epochs = 1, device='cpu'):

    optimiser = optim.Adam(model.parameters(), lr=lr)

    losses = []
    for epoch in range(num_epochs):
        model.train()
        optimiser.zero_grad()
        y_hat = model(x_train)
        loss = F.cross_entropy(y_hat, y_train)
        loss.backward()
        optimiser.step()
        losses.append(loss.item())

    return losses


def eval(model, x_test, y_test):
    model.eval()
    with torch.no_grad():
        y_hat = model(x_test)
        num_correct = y_hat.max(axis=1).indices.eq(y_test.max(axis=1).indices).sum()
        num_total = y_hat.shape[0]
        accuracy = 100.0 * (num_correct/num_total)
        print('accuracy:', accuracy.item())
    return accuracy