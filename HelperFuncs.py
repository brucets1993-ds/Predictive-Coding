import torch
import torch.nn as nn

def build_data(n, dims=400):
    """
    This function randomly generates data for training.
    The intention here is to simply create data with several key properties:
      - The targets (y) are directly determined by the features (x)
        - This ensures that there should be some learnable relatinships between x and y
      - The relationship is sufficently complex that we can only realisticly hope to find an approximate mapping F(x)->y
    
    Inputs:
      n - integer: The number of samples to generate
      dims - integer: The number of dimensions each sample has

    Returns:
      x - torch.Tensor of size (n, dims): The inputs' features
      y - torch.Tensor of size (n,): The target values
    """
    x = torch.randn(n,dims)
    x_sin = torch.sin(x)
    x_cos = torch.cos(x)
    x_2_p1 = x**2+1
    x_interweave = x[:,::2]*x[:,1::2]
    x_double_interweave = x_interweave[:,::2]*x_interweave[:,1::2]
    x_double_sin = x_double_interweave*x_sin[:,::4]+x_double_interweave*x_sin[:,1::4]+x_double_interweave*x_sin[:,2::4]+x_double_interweave*x_sin[:,3::4]
    x_double_cos = x_double_interweave*x_cos[:,::4]+x_double_interweave*x_cos[:,1::4]+x_double_interweave*x_cos[:,2::4]+x_double_interweave*x_cos[:,3::4]
    x_double_2p1 = x_double_interweave*x_2_p1[:,::4]+x_double_interweave*x_2_p1[:,1::4]+x_double_interweave*x_2_p1[:,2::4]+x_double_interweave*x_2_p1[:,3::4]
    x_combined = x_double_sin+x_double_cos+x_double_2p1
    x_plus = x[:,::4]+x[:,1::4]
    x_minus = x[:,2::4]-x[:,3::4]
    x_all = (torch.cos(x_plus*x_combined)+torch.cos(x_minus*x_combined))
    y = x_all.sum(dim=1)/(x_all**2).max(dim=1).values
    return x, y


class PredictiveChanger(nn.Module):
    def __init__(self,input_dims,hidden_dims,n_layers,pseudo_limit):
        """
        For this example, we will implement a simple sequential model with dense Linear layers and ReLu activation
        """
        super().__init__()
        self.act = nn.ReLU()
        self.layer1 = nn.Linear(input_dims,hidden_dims)
        self.layers = nn.ModuleList([nn.Linear(hidden_dims,hidden_dims) for _ in range(n_layers)])
        self.final = nn.Linear(hidden_dims,1)
        self.pseudo_limit = pseudo_limit

    def forward(self,x):
        next_predicted = self.act(self.layer1(x))
        #We need to track the output of each layer, this will be necessary when we implement the predictive optimization algorithm
        predicted_outputs = [torch.clone(next_predicted).detach()]
        for layer in self.layers:
            next_predicted = self.act(layer(next_predicted))
            predicted_outputs.append(torch.clone(next_predicted).detach())
        final = self.final(next_predicted)
        predicted_outputs.append(torch.clone(final).detach())
        return final, predicted_outputs
    
    def predictive_optimize(self,x,preds,actuals,optimizers,objective):
        loss = objective(self.final(preds[-2]).ravel(),actuals.ravel())
        loss.backward()
        optimizers[-1].step()
        optimizers[-1].zero_grad()
        actuals = torch.clone(torch.matmul(actuals.reshape((-1,1))-self.final.bias,torch.linalg.pinv(self.final.weight.T))).detach()
        #I added this logic to avoid numerical instability, in practice the pseudo inverse will often blow up, so we trim values to avoid this
        while torch.abs(actuals).max()>self.pseudo_limit:
            actuals = actuals/2
        for layer_input, optim, layer in zip(preds[:-2][::-1],optimizers[1:-1][::-1], self.layers[::-1]):
            loss = objective(layer(layer_input).ravel(),actuals.ravel())
            loss.backward()
            optim.step()
            optim.zero_grad()
            actuals = torch.clone(torch.matmul(actuals-layer.bias,torch.linalg.pinv(layer.weight))).detach()
            while torch.abs(actuals).max()>self.pseudo_limit:
                actuals = actuals/2
        loss = objective(self.layer1(x).ravel(),actuals.ravel())
        loss.backward()
        optimizers[0].step()
        optimizers[0].zero_grad()