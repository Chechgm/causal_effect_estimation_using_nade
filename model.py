"""Defines the neural network and the loss function. If there was any accuracy metric it should also be included here"""

import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions.bernoulli import Bernoulli

class Kidney_net(nn.Module):
    def __init__(self, N_HU):
        super().__init__()

        # Hidden layers
        self.hidden_L = nn.Linear(1, N_HU) # Kidney stone is not related to anything, so rece3ives constant as input of size 1
        self.hidden_T = nn.Linear(1, N_HU) # Treatment is affected by size of the stone so receives variable L of size 1 as input
        self.hidden_R = nn.Linear(2, N_HU) # Recovery is affected both by size and treatment, so receives two variables of size 1 as input

        # Output layers: all the variables in this case have only one parameter p as output
        self.out_L = nn.Linear(N_HU, 1)
        self.out_T = nn.Linear(N_HU, 1)
        self.out_R = nn.Linear(N_HU, 1)

    def forward(self, x):
        const = torch.ones_like(x[:,0]) # Constant for the exogenous variables

        # We have to use the following "view" because of the input shape
        h_L = F.relu(self.hidden_L(const.view(-1,1)))
        h_T = F.relu(self.hidden_T(x[:,1].view(-1,1)))
        h_R = F.relu(self.hidden_R(x[:,[0,1]].view(-1,2)))

        #h_L = torch.sigmoid(self.hidden_L(const.view(-1,1)))
        #h_T = torch.sigmoid(self.hidden_T(x[:,1].view(-1,1)))
        #h_R = torch.sigmoid(self.hidden_R(x[:,[0,1]].view(-1,2)))

        o_L = torch.sigmoid(self.out_L(h_L))
        o_T = torch.sigmoid(self.out_T(h_T))
        o_R = torch.sigmoid(self.out_R(h_R))

        return o_L, o_T, o_R

def neg_loglik(output, x):
    """
    Compute the negative log-likelihood of our data given the output parameters and the data
    """
    p_L, p_T, p_R = output # Unpack the parameters of the distributions

    # Define the distributions
    dist_L = Bernoulli(p_L)
    dist_T = Bernoulli(p_T)
    dist_R = Bernoulli(p_R)

    # Estimate the log-likelihoods
    NLL = -torch.mean(dist_L.log_prob(x[:,0].view(-1,1)) + dist_T.log_prob(x[:,1].view(-1,1)) + dist_R.log_prob(x[:,2].view(-1,1)))

    return NLL
