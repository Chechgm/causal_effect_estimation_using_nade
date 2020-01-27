"""Defines the neural network and the loss function. If there was any accuracy metric it should also be included here"""

import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.log_normal import LogNormal
from torch.distributions.normal import Normal

###############################################################################
### Binary kidney stones neural network and negative log-likelihood
###############################################################################
class binary_ks_net(nn.Module):
    def __init__(self, N_HU, NLA):
        super().__init__()

        # Hidden layers
        self.hidden_L = nn.Linear(1, N_HU, bias=False) # Kidney stone is not related to anything, so receives constant as input of size 1
        self.hidden_L_2 = nn.Linear(N_HU, N_HU)#, bias=False) 
        
        self.hidden_T = nn.Linear(1, N_HU) # Treatment is affected by size of the stone so receives variable L of size 1 as input
        self.hidden_T_2 = nn.Linear(N_HU, N_HU) 
        
        self.hidden_R = nn.Linear(2, N_HU) # Recovery is affected both by size and treatment, so receives two variables of size 1 as input
        self.hidden_R_2 = nn.Linear(N_HU, N_HU)

        # Output layers: all the variables in this case have only one parameter p as output
        self.out_L = nn.Linear(N_HU, 1, bias=False)
        self.out_T = nn.Linear(N_HU, 1)
        self.out_R = nn.Linear(N_HU, 1)
        
        # Activation functions
        self.nla = NLA

    def forward(self, x):
        const = torch.ones_like(x[:,0]) # Constant for the exogenous variables

        # We have to use the following "view" because of the input shape
        h_L = self.nla(self.hidden_L(const.view(-1,1)))
        h_T = self.nla(self.hidden_T(x[:,0].view(-1,1)))
        h_R = self.nla(self.hidden_R(x[:,[0,1]].view(-1,2)))

        h_L = self.nla(self.hidden_L_2(h_L))
        h_T = self.nla(self.hidden_T_2(h_T))
        h_R = self.nla(self.hidden_R_2(h_R))

        p_L = torch.sigmoid(self.out_L(h_L))
        p_T = torch.sigmoid(self.out_T(h_T))
        p_R = torch.sigmoid(self.out_R(h_R))

        return p_L, p_T, p_R

def binary_neg_loglik(output, x):
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

###############################################################################
### Continuous recovery neural network and negative log-likelihood          ###
###############################################################################
class cont_rec_ks_net(nn.Module):
    def __init__(self, N_HU, NLA):
        super().__init__()

        # Hidden layers
        self.hidden_L = nn.Linear(1, N_HU, bias=False) # Kidney stone is not related to anything, so receives constant as input of size 1
        self.hidden_L_2 = nn.Linear(N_HU, N_HU)#, bias=False) 
        
        self.hidden_T = nn.Linear(1, N_HU) # Treatment is affected by size of the stone so receives variable L of size 1 as input
        self.hidden_T_2 = nn.Linear(N_HU, N_HU) 
        
        self.hidden_R = nn.Linear(2, N_HU) # Recovery is affected both by size and treatment, so receives two variables of size 1 as input
        self.hidden_R_2 = nn.Linear(N_HU, N_HU)

        # Output layers: all the variables in this case have only one parameter p as output
        self.out_L = nn.Linear(N_HU, 1, bias=False)
        self.out_T = nn.Linear(N_HU, 1)
        self.out_R_a = nn.Linear(N_HU, 1)
        self.out_R_b = nn.Linear(N_HU, 1)

        # Activation functions
        self.nla = NLA

    def forward(self, x):
        const = torch.ones_like(x[:,0]) # Constant for the exogenous variables

        # We have to use the following "view" because of the input shape
        h_L = self.nla(self.hidden_L(const.view(-1,1)))
        h_T = self.nla(self.hidden_T(x[:,0].view(-1,1)))
        h_R = self.nla(self.hidden_R(x[:,[0,1]].view(-1,2)))

        h_L = self.nla(self.hidden_L_2(h_L))
        h_T = self.nla(self.hidden_T_2(h_T))
        h_R = self.nla(self.hidden_R_2(h_R))

        p_L = torch.sigmoid(self.out_L(h_L))
        p_T = torch.sigmoid(self.out_T(h_T))
        a_R = self.out_R_a(h_R) # a and b are real valued parameters that can be transformed into strictly positive numbers if needed
        b_R = self.out_R_b(h_R)

        return p_L, p_T, a_R, b_R

def cont_rec_neg_loglik(output, x):
    """
    Compute the negative log-likelihood of our data given the output parameters and the data
    """
    p_L, p_T, mu_R, log_sigma_R = output # Unpack the parameters of the distributions
    sigma_R = torch.exp(log_sigma_R) # Convert the log scale

    # Define the distributions
    dist_L = Bernoulli(p_L)
    dist_T = Bernoulli(p_T)
    dist_R = Normal(mu_R, sigma_R)

    # Estimate the log-likelihoods
    NLL = -torch.mean(dist_L.log_prob(x[:,0].view(-1,1)) + dist_T.log_prob(x[:,1].view(-1,1)) + dist_R.log_prob(x[:,2].view(-1,1)))

    return NLL

###############################################################################
###       Continuous size neural network and negative log-likelihood        ###
###############################################################################
class cont_size_ks_net(nn.Module):
    def __init__(self, N_HU, NLA):
        super().__init__()

        # Hidden layers
        self.hidden_L = nn.Linear(1, N_HU, bias=False) # Kidney stone is not related to anything, so receives constant as input of size 1
        self.hidden_L_2 = nn.Linear(N_HU, N_HU)#, bias=False) 
        
        self.hidden_T = nn.Linear(1, N_HU) # Treatment is affected by size of the stone so receives variable L of size 1 as input
        self.hidden_T_2 = nn.Linear(N_HU, N_HU) 
        
        self.hidden_R = nn.Linear(2, N_HU) # Recovery is affected both by size and treatment, so receives two variables of size 1 as input
        self.hidden_R_2 = nn.Linear(N_HU, N_HU)

        # Output layers: all the variables in this case have only one parameter p as output
        self.out_L_a = nn.Linear(N_HU, 1)#, bias=False)
        self.out_L_b = nn.Linear(N_HU, 1)#, bias=False)
        self.out_T   = nn.Linear(N_HU, 1)
        self.out_R_a = nn.Linear(N_HU, 1)
        self.out_R_b = nn.Linear(N_HU, 1)

        # Activation functions
        self.nla = NLA

    def forward(self, x):
        const = torch.ones_like(x[:,0]) # Constant for the exogenous variables

        # We have to use the following "view" because of the input shape
        h_L = self.nla(self.hidden_L(const.view(-1,1)))
        h_T = self.nla(self.hidden_T(x[:,0].view(-1,1)))
        h_R = self.nla(self.hidden_R(x[:,[0,1]].view(-1,2)))

        h_L = self.nla(self.hidden_L_2(h_L))
        h_T = self.nla(self.hidden_T_2(h_T))
        h_R = self.nla(self.hidden_R_2(h_R))

        a_L = self.out_L_a(h_L) # a and b are both real valued parameters, they can be parametrized to be strictly positive
        b_L = self.out_L_b(h_L)
        p_T = torch.sigmoid(self.out_T(h_T))
        a_R = self.out_R_a(h_R)
        b_R = self.out_R_b(h_R)

        return a_L, b_L, p_T, a_R, b_R

### If the size is parametrized as a log-normal
def cont_size_neg_loglik(output, x):
    """
    Compute the negative log-likelihood of our data given the output parameters and the data
    """
    mu_L, log_sigma_L, p_T, mu_R, log_sigma_R = output # Unpack the parameters of the distributions

    # Convert the log variables into positive values
    sigma_R = torch.exp(log_sigma_R)
    sigma_L = torch.exp(log_sigma_L)

    # Define the distributions
    dist_L = LogNormal(mu_L, sigma_L)
    dist_T = Bernoulli(p_T)
    dist_R = Normal(mu_R, sigma_R)

    # Estimate the log-likelihoods
    NLL = -torch.mean(dist_L.log_prob(x[:,0].view(-1,1)) + dist_T.log_prob(x[:,1].view(-1,1)) + dist_R.log_prob(x[:,2].view(-1,1)))

    return NLL
