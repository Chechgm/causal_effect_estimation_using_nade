#! ./model.py
"""Defines the models and their loss functions.

Classes available:
- BernoulliMLP
- LocationScaleMLP
- Binary
- ContinuousOutcome
- ContinuousConfounderAndOutcome
- FrontDoor

The available functions are:
- binary_loss
- continuous_outcome_loss
- continuous_confounder_outcome_loss
- front_door_loss
"""
import torch
from torch import nn
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.log_normal import LogNormal
from torch.distributions.normal import Normal


class BernoulliMLP(nn.Module):
    def __init__(self, ARCH, NLA, root=False):
        super().__init__()

        if root:
            self.layers = [nn.Linear(ARCH[i], ARCH[i+1], bias=False) for i in range(len(ARCH)-1)]
            self.out = nn.Linear(ARCH[-1], 1, bias=False)
        else:
            self.layers = [nn.Linear(ARCH[i], ARCH[i+1]) for i in range(len(ARCH)-1)]
            self.out = nn.Linear(ARCH[-1], 1)

        self.intermediate = nn.ModuleList(self.layers)

        # Activation functions
        self.nla = NLA

    def forward(self, x):
        for _, layer in enumerate(self.intermediate):
            x = self.nla(layer(x))
        p = torch.sigmoid(self.out(x))

        return p


class LocationScaleMLP(nn.Module):
    def __init__(self, ARCH, NLA, root=False):
        super().__init__()

        if root:
            self.layers = [nn.Linear(ARCH[i], ARCH[i+1], bias=False) for i in range(len(ARCH)-1)]
            self.out_location = nn.Linear(ARCH[-1], 1, bias=False)
            self.out_scale = nn.Linear(ARCH[-1], 1, bias=False)
        else:
            self.layers = [nn.Linear(ARCH[i], ARCH[i+1]) for i in range(len(ARCH)-1)]
            self.out_location = nn.Linear(ARCH[-1], 1)
            self.out_scale = nn.Linear(ARCH[-1], 1)

        self.intermediate = nn.ModuleList(self.layers)

        # Activation functions
        self.nla = NLA

    def forward(self, x):
        for _, layer in enumerate(self.intermediate):
            x = self.nla(layer(x))
        location = self.out_location(x)
        scale = self.out_scale(x)

        return location, scale


###############################################################################
#       Binary kidney stones neural network and negative log-likelihood       #
###############################################################################
class Binary(nn.Module):
    def __init__(self, ARCH, NLA):
        super().__init__()

        ARCH = ARCH + [1]

        # Independent Causal Mechanisms
        self.ks_mlp = BernoulliMLP([1]+ARCH, NLA, root=True)
        self.t_mlp = BernoulliMLP([1]+ARCH, NLA)
        self.r_mlp = BernoulliMLP([2]+ARCH, NLA)

    def forward(self, x):
        const = torch.ones_like(x[:,0])  # Constant for the root variables

        # We have to use the following "view" because of the input shape
        l_p = self.ks_mlp(const.view(-1,1))
        t_p = self.t_mlp(x[:,0].view(-1,1))
        r_p = self.r_mlp(x[:,[0,1]].view(-1,2))

        return l_p, t_p, r_p


def binary_loss(output, x):
    """ Compute the negative log-likelihood of output parameters given the data.
    """
    p_L, p_T, p_R = output  # Unpack the parameters of the distributions

    # Define the distributions
    dist_L = Bernoulli(p_L)
    dist_T = Bernoulli(p_T)
    dist_R = Bernoulli(p_R)

    # Estimate the log-likelihoods
    NLL = -torch.mean(dist_L.log_prob(x[:,0].view(-1,1)) +
                      dist_T.log_prob(x[:,1].view(-1,1)) +
                      dist_R.log_prob(x[:,2].view(-1,1)))

    return NLL


###############################################################################
#      Continuous recovery neural network and negative log-likelihood         #
###############################################################################
class ContinuousOutcome(nn.Module):
    def __init__(self, ARCH, NLA):
        super().__init__()

        ARCH = ARCH + [1]

        # Independent Causal Mechanisms
        self.ks_mlp = BernoulliMLP([1]+ARCH, NLA, root=True)
        self.t_mlp = BernoulliMLP([1]+ARCH, NLA)
        self.r_mlp = LocationScaleMLP([2]+ARCH, NLA)

    def forward(self, x):
        const = torch.ones_like(x[:,0])  # Constant for the root variables

        # We have to use the following "view" because of the input shape
        l_p = self.ks_mlp(const.view(-1,1))
        t_p = self.t_mlp(x[:,0].view(-1,1))
        r_l, r_s = self.r_mlp(x[:,[0,1]].view(-1,2))

        return l_p, t_p, r_l, r_s


def continuous_outcome_loss(output, x):
    """ Compute the negative log-likelihood of output parameters given the data.
    """
    p_L, p_T, mu_R, log_sigma_R = output  # Unpack the parameters of the distributions
    sigma_R = torch.exp(log_sigma_R)  # Convert the log scale

    # Define the distributions
    dist_L = Bernoulli(p_L)
    dist_T = Bernoulli(p_T)
    dist_R = Normal(mu_R, sigma_R)

    # Estimate the log-likelihoods
    NLL = -torch.mean(dist_L.log_prob(x[:,0].view(-1,1)) +
                      dist_T.log_prob(x[:,1].view(-1,1)) +
                      dist_R.log_prob(x[:,2].view(-1,1)))

    return NLL


###############################################################################
#         Continuous size neural network and negative log-likelihood          #
###############################################################################
class ContinuousConfounderAndOutcome(nn.Module):
    def __init__(self, ARCH, NLA):
        super().__init__()

        ARCH = ARCH + [1]

        # Independent Causal Mechanisms
        self.ks_mlp = LocationScaleMLP([1]+ARCH, NLA, root=True)
        self.t_mlp = BernoulliMLP([1]+ARCH, NLA)
        self.r_mlp = LocationScaleMLP([2]+ARCH, NLA)

    def forward(self, x):
        const = torch.ones_like(x[:,0])  # Constant for the root variables

        # We have to use the following "view" because of the input shape
        c_l, c_s = self.ks_mlp(const.view(-1,1))
        t_p = self.t_mlp(x[:,0].view(-1,1))
        o_l, o_s = self.r_mlp(x[:,[0,1]].view(-1,2))

        return c_l, c_s, t_p, o_l, o_s


# If the size is parametrized as a log-normal
def continuous_confounder_outcome_loss(output, x):
    """ Compute the negative log-likelihood of output parameters given the data.
    """
    mu_L, log_sigma_L, p_T, mu_R, log_sigma_R = output  # Unpack the parameters of the distributions

    # Convert the log variables into positive values
    sigma_R = torch.exp(log_sigma_R)
    sigma_L = torch.exp(log_sigma_L)

    # Define the distributions
    dist_L = LogNormal(mu_L, sigma_L)
    dist_T = Bernoulli(p_T)
    dist_R = Normal(mu_R, sigma_R)

    # Estimate the log-likelihoods
    NLL = -torch.mean(dist_L.log_prob(x[:,0].view(-1,1)) +
                      dist_T.log_prob(x[:,1].view(-1,1)) +
                      dist_R.log_prob(x[:,2].view(-1,1)))

    return NLL


###############################################################################
#                          Front-door neural network                          #
###############################################################################
class FrontDoor(nn.Module):
    def __init__(self, ARCH, NLA):
        super().__init__()

        ARCH = ARCH + [1]

        # Independent Causal Mechanisms
        self.x_mlp = LocationScaleMLP([1]+ARCH, NLA, root=True)
        self.z_mlp = BernoulliMLP([1]+ARCH, NLA)
        self.y_mlp = LocationScaleMLP([1]+ARCH, NLA)

        # Auxiliary
        self.y_aux_mlp = LocationScaleMLP([2]+ARCH, NLA)

    def forward(self, x):
        const = torch.ones_like(x[:,0])  # Constant for the root variables

        loc_X, scale_X = self.x_mlp(const.view(-1,1))
        loc_Z, scale_Z = self.z_mlp(x[:,0].view(-1,1))
        loc_Y, scale_Y = self.y_mlp(x[:,1].view(-1,1))
        # Auxiliary networks
        # X and Z to Y
        loc_aux_Y, scale_aux_Y = self.y_aux_mlp(x[:,[0,1]].view(-1,2))

        return loc_X, scale_X, loc_Z, scale_Z, loc_Y, scale_aux_Y, loc_Y, scale_aux_Y


# If the size is parametrized as a log-normal
def front_door_loss(output, x):
    """ Compute the negative log-likelihood of output parameters given the data.
    """
    # Unpack the parameters of the distributions
    mu_X, log_sigma_X, \
        mu_Z, log_sigma_Z, \
        mu_Y, log_sigma_Y, \
        mu_ZX_Y, log_sigma_ZX_Y = output

    # Convert the log variables into positive values
    sigma_X = torch.exp(log_sigma_X)
    sigma_Z = torch.exp(log_sigma_Z)
    sigma_Y = torch.exp(log_sigma_Y)

    sigma_ZX_Y = torch.exp(log_sigma_ZX_Y)

    # Define the forward distributions
    dist_X = Normal(mu_X, sigma_X)
    dist_Z = Normal(mu_Z, sigma_Z)
    dist_Y = Normal(mu_Y, sigma_Y)

    # Auxiliary distribution
    # The normal for Y|X,Z
    dist_ZX_Y = Normal(mu_ZX_Y, sigma_ZX_Y)

    # Estimate the log-likelihoods
    NLL = -torch.mean(dist_X.log_prob(x[:,0].view(-1,1)) +
                      dist_Z.log_prob(x[:,1].view(-1,1)) +
                      dist_Y.log_prob(x[:,2].view(-1,1))) - \
        torch.mean(dist_ZX_Y.log_prob(x[:,2].view(-1,1)))

    return NLL
