import numpy as np
import scipy

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

from torch.distributions.bernoulli import Bernoulli 
from torch.distributions.categorical import Categorical 
from torch.distributions.log_normal import LogNormal
from torch.distributions.normal import Normal

# Graphs
import matplotlib.pyplot as plt 
import seaborn as sns

# Internal packages
from data_loader import KidneyStoneDataset, ToTensor
from model import front_door_net, front_door_neg_loglik
from train import train

with open("./experiments/default_params.yaml", 'r') as f:
    params = yaml.load(f, Loader=yaml.FullLoader)

NLA = nn.LeakyReLU(1) # For a linear neural network

sd_idx = [0, 1, 2, 3] # Standarize the two continuous variables

# Initialize the dataset
data = KidneyStoneDataset("./data/front_door_data.npy", transform=ToTensor(), idx_sd=sd_idx)
#data = KidneyStoneDataset("./data/ks_non_linear_data_lp.npy", transform=ToTensor(), idx_mean=mean_idx, idx_sd=sd_idx)
train_loader = DataLoader(data, batch_size=params["batch_size"])

# Initialize the model
model = front_door_net(N_HU, NLA)

# Optimizers
#optimizer = optim.SGD(model.parameters(), lr=LEARN_R)
optimizer = optim.RMSprop(model.parameters(), lr=params["learn_rate"])

#
cum_loss = train(model, optimizer, front_door_neg_loglik, train_loader, params)

#
# First, we get the parameters of the size variable:
n_samples = 50
arbitrary_query = torch.tensor([1., 1., 1., 1.]) # It is only important that the first element is 1

mu_X, log_sigma_X, _, _, _, _, _, _ = model(arbitrary_query.unsqueeze(0))
sigma_X = torch.exp(log_sigma_X)

X_dist = Normal(mu_X, sigma_X)
X_samples = X_dist.sample((n_samples,)).view(n_samples,1)

#
_, _, mu_Z, log_sigma_Z, _, _, _, _ = model(torch.tensor(data.ks_dataset).float())

# Samples from Z given X
sigma_Z = torch.exp(log_sigma_Z)
dist_Z  = Normal(mu_Z, sigma_Z)
samples_Z = dist_Z.sample()

# Front door adjustment with x==0, we have to code for x==0.5
# Front door
# sum(z) P(Z=z | X=int_x) sum(x') P(Y|X=x', Z=z)P(X=x')


def front_door_adjustment()
    """ Estimates an interventional distribution using the front-door adjustment.
    """
    n_samples = 5000

    intervention_query = torch.tensor([0., 1., 1., 1.])/data.sd[0] # x == 0

    # P(Z=z | X=int_x)
    mu_X, log_sigma_X, mu_Z, log_sigma_Z, _, _, _, _ = model(intervention_query.unsqueeze(0))

    sigma_X = torch.exp(log_sigma_X)
    sigma_Z = torch.exp(log_sigma_Z)

    Z_dist = Normal(mu_Z, sigma_Z)
    Z_samples = Z_dist.sample((n_samples,)).view(n_samples,1)

    # P(X=x')
    X_dist = Normal(mu_X, sigma_X)
    X_samples = X_dist.sample((n_samples,)).view(n_samples,1)

    # P(Y|X=x', Z=z)
    # Pack the samples from the distributions:
    ZX_samples = torch.cat([X_samples, Z_samples, torch.ones([n_samples, 2])], dim=1)
    _, _, _, _, _, _, mu_ZX_Y_0, _ = model(ZX_samples)
    ######## PACK IN FUNCTION ########


def conditional_estimation():
    """ Conditional effect estimation for comparison purposes.
    """
    ### Condtional effect:
    # Samples from X
    idx_z = np.random.choice(data.ks_dataset.shape[0], size=n_samples)
    Z_samples = torch.tensor(data.ks_dataset[idx_z, 1]).float()
    n_z = Z_samples.shape[0]

    # P(Y|X=x', Z=z)
    # Do(X=0)
    ZX_samples = torch.cat([torch.zeros([n_z, 1]), Z_samples.view(-1,1), torch.ones([n_z, 2])], dim=1)
    _, _, _, _, _, _, mu_ZX_Y_0, _ = model(ZX_samples)

    # Do(X=0.5)
    ZX_samples = torch.cat([torch.ones([n_z, 1])*0.5, Z_samples.view(-1,1), torch.ones([n_z, 2])], dim=1)
    _, _, _, _, _, _, mu_ZX_Y_05, _ = model(ZX_samples)


# you can get these from NB 6, restore data from the analytical conditional
%store -r test_1_0
%store -r test_1_05

# Causal effect
print(torch.mean(mu_ZX_Y_05*data.sd[2])-torch.mean(mu_ZX_Y_0*data.sd[2]))
print(np.mean(test_1_05)-np.mean(test_1_0))

def plot_front_door():
    """ Utility to plot the front-door adjustment data.
    """
    # Plot of true and neural
    #ax = sns.distplot((mu_ZX_Y_0*data.sd[2]).detach().numpy(), label="Neural $do(X=0)$")
    ax = sns.distplot((mu_ZX_Y_05*data.sd[2]).detach().numpy(), label="Neural $do(X=0.5)$")
    #ax = sns.distplot(test_1_0, label="True $do(X=0)$")
    ax = sns.distplot(test_1_05, label="True $do(X=0.5)$")

    #plt.title("True vs. Neural $do(X=0)$", y=1.10)
    plt.title("True vs. Neural $do(X=0.5)$", y=1.10)
    ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.10), borderaxespad=0, frameon=False)

    #ax.text(0.5, 1,"WD: %.2f" % (scipy.stats.wasserstein_distance(test_1_0, (mu_ZX_Y_0*data.sd[2]).detach().numpy()[:,0])), fontsize=11)
    ax.text(0.5, 1,"WD: %.2f" % (scipy.stats.wasserstein_distance(test_1_05, (mu_ZX_Y_05*data.sd[2]).detach().numpy()[:,0])), fontsize=11)

    ax.set_xlim(0,10)
    ax.set_ylim(0,1.2)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.savefig("./results/neural_05.pdf", ppi=300, bbox_inches='tight');
