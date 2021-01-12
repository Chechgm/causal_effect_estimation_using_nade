import numpy as np
import scipy
import yaml

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
from model import FrontDoor, front_door_loss
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
model = FrontDoor(params["architecture"], NLA)

# Optimizers
optimizer = optim.RMSprop(model.parameters(), lr=params["learn_rate"])

cum_loss = train(model, optimizer, front_door_loss, train_loader, params)


def front_door_adjustment(model, value_intervention, data):
    """ Estimates an interventional distribution using the front-door adjustment.

    sum(z) P(Z=z | X=int_x) sum(x') P(Y|X=x', Z=z)P(X=x')
    """
    n_samples = 5000

    # P(X=x')
    input_x_mlp = torch.tensor([1.]).view(-1,1)
    mu_x, log_sigma_x = model.x_mlp(input_x_mlp)
    sigma_x = torch.exp(log_sigma_x)

    x_dist = Normal(mu_x, sigma_x)
    x_samples = x_dist.sample((n_samples,)).view(n_samples,1)

    # P(Z=z | X=int_x)
    input_z_mlp = torch.tensor([value_intervention/data.sd[0]]).view(-1,1)
    mu_z, log_sigma_z = model.z_mlp(input_z_mlp)
    sigma_z = torch.exp(log_sigma_z)

    z_dist = Normal(mu_z, sigma_z)
    z_samples = z_dist.sample((n_samples,)).view(n_samples,1)

    # P(Y|X=x', Z=z)
    xz_samples = torch.cat([x_samples/data.sd[0], z_samples/data.sd[1]], dim=1)
    means_outcome, _ = model.y_aux_mlp(xz_samples)

    return np.squeeze((means_outcome*data.sd[2]).detach().numpy()).tolist()

# Causal effect
interventional_dist_05 = front_door_adjustment(model, 0.5, data)
interventional_dist_0 = front_door_adjustment(model, 0, data)
print(np.mean(interventional_dist_05)-np.mean(interventional_dist_0))
#print(np.mean(test_1_05)-np.mean(test_1_0))

def conditional_estimation(data):
    """ Conditional effect estimation for comparison purposes.
    """
    ### Condtional effect:
    # Samples from X
    idx_z = np.random.choice(data.ks_dataset.shape[0], size=n_samples)
    z_samples = torch.tensor(data.ks_dataset[idx_z, 1]).float()
    n_z = z_samples.shape[0]

    # P(Y|X=x', Z=z)
    # Do(X=0)
    xz_samples = torch.cat([torch.zeros([n_z, 1]), z_samples.view(-1,1), torch.ones([n_z, 2])], dim=1)
    _, _, _, _, _, _, mu_xz_y_0, _ = model(xz_samples)

    # Do(X=0.5)
    xz_samples = torch.cat([torch.ones([n_z, 1])*0.5, z_samples.view(-1,1), torch.ones([n_z, 2])], dim=1)
    _, _, _, _, _, _, mu_xz_y_05, _ = model(xz_samples)


# you can get these from NB 6, restore data from the analytical conditional
#%store -r test_1_0
#%store -r test_1_05

def plot_front_door(estimate, true_value, value_intervention, label="Neural"):
    """ Utility to plot the front-door adjustment data.
    """
    # Plot of true and estimate (Linear, Neural, Conditional)
    ax = sns.distplot(estimate, label=f"{label} $do(X={value_intervention})$")
    ax = sns.distplot(true_value, label="True $do(X={value_intervention})$")

    plt.title(f"True vs. {label} $do(X={value_intervention})$", y=1.10)
    ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.10), borderaxespad=0, frameon=False)

    ax.text(0.5, 1,"WD: %.2f" % (scipy.stats.wasserstein_distance(true_value, estimate)), fontsize=11)

    ax.set_xlim(0,10)
    ax.set_ylim(0,1.2)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.savefig(f"./results/{label.lower()}_{value_intervention}.pdf", ppi=300, bbox_inches='tight');
