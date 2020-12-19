import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim

# Internal packages
from data_loader import KidneyStoneDataset, ToTensor
from model import ContinuousConfounderAndOutcome, cont_size_neg_loglik
from train import train

# For graphing purposes below
import matplotlib.pyplot as plt
import seaborn as sns

from torch.distributions.normal import Normal

import numpy as np

# Hyperparameters
BATCH_SIZE = 128
EPOCHS     = 150
LEARN_R    = 5e-4 #1e-3 #1e-2 # RMS
N_HU       = 8
#NLA = torch.tanh #torch.sigmoid #F.relu
NLA = nn.LeakyReLU(1) # For a linear neural network

mean_idx = [2]
sd_idx = [0, 2] # Standarize the two continuous variables

# Initialize the dataset
data = KidneyStoneDataset("./data/ks_non_linear_data_lp.npy", transform=ToTensor(), idx_mean=mean_idx, idx_sd=sd_idx)
#data = KidneyStoneDataset("./data/ks_non_linear_data_lp.npy", transform=ToTensor(), idx_mean=mean_idx, idx_sd=sd_idx)
train_loader = DataLoader(data, batch_size=BATCH_SIZE)

# Initialize the model
model = ContinuousConfounderAndOutcome([N_HU], NLA)

# Optimizers
#optimizer = optim.SGD(model.parameters(), lr=LEARN_R)
optimizer = optim.RMSprop(model.parameters(), lr=LEARN_R)

cum_loss = train(model, optimizer, cont_size_neg_loglik, train_loader, EPOCHS)

# Ancestral sampling
########################### KS SAMPLES ###########################
# First, we get the parameters of the size variable:
L_samples = torch.arange(5, 25, 0.1)/data.sd[0]
n = L_samples.shape[0]

########################### T SAMPLES ###########################
T1_samples = torch.ones(n, 1)
T0_samples = torch.zeros(n, 1)

########################### R SAMPLES ###########################
# T1
_, _, _, mu_R1, log_sigma_R1 = model(torch.cat((L_samples.view(-1,1), T1_samples, torch.ones(n, 1)), 1))

sigma_R1 = torch.exp(log_sigma_R1)

R1_dist = Normal(mu_R1, sigma_R1)
R1_samples = R1_dist.sample().view(n,1)

#T0
_, _, _, mu_R0, log_sigma_R0 = model(torch.cat((L_samples.view(-1,1), T0_samples, torch.ones(n, 1)), 1))

sigma_R0 = torch.exp(log_sigma_R0)

R0_dist = Normal(mu_R0, sigma_R0)
R0_samples = R0_dist.sample().view(n,1)

# First run NN and save as neural_TE, then run linear and save as TE
TE = ((mu_R1*data.sd[2]+data.mean[2])-(mu_R0*data.sd[2]+data.mean[2])).detach().numpy()


###### TODO: WRAP THIS AROUND A PLOTTING FUNCTION!
# In order to run this cell, one must save the results from the neural model before
ax = sns.lineplot(x=(L_samples*data.sd[0]).numpy(), y=TE[:,0], label="Linear TE")
ax = sns.lineplot(x=(L_samples*data.sd[0]).numpy(), y=neural_TE[:,0], label="Neural TE")
ax = sns.lineplot(x=(L_samples*data.sd[0]).numpy(), y=(50/(3+L_samples*data.sd[0])).numpy(), label="True TE")
ax = sns.distplot(data.ks_dataset[:,0], hist=True, kde=False, color='silver', hist_kws={'alpha': 0.8, 'weights':0.028*np.ones(len(data.ks_dataset))})
#ax = sns.distplot(data.ks_dataset[:,0], rug=True, hist=False, kde=False)

plt.title("Comparison between true and \n estimated conditional Treatment Effects", y=1.10)
ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.10), borderaxespad=0, frameon=False)

ax.set_xlim(4.1, 25.8)
ax.set_ylim(0, 6.5)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.savefig("./results/linear_vs_non-linear_hist.pdf", ppi=300, bbox_inches='tight');
