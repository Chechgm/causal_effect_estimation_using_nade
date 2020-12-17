import yaml

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

from torch.distributions.log_normal import LogNormal

# Internal packages
from data_loader import KidneyStoneDataset, ToTensor
from model import ContinuousConfounderAndOutcome, cont_size_neg_loglik
from train import train

yaml_dir = "./experiments/default_params.yaml"
with open(yaml_dir, 'r') as f:
    params = yaml.load(f, Loader=yaml.FullLoader)

NLU = torch.tanh

# Gama generative process
# Initialize the dataset
data = KidneyStoneDataset("./data/ks_cont_size_data_g.npy", transform=ToTensor())
train_loader = DataLoader(data, batch_size=params["batch_size"])

# Initialize the model
model = ContinuousConfounderAndOutcome(params["architecture"], NLA)

# Optimizers
#optimizer = optim.SGD(model.parameters(), lr=LEARN_R, weight_decay=0.002)
#optimizer = optim.RMSprop(model.parameters(), lr=LEARN_R, weight_decay=0.002)
optimizer = optim.RMSprop(model.parameters(), lr=params["learn_rate"])

cum_loss = train(model, optimizer, cont_size_neg_loglik, train_loader, params)

# Causal effect estimation
# First, we get the parameters of the size variable:
n_samples = 50
arbitrary_query = torch.tensor([1., 1., 1.])  # It is only important that the first element is 1
mu_L, log_sigma_L, _, _, _ = model(arbitrary_query.unsqueeze(0))
sigma_L = torch.exp(log_sigma_L)
L_dist = LogNormal(mu_L, sigma_L)
L_samples = L_dist.sample((n_samples,)).view(n_samples,1)

t_1 = torch.cat((L_samples, torch.ones(n_samples, 2)), 1)
t_0 = torch.cat((L_samples, torch.zeros(n_samples, 1), torch.ones(n_samples, 1)), 1)

# Probabilities with Treatment A
_, _, _, mu_t1, _ = model(t_1)  # Probabilities with KS = L
_, _, _, mu_t0, _ = model(t_0)  # Probabilities with KS = L

# By monte carlo integration, we are getting mu from the value of the variables from the distribution
for n in [1, 5, 25, 50]:
    print("%d: %f" % (n, torch.mean(mu_t1[:n])-torch.mean(mu_t0[:n])))


# Log normal generative model
data = KidneyStoneDataset("./data/ks_cont_size_data_ln.npy", transform=ToTensor())
train_loader = DataLoader(data, batch_size=params["batch_size"])

# Initialize the model
model = ContinuousConfounderAndOutcome(params["architecture"], NLA)

# Optimizers
#optimizer = optim.SGD(model.parameters(), lr=LEARN_R, weight_decay=0.1)
optimizer = optim.RMSprop(model.parameters(), lr=params["learn_rate"])

cum_loss = train(model, optimizer, cont_size_neg_loglik, train_loader, params)

# First, we get the parameters of the size variable:
n_samples = 50
arbitrary_query = torch.tensor([1., 1., 1.])  # It is only important that the first element is 1
mu_L, log_sigma_L, _, _, _ = model(arbitrary_query.unsqueeze(0))
sigma_L = torch.exp(log_sigma_L)
L_dist = LogNormal(mu_L, sigma_L)
L_samples = L_dist.sample((n_samples,)).view(n_samples,1)

t_1 = torch.cat((L_samples, torch.ones(n_samples, 2)), 1)
t_0 = torch.cat((L_samples, torch.zeros(n_samples, 1) ,torch.ones(n_samples, 1)), 1)

# Probabilities with Treatment A
_, _, _, mu_t1, _ = model(t_1)  # Probabilities with KS = L
_, _, _, mu_t0, _ = model(t_0)  # Probabilities with KS = L

# By monte carlo integration, we are getting mu from the value of the variables from the distribution
for n in [1, 5, 25, 50]:
    print("%d: %f" % (n, torch.mean(mu_t1[:n])-torch.mean(mu_t0[:n])))
