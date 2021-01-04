# For graphing purposes below
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import yaml

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim

# Internal packages
from causal_estimates import continuous_confounder_and_outcome_backdoor_adjustment_linspace
from data_loader import KidneyStoneDataset, ToTensor
from model import ContinuousConfounderAndOutcome, continuous_confounder_outcome_loss
from plot_utils import plot_non_linear
from train import train

mean_idx = [2]
sd_idx = [0, 2]  # Standarize the two continuous variables

with open("./experiments/default_params.yaml", 'r') as f:
    params = yaml.load(f, Loader=yaml.FullLoader)

# Initialize the dataset
data = KidneyStoneDataset("./data/non_linear_data.npy", transform=ToTensor(), idx_mean=mean_idx, idx_sd=sd_idx)
train_loader = DataLoader(data, batch_size=params["batch_size"])
NLA = torch.tanh

# Initialize the model
linear_model = ContinuousConfounderAndOutcome(params["architecture"], nn.LeakyReLU(1))
linear_optimizer = optim.RMSprop(linear_model.parameters(), lr=params["learn_rate"])
linear_loss = train(linear_model, linear_optimizer, continuous_confounder_outcome_loss, train_loader, params)

neural_model = ContinuousConfounderAndOutcome(params["architecture"], torch.tanh)
neural_optimizer = optim.RMSprop(neural_model.parameters(), lr=params["learn_rate"])
neural_loss = train(neural_model, neural_optimizer, continuous_confounder_outcome_loss, train_loader, params)

# Estimate the causal effects
neural_interventional_dist_1 = continuous_confounder_and_outcome_backdoor_adjustment_linspace(neural_model.r_mlp, 1., data)
neural_interventional_dist_0 = continuous_confounder_and_outcome_backdoor_adjustment_linspace(neural_model.r_mlp, 0., data)

linear_interventional_dist_1 = continuous_confounder_and_outcome_backdoor_adjustment_linspace(linear_model.r_mlp, 1., data)
linear_interventional_dist_0 = continuous_confounder_and_outcome_backdoor_adjustment_linspace(linear_model.r_mlp, 0., data)

neural_causal_effect = [int_1-int_0 for int_1, int_0 in zip(neural_interventional_dist_1, neural_interventional_dist_0)]
linear_causal_effect = [int_1-int_0 for int_1, int_0 in zip(linear_interventional_dist_1, linear_interventional_dist_0)]

plot_non_linear(linear_causal_effect, neural_causal_effect, data)
