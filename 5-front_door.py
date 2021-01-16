import numpy as np
import yaml

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

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

# Causal effect
interventional_dist_05 = front_door_adjustment(model, 0.5, data)
interventional_dist_0 = front_door_adjustment(model, 0, data)
print("treatment effect: ", np.mean(interventional_dist_05)-np.mean(interventional_dist_0))
print("random conditional distribution: ", conditional_estimate(model, 0.5, data))
#print(np.mean(test_1_05)-np.mean(test_1_0))