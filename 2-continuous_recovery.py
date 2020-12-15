import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

# Internal packages
from data_loader import KidneyStoneDataset, ToTensor
from model import cont_rec_ks_net, binary_neg_loglik, cont_rec_neg_loglik, ContinuousOutcome
from train import train

# Hyperparameters
BATCH_SIZE = 128
EPOCHS     = 150
LEARN_R    = 1e-2
N_HU       = 4
#NLA = F.relu
NLA = nn.LeakyReLU(1) # For a linear neural network

# Initialize the dataset
data = KidneyStoneDataset("./data/ks_cont_rec_data.npy", transform=ToTensor())
train_loader = DataLoader(data, batch_size=BATCH_SIZE)

# Initialize the model
model = ContinuousOutcome([N_HU], NLA)

# Optimizers
#optimizer = optim.SGD(model.parameters(), lr=LEARN_R, weight_decay=0.1)
optimizer = optim.RMSprop(model.parameters(), lr=LEARN_R)

cum_loss = train(model, optimizer, cont_rec_neg_loglik, train_loader, EPOCHS)

# We want to query an intervention on the treatment. In order to do that, we estimate p(R=1|L=1, T=do(A))*P(L=1)
l_1_t_1_r_1 = torch.tensor([1., 1., 1.])
l_0_t_1_r_1 = torch.tensor([0., 1., 1.])

# We want to query an intervention on the treatment. In order to do that, we estimate p(R=1|L=1, T=do(B))*P(L=1)
l_1_t_0_r_1 = torch.tensor([1., 0., 1.])
l_0_t_0_r_1 = torch.tensor([0., 0., 1.])

# Probabilities with Treatment A
p_ks, p_t1_l1, mu_r1_t1_l1, log_sigma_r1_t1_l1 = model(l_1_t_1_r_1.unsqueeze(0)) # Probabilities with KS = L
_, p_t1_l0, mu_r1_t1_l0, log_sigma_r1_t1_l0 = model(l_0_t_1_r_1.unsqueeze(0))    # Probabilities with KS = S

# Probabilities with Treatment B
_, p_t0_l1, mu_r1_t0_l1, log_sigma_r1_t0_l1 = model(l_1_t_0_r_1.unsqueeze(0))  # Probabilities with KS = L
_, p_t0_l0, mu_r1_t0_l0, log_sigma_r1_t0_l0 = model(l_0_t_0_r_1.unsqueeze(0))  # Probabilities with KS = S


print(mu_r1_t0_l0.item(), mu_r1_t1_l0.item()) # The first should be around 1, the second around 4
# 0.873916745185852 4.568049430847168

int_t1 = mu_r1_t1_l1*p_ks + mu_r1_t1_l0*(1-p_ks)
int_t0 = mu_r1_t0_l1*p_ks + mu_r1_t0_l0*(1-p_ks)

c_effect = int_t1 - int_t0
print(c_effect.item())
# 3.694132089614868
