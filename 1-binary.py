# From the binary notebook
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim

from data_loader import KidneyStoneDataset, ToTensor
from model import Binary, binary_neg_loglik
from train import train

# Hyperparameters
BATCH_SIZE = 128
EPOCHS = 150
LEARN_R = 1e-2
ARCH = [4]
#NLA = F.relu
NLA = nn.LeakyReLU(1)  # For a linear neural network

# Initialize the dataset
data = KidneyStoneDataset("./data/ks_binary_data.npy", transform=ToTensor())
train_loader = DataLoader(data, batch_size=BATCH_SIZE)

# Initialize the model
model = Binary(ARCH, NLA)

# Optimizers
#optimizer = optim.SGD(model.parameters(), lr=LEARN_R, weight_decay=0.1)
optimizer = optim.RMSprop(model.parameters(), lr=LEARN_R)

cum_loss = train(model, optimizer, binary_neg_loglik, train_loader, EPOCHS)


# We want to query an intervention on the treatment. In order to do that, we estimate p(R=1|L=1, T=do(A))*P(L=1)
l_1_t_1_r_1 = torch.tensor([1., 1., 1.])
l_0_t_1_r_1 = torch.tensor([0., 1., 1.])

# We want to query an intervention on the treatment. In order to do that, we estimate p(R=1|L=1, T=do(B))*P(L=1)
l_1_t_0_r_1 = torch.tensor([1., 0., 1.])
l_0_t_0_r_1 = torch.tensor([0., 0., 1.])

# Probabilities with Treatment A
p_ks, p_t1_l1, p_r1_t1_l1 = model(l_1_t_1_r_1.unsqueeze(0))  # Probabilities with KS = L
_, p_t1_l0, p_r1_t1_l0 = model(l_0_t_1_r_1.unsqueeze(0))  # Probabilities with KS = S

# Probabilities with Treatment B
_, p_t0_l1, p_r1_t0_l1 = model(l_1_t_0_r_1.unsqueeze(0))  # Probabilities with KS = L
_, p_t0_l0, p_r1_t0_l0 = model(l_0_t_0_r_1.unsqueeze(0))  # Probabilities with KS = S

print("The estimated probability of a Large kidney stone is: %.4f\n\
The estimated probability of Recovery given large stones and treatment A is: %.4f\n\
The estimated probability of Recovery given large stones and treatment B is: %.4f\n\
The estimated probability of Recovery given small stones and treatment A is: %.4f\n\
The estimated probability of Recovery given large stones and treatment B is: %.4f\n"
      % (p_ks, p_r1_t1_l1, p_r1_t0_l1, p_r1_t1_l0, p_r1_t0_l0))

int_t1 = p_r1_t1_l1*p_ks + p_r1_t1_l0*(1-p_ks)
int_t0 = p_r1_t0_l1*p_ks + p_r1_t0_l0*(1-p_ks)

c_effect = int_t1 - int_t0
print(c_effect.item())  # Causal effect


def outcome_model(neural_net, outcome_value):
    """
    """
    return None


def backdoor_adjustment(outcome_model, adjustment_set_model, adjustment_set_values):
    """Estimates the backdoor adjustment

    TODO outcome_model can be a partial function of the adjustment set values
    """
    estimate = 0
    for value in adjustment_set_values:
        estimate += (outcome_model(value)*adjustment_set_model(value))

    return estimate
