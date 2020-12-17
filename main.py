#! main.py
""" Main file

TODO: Consider the possibility of having a parameters class.

Available functions:
- load_and_intialize
- causal_estimates
- main
"""
import argparse
import csv
import os
import yaml

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

from causal_estimates import binary_backdoor_adjustment, continuous_outcome_backdoor_adjustment
from data_loader import KidneyStoneDataset, ToTensor
from model import Binary, ContinuousOutcome, ContinuousConfounderAndOutcome, \
                    FrontDoor, binary_loss, continuous_outcome_loss, \
                    continuous_confounder_outcome_loss, front_door_loss
from train import train


def load_and_intialize(params):
    """ Loads the right dataset, intializes the right NN and its respective loss
    """
    # Choose the right activation function
    if params["activation"] == "linear":
        NLU = nn.LeakyReLU(1)
    elif params["activation"] == "relu":
        NLU = F.relu
    elif params["activation"] == "tanh":
        NLU = torch.tanh

    # Load the data, intialize the NN and choose the loss depending on the experiment
    if params["model"] == "binary":
        data = KidneyStoneDataset("./data/binary_data.npy", transform=ToTensor())
        model = Binary(params["architecture"], NLU).cuda() if params["cuda"] else Binary(params["architecture"], NLU)
        loss_fn = binary_loss
    elif params["model"] == "continuous_outcome":
        data = KidneyStoneDataset("./data/continuous_outcome_data.npy", transform=ToTensor())
        model = ContinuousOutcome(params["architecture"], NLU).cuda() if params["cuda"] else ContinuousOutcome(params["architecture"], NLU)
        loss_fn = continuous_outcome_loss
    elif params["model"] == "continuous_confounder_gamma":
        data = KidneyStoneDataset("./data/continuous_confounder_gamma_data.npy", transform=ToTensor())
        model = ContinuousConfounderAndOutcome(params["architecture"], NLU).cuda() if params["cuda"] else ContinuousConfounderAndOutcome(params["architecture"], NLU)
        loss_fn = continuous_confounder_outcome_loss
    elif params["model"] == "continuous_confounder_logn":
        data = KidneyStoneDataset("./data/continuous_confounder_logn_data.npy", transform=ToTensor())
        model = ContinuousConfounderAndOutcome(params["architecture"], NLU).cuda() if params["cuda"] else ContinuousConfounderAndOutcome(params["architecture"], NLU)
        loss_fn = continuous_confounder_outcome_loss
    elif params["model"] == "non_linear":
        data = KidneyStoneDataset("./data/non_linear_data.npy", transform=ToTensor())
        model = ContinuousConfounderAndOutcome(params["architecture"], NLU).cuda() if params["cuda"] else ContinuousConfounderAndOutcome(params["architecture"], NLU)
        loss_fn = continuous_confounder_outcome_loss
    elif params["model"] == "front_door":
        data = KidneyStoneDataset("./data/front_door_data.npy", transform=ToTensor())
        model = FrontDoor(params["architecture"], NLU).cuda() if params["cuda"] else FrontDoor(params["architecture"], NLU)
        loss_fn = front_door_loss

    return data, model, loss_fn


def causal_effect_estimation(model, params):
    """ Chooses the right causal estimate depending on the experiment
    """

    if params["model"] == "binary":
        interventional_dist_1 = binary_backdoor_adjustment(model.r_mlp, 1, model.ks_mlp, [0., 1.])
        interventional_dist_0 = binary_backdoor_adjustment(model.r_mlp, 0, model.ks_mlp, [0., 1.])
        causal_effect = interventional_dist_1 - interventional_dist_0
    elif params["model"] == "continuous_outcome":
        interventional_dist_1 = continuous_outcome_backdoor_adjustment(model.r_mlp, 1, model.ks_mlp, [0., 1.])
        interventional_dist_0 = continuous_outcome_backdoor_adjustment(model.r_mlp, 0, model.ks_mlp, [0., 1.])
        causal_effect = interventional_dist_1 - interventional_dist_0
    elif params["model"] == "continuous_confounder_gamma":
        causal_effect = "Not implemented"
    elif params["model"] == "continuous_confounder_logn":
        causal_effect = "Not implemented"
    elif params["model"] == "nono_linear":
        causal_effect = "Not implemented"
    elif params["model"] == "front_door":
        causal_effect = "Not implemented"

    return causal_effect


def main(params):
    """ Main function for the experiments of "Causal effect estimation using neural autoregressive density estimators"
    """
    # Initalise the results dictionary
    results = {}

    data, model, loss_fn = load_and_intialize(params)

    train_loader = DataLoader(data, batch_size=params["batch_size"])

    # Optimizers
    if params["optimizer"] == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=params["learn_rate"])
    elif params["optimizer"] == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), lr=params["learn_rate"])

    # Train the NN
    cum_loss = train(model, optimizer, loss_fn, train_loader, params)

    # Evaluate
    results["final_loss"] = cum_loss[-1]
    results["causal_effect"] = causal_effect_estimation(model, params)

    # Save the results
    save_dict = {**params, **results}
    # Write the results and architecture somewhere
    if os.path.exists('./results/results.csv'):
        with open('./results/results.csv', 'a') as f:
            fieldnames = save_dict.keys()
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(save_dict)
    else:
        with open('./results/results.csv', 'w') as f:
            fieldnames = save_dict.keys()
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(save_dict)


if __name__ == '__main__':

    # Load the parameters from json file
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='./data/',
                        help="Directory containing the dataset")
    parser.add_argument('--yaml_dir', default='./experiments/default_params.yaml',
                        help="Directory containing default_params.json")
    args = parser.parse_args()

    assert os.path.isfile(
        args.yaml_dir), "No YAML configuration file found at {}".format(args.yaml_path)

    with open(args.yaml_dir, 'r') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    # use GPU if available
    params["cuda"] = torch.cuda.is_available()

    # Set the random seed for reproducible experiments
    torch.manual_seed(params["random_seed"])
    if params["cuda"]:
        torch.cuda.manual_seed(params["random_seed"])

    main(params)
