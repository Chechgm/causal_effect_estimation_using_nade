#! main.py
""" Main file

TODO: Consider the possibility of having a parameters class.
TODO: Make a function for the logger.
TODO: Modify plot_wrapper for its intended purpose

Available functions:
- load_and_intialize
- causal_estimates
- main
"""
import argparse
import csv
import logging
import os
import sys
import yaml

import torch
from torch import nn
from torch.distributions.log_normal import LogNormal
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

from causal_estimates import binary_backdoor_adjustment, \
                                continuous_outcome_backdoor_adjustment, \
                                continuous_confounder_and_outcome_backdoor_adjustment, \
                                continuous_confounder_and_outcome_backdoor_adjustment_linspace
from data_loader import KidneyStoneDataset, ToTensor
from model import Binary, ContinuousOutcome, ContinuousConfounderAndOutcome, \
                    FrontDoor, binary_loss, continuous_outcome_loss, \
                    continuous_confounder_outcome_loss, front_door_loss
from train import train

# Logger set-up
logging.basicConfig(filename='./results/training_logger.log',
                    format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)

output_file_handler = logging.FileHandler('./results/training_logger.log')
stdout_handler = logging.StreamHandler(sys.stdout)

logger.addHandler(output_file_handler)
logger.addHandler(stdout_handler)


def load_and_intialize(params):
    """ Loads the right dataset, intializes the right NN and its respective loss
    """
    # Choose the right activation function
    if params["activation"] == "linear":
        NLA = nn.LeakyReLU(1)
    elif params["activation"] == "relu":
        NLA = F.relu
    elif params["activation"] == "tanh":
        NLA = torch.tanh

    # Load the data, intialize the NN and choose the loss depending on the experiment
    if params["model"] == "binary":
        data = KidneyStoneDataset("./data/binary_data.npy", transform=ToTensor())
        model = Binary(params["architecture"], NLA).cuda() if params["cuda"] else Binary(params["architecture"], NLA)
        loss_fn = binary_loss
    elif params["model"] == "continuous_outcome":
        data = KidneyStoneDataset("./data/continuous_outcome_data.npy", transform=ToTensor(), idx_mean=[2], idx_sd=[2])
        model = ContinuousOutcome(params["architecture"], NLA).cuda() if params["cuda"] else ContinuousOutcome(params["architecture"], NLA)
        loss_fn = continuous_outcome_loss
    elif params["model"] == "continuous_confounder_gamma":
        data = KidneyStoneDataset("./data/continuous_confounder_gamma_data.npy", transform=ToTensor(), idx_mean=[2], idx_sd=[0,2])
        model = ContinuousConfounderAndOutcome(params["architecture"], NLA).cuda() if params["cuda"] else ContinuousConfounderAndOutcome(params["architecture"], NLA)
        loss_fn = continuous_confounder_outcome_loss
    elif params["model"] == "continuous_confounder_logn":
        data = KidneyStoneDataset("./data/continuous_confounder_logn_data.npy", transform=ToTensor(), idx_mean=[2], idx_sd=[0,2])
        model = ContinuousConfounderAndOutcome(params["architecture"], NLA).cuda() if params["cuda"] else ContinuousConfounderAndOutcome(params["architecture"], NLA)
        loss_fn = continuous_confounder_outcome_loss
    elif params["model"] == "non_linear":
        data = KidneyStoneDataset("./data/non_linear_data.npy", transform=ToTensor(), idx_mean=[2], idx_sd=[0,2])
        model = ContinuousConfounderAndOutcome(params["architecture"], NLA).cuda() if params["cuda"] else ContinuousConfounderAndOutcome(params["architecture"], NLA)
        loss_fn = continuous_confounder_outcome_loss
    elif params["model"] == "unobserved_confounder_mild":
        data = KidneyStoneDataset("./data/unobserved_confounder_mild_data.npy", transform=ToTensor(), idx_mean=[2], idx_sd=[0,2])
        model = ContinuousConfounderAndOutcome(params["architecture"], NLA).cuda() if params["cuda"] else ContinuousConfounderAndOutcome(params["architecture"], NLA)
        loss_fn = continuous_confounder_outcome_loss
    elif params["model"] == "unobserved_confounder_strong":
        data = KidneyStoneDataset("./data/unobserved_confounder_strong_data.npy", transform=ToTensor(), idx_mean=[2], idx_sd=[0,2])
        model = ContinuousConfounderAndOutcome(params["architecture"], NLA).cuda() if params["cuda"] else ContinuousConfounderAndOutcome(params["architecture"], NLA)
        loss_fn = continuous_confounder_outcome_loss
    elif params["model"] == "front_door":
        data = KidneyStoneDataset("./data/front_door_data.npy", transform=ToTensor())
        model = FrontDoor(params["architecture"], NLA).cuda() if params["cuda"] else FrontDoor(params["architecture"], NLA)
        loss_fn = front_door_loss

    return data, model, loss_fn


def causal_effect_estimation(model, params, data):
    """ Chooses the right causal estimate depending on the experiment
    """

    if params["model"] == "binary":
        interventional_dist_1 = binary_backdoor_adjustment(model.r_mlp, 1, model.ks_mlp, [0., 1.])
        interventional_dist_0 = binary_backdoor_adjustment(model.r_mlp, 0, model.ks_mlp, [0., 1.])
        causal_effect = interventional_dist_1 - interventional_dist_0
    elif params["model"] == "continuous_outcome":
        interventional_dist_1 = continuous_outcome_backdoor_adjustment(model.r_mlp, 1, model.ks_mlp, [0., 1.], data)
        interventional_dist_0 = continuous_outcome_backdoor_adjustment(model.r_mlp, 0, model.ks_mlp, [0., 1.], data)
        causal_effect = interventional_dist_1 - interventional_dist_0
    elif params["model"] == "continuous_confounder_gamma":
        interventional_dist_1 = continuous_confounder_and_outcome_backdoor_adjustment(model.r_mlp, 1., model.ks_mlp, LogNormal, [1, 5, 50, 100, 1000], data)
        interventional_dist_0 = continuous_confounder_and_outcome_backdoor_adjustment(model.r_mlp, 0., model.ks_mlp, LogNormal, [1, 5, 50, 100, 1000], data)
        causal_effect = [int_1-int_0 for int_1, int_0 in zip(interventional_dist_1, interventional_dist_0)]
    elif params["model"] == "continuous_confounder_logn":
        interventional_dist_1 = continuous_confounder_and_outcome_backdoor_adjustment(model.r_mlp, 1., model.ks_mlp, LogNormal, [1, 5, 50, 100, 1000], data)
        interventional_dist_0 = continuous_confounder_and_outcome_backdoor_adjustment(model.r_mlp, 0., model.ks_mlp, LogNormal, [1, 5, 50, 100, 1000], data)
        causal_effect = [int_1-int_0 for int_1, int_0 in zip(interventional_dist_1, interventional_dist_0)]
    elif params["model"] == "non_linear":
        interventional_dist_1 = continuous_confounder_and_outcome_backdoor_adjustment_linspace(model.r_mlp, 5., 25., 1., data)
        interventional_dist_0 = continuous_confounder_and_outcome_backdoor_adjustment_linspace(model.r_mlp, 5., 25., 0., data)
        causal_effect = [int_1-int_0 for int_1, int_0 in zip(interventional_dist_1, interventional_dist_0)]
    elif params["model"] == "unobserved_confounder_mild":
        interventional_dist_1 = continuous_confounder_and_outcome_backdoor_adjustment_linspace(model.r_mlp, 5., 25., 1., data)
        interventional_dist_0 = continuous_confounder_and_outcome_backdoor_adjustment_linspace(model.r_mlp, 5., 25., 0., data)
        causal_effect = [int_1-int_0 for int_1, int_0 in zip(interventional_dist_1, interventional_dist_0)]
    elif params["model"] == "unobserved_confounder_strong":
        interventional_dist_1 = continuous_confounder_and_outcome_backdoor_adjustment_linspace(model.r_mlp, 5., 25., 1., data)
        interventional_dist_0 = continuous_confounder_and_outcome_backdoor_adjustment_linspace(model.r_mlp, 5., 25., 0., data)
        causal_effect = [int_1-int_0 for int_1, int_0 in zip(interventional_dist_1, interventional_dist_0)]
    elif params["model"] == "front_door":
        causal_effect = "Not implemented"

    return causal_effect

def plot_wrapper(params, data):
    """ Main plotter function
    """

    if params["model"]=="non_linear":
        plot_non_linear(linear_causal_effect, neural_causal_effect, data)



def save_csv(csv_path, save_dict):
    """ Save the parameters and results of an experiment in a .csv file.
    """
    if os.path.exists(csv_path):
        with open(csv_path, 'a') as f:
            fieldnames = save_dict.keys()
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(save_dict)
    else:
        with open(csv_path, 'w') as f:
            fieldnames = save_dict.keys()
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(save_dict)


def main(params, logger):
    """ Main function for the experiments of "Causal effect estimation using neural autoregressive density estimators".
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
    results["causal_effect"] = causal_effect_estimation(model, params, data)

    # log the estimated causal effect
    logger.info(f'The estimated causal effect is: {results["causal_effect"]}')

    # Save the results
    save_dict = {**params, **results}
    # Write the results and architecture in the result.csv file
    save_csv('./results/results.csv', save_dict)


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

    # Set up the experiment name:
    params["name"] = f'{params["model"]}_{params["optimizer"]}_{params["activation"]}_{params["architecture"]}'

    # use GPU if available
    params["cuda"] = torch.cuda.is_available()

    # Set the random seed for reproducible experiments
    torch.manual_seed(params["random_seed"])
    if params["cuda"]:
        torch.cuda.manual_seed(params["random_seed"])

    main(params, logger)
