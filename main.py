#! main.py
""" Main file

TODO: Consider the possibility of having a parameters class.
TODO: Make a function for the logger.

Available functions:
- load_and_intialize
- causal_estimates
- main
"""
import argparse
import csv
import logging
import numpy as np
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
                                continuous_confounder_and_outcome_backdoor_adjustment_linspace, \
                                front_door_adjustment, true_front_door_approximation, \
                                conditional_estimate
from data_loader import KidneyStoneDataset, ToTensor
from model import Binary, ContinuousOutcome, ContinuousConfounderAndOutcome, \
                    FrontDoor, binary_loss, continuous_outcome_loss, \
                    continuous_confounder_outcome_loss, front_door_loss
from plot_utils import plot_non_linear, plot_front_door
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
        data = KidneyStoneDataset("./data/front_door_data.npy", transform=ToTensor(), idx_sd=[0, 1, 2, 3])
        model = FrontDoor(params["architecture"], NLA).cuda() if params["cuda"] else FrontDoor(params["architecture"], NLA)
        loss_fn = front_door_loss

    return data, model, loss_fn


def causal_effect_estimation_and_plotting(model, params, data):
    """ Chooses the right causal estimate and type of plot depending on the experiment
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

        if params["plot"]==True:
            confounder_linspace = np.linspace(5, 25, len(causal_effect))
            true_value = (50/(3+confounder_linspace))
            plot_non_linear(causal_effect, true_value, confounder_linspace, data, params)

    elif params["model"] == "unobserved_confounder_mild":
        interventional_dist_1 = continuous_confounder_and_outcome_backdoor_adjustment_linspace(model.r_mlp, 5., 25., 1., data)
        interventional_dist_0 = continuous_confounder_and_outcome_backdoor_adjustment_linspace(model.r_mlp, 5., 25., 0., data)
        causal_effect = [int_1-int_0 for int_1, int_0 in zip(interventional_dist_1, interventional_dist_0)]

        if params["plot"]==True:
            plot_non_linear(causal_effect, data, params)

    elif params["model"] == "unobserved_confounder_strong":
        interventional_dist_1 = continuous_confounder_and_outcome_backdoor_adjustment_linspace(model.r_mlp, 5., 25., 1., data)
        interventional_dist_0 = continuous_confounder_and_outcome_backdoor_adjustment_linspace(model.r_mlp, 5., 25., 0., data)
        causal_effect = [int_1-int_0 for int_1, int_0 in zip(interventional_dist_1, interventional_dist_0)]

        if params["plot"]==True:
            plot_non_linear(causal_effect, data, params)

    elif params["model"] == "front_door":
        interventional_dist_05 = front_door_adjustment(model, 0.5, data)
        interventional_dist_0 = front_door_adjustment(model, 0., data)

        if params["plot"]==True:
            mc_interventional_dist_05 = true_front_door_approximation(0.5, data, n_samples=500)
            mc_interventional_dist_0 = true_front_door_approximation(0.0, data, n_samples=500)

            plot_front_door(interventional_dist_05, mc_interventional_dist_05, 0.5, params)
            plot_front_door(interventional_dist_0, mc_interventional_dist_0, 0., params)

            conditional_dist_05 = conditional_estimate(model, 0.5, data)
            conditional_dist_0 = conditional_estimate(model, 0., data)

            plot_front_door(conditional_dist_05, mc_interventional_dist_05, 0.5, params, conditional=True)
            plot_front_door(conditional_dist_0, mc_interventional_dist_0, 0., params, conditional=True)
            
        causal_effect = np.mean(interventional_dist_05) - np.mean(interventional_dist_0)

    return causal_effect


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
    # Set up the experiment name (it must contain all the hyper-parameters we are searching over):
    params["name"] = f'{params["model"]}_{params["optimizer"]}_{params["activation"]}_{params["architecture"]}'

    # Create the results folder for that particular experiment:
    if not os.path.exists(f'./results/{params["name"]}'):
        os.mkdir(f'./results/{params["name"]}')

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
    results["causal_effect"] = causal_effect_estimation_and_plotting(model, params, data)

    # Log the estimated causal effect
    logger.info(f'The estimated causal effect is: {results["causal_effect"]}')

    # Save the results
    save_dict = {**params, **results}
    # Write the results and architecture in the result.csv file
    save_csv('./results/results.csv', save_dict)


if __name__ == '__main__':

    # Load the parameters from yaml file
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='./data/',
                        help="Directory containing the dataset")
    parser.add_argument('--yaml_dir', default='./experiments/default_params.yaml',
                        help="Directory containing default_params.yaml")
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

    main(params, logger)
