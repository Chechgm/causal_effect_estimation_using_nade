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
import logging
import numpy as np
import os
import yaml

import torch
from torch import nn
from torch.distributions.log_normal import LogNormal
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

from src.models.causal_estimates import binary_backdoor_adjustment, \
                                continuous_outcome_backdoor_adjustment, \
                                continuous_confounder_and_outcome_backdoor_adjustment, \
                                continuous_confounder_and_outcome_backdoor_adjustment_linspace, \
                                front_door_adjustment, true_front_door_approximation, \
                                conditional_estimate
from src.models.data_loader import KidneyStoneDataset, ToTensor
from src.models.model import Binary, ContinuousOutcome, ContinuousConfounderAndOutcome, \
                    FrontDoor, binary_loss, continuous_outcome_loss, \
                    continuous_confounder_outcome_loss, front_door_loss
from src.utils.plot_utils import plot_non_linear, plot_front_door, plot_loss
from src.models.train import train, evaluate
from src.utils.utils import get_freer_gpu, initialize_logger


def get_args():
    """ Get the arguments to be passed to the main function.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--yaml_dir', default='./experiments/default_params.yaml',
                        help="Directory containing default_params.yaml")

    return parser.parse_args()


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
        data = KidneyStoneDataset("./data/binary_data.npy", 
                                    bootstrap=params["bootstrap_seed"], 
                                    transform=ToTensor())
        model = Binary(params["architecture"], NLA).to(params["device"])
        loss_fn = binary_loss

    elif params["model"] == "continuous_outcome":
        data = KidneyStoneDataset("./data/continuous_outcome_data.npy", 
                                    bootstrap=params["bootstrap_seed"], 
                                    transform=ToTensor(), idx_mean=[2], idx_sd=[2])
        model = ContinuousOutcome(params["architecture"], NLA).to(params["device"])
        loss_fn = continuous_outcome_loss

    elif params["model"] == "continuous_confounder_gamma":
        data = KidneyStoneDataset("./data/continuous_confounder_gamma_data.npy", 
                                    bootstrap=params["bootstrap_seed"], 
                                    transform=ToTensor(), idx_mean=[2], idx_sd=[0,2])
        model = ContinuousConfounderAndOutcome(params["architecture"], NLA).to(params["device"])
        loss_fn = continuous_confounder_outcome_loss

    elif params["model"] == "continuous_confounder_logn":
        data = KidneyStoneDataset("./data/continuous_confounder_logn_data.npy", 
                                    bootstrap=params["bootstrap_seed"], 
                                    transform=ToTensor(), idx_mean=[2], idx_sd=[0,2])
        model = ContinuousConfounderAndOutcome(params["architecture"], NLA).to(params["device"])
        loss_fn = continuous_confounder_outcome_loss

    elif params["model"] == "non_linear":
        data = KidneyStoneDataset("./data/non_linear_data.npy", 
                                    bootstrap=params["bootstrap_seed"], 
                                    transform=ToTensor(), idx_mean=[2], 
                                    idx_sd=[0,2], use_polynomials=params["polynomials"])
        model = ContinuousConfounderAndOutcome(params["architecture"], NLA,
                                                use_polynomials=params["polynomials"]).to(params["device"])
        loss_fn = continuous_confounder_outcome_loss

    elif params["model"] == "mild_unobserved_confounder":
        data = KidneyStoneDataset("./data/mild_unobserved_confounder_data.npy", 
                                    bootstrap=params["bootstrap_seed"], 
                                    transform=ToTensor(), idx_mean=[2], idx_sd=[0,2])
        model = ContinuousConfounderAndOutcome(params["architecture"], NLA,
                                                use_polynomials=params["polynomials"]).to(params["device"])
        loss_fn = continuous_confounder_outcome_loss

    elif params["model"] == "strong_unobserved_confounder":
        data = KidneyStoneDataset("./data/strong_unobserved_confounder_data.npy", 
                                    bootstrap=params["bootstrap_seed"], 
                                    transform=ToTensor(), idx_mean=[2], idx_sd=[0,2])
        model = ContinuousConfounderAndOutcome(params["architecture"], NLA,
                                                use_polynomials=params["polynomials"]).to(params["device"])
        loss_fn = continuous_confounder_outcome_loss

    elif params["model"] == "non_linear_unobserved_confounder":
        data = KidneyStoneDataset("./data/non_linear_unobserved_confounder_data.npy", 
                                    bootstrap=params["bootstrap_seed"], 
                                    transform=ToTensor(), idx_mean=[2], idx_sd=[0,2])
        model = ContinuousConfounderAndOutcome(params["architecture"], NLA,
                                                use_polynomials=params["polynomials"]).to(params["device"])
        loss_fn = continuous_confounder_outcome_loss

    elif params["model"] == "front_door":
        data = KidneyStoneDataset("./data/front_door_data.npy", 
                                    bootstrap=params["bootstrap_seed"], 
                                    transform=ToTensor(), idx_sd=[0, 1, 2, 3])
        model = FrontDoor(params["architecture"], NLA).to(params["device"])
        loss_fn = front_door_loss

    train_loader = DataLoader(data, batch_size=params["batch_size"])

    # Optimizer
    if params["optimizer"] == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=params["learn_rate"])
    elif params["optimizer"] == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), lr=params["learn_rate"])

    return data, train_loader, model, loss_fn, optimizer


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
        interventional_dist_1 = continuous_confounder_and_outcome_backdoor_adjustment_linspace(model.r_mlp, 5., 25., 1., data, use_polynomials=params["polynomials"])
        interventional_dist_0 = continuous_confounder_and_outcome_backdoor_adjustment_linspace(model.r_mlp, 5., 25., 0., data, use_polynomials=params["polynomials"])
        causal_effect = [int_1-int_0 for int_1, int_0 in zip(interventional_dist_1, interventional_dist_0)]

        if params["plot"]==True:
            confounder_linspace = np.linspace(5, 25, len(causal_effect))
            true_value = (50/(3+confounder_linspace))
            plot_non_linear(causal_effect, true_value, confounder_linspace, data, params)

    elif params["model"] == "mild_unobserved_confounder":
        interventional_dist_1 = continuous_confounder_and_outcome_backdoor_adjustment_linspace(model.r_mlp, 5., 25., 1., data, use_polynomials=params["polynomials"])
        interventional_dist_0 = continuous_confounder_and_outcome_backdoor_adjustment_linspace(model.r_mlp, 5., 25., 0., data, use_polynomials=params["polynomials"])
        causal_effect = [int_1-int_0 for int_1, int_0 in zip(interventional_dist_1, interventional_dist_0)]

        if params["plot"]==True:
            confounder_linspace = np.linspace(5, 25, len(causal_effect))
            true_value = (50/(3+confounder_linspace)) + 0.3
            plot_non_linear(causal_effect, true_value, confounder_linspace, data, params)

    elif params["model"] == "strong_unobserved_confounder":
        interventional_dist_1 = continuous_confounder_and_outcome_backdoor_adjustment_linspace(model.r_mlp, 5., 25., 1., data, use_polynomials=params["polynomials"])
        interventional_dist_0 = continuous_confounder_and_outcome_backdoor_adjustment_linspace(model.r_mlp, 5., 25., 0., data, use_polynomials=params["polynomials"])
        causal_effect = [int_1-int_0 for int_1, int_0 in zip(interventional_dist_1, interventional_dist_0)]

        if params["plot"]==True:
            confounder_linspace = np.linspace(5, 25, len(causal_effect))
            true_value = (50/(3+confounder_linspace)) + 3.
            plot_non_linear(causal_effect, true_value, confounder_linspace, data, params)

    elif params["model"] == "non_linear_unobserved_confounder":
        interventional_dist_1 = continuous_confounder_and_outcome_backdoor_adjustment_linspace(model.r_mlp, 5., 25., 1., data, use_polynomials=params["polynomials"])
        interventional_dist_0 = continuous_confounder_and_outcome_backdoor_adjustment_linspace(model.r_mlp, 5., 25., 0., data, use_polynomials=params["polynomials"])
        causal_effect = [int_1-int_0 for int_1, int_0 in zip(interventional_dist_1, interventional_dist_0)]

        if params["plot"]==True:
            confounder_linspace = np.linspace(5, 25, len(causal_effect))
            true_value = (50/(3+confounder_linspace))
            plot_non_linear(causal_effect, true_value, confounder_linspace, data, params)

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


def main(params):
    """ Main function for the experiments of "Causal effect estimation using neural autoregressive density estimators".
    """
    # Initialize the results logger.
    logger = initialize_logger('./results/training_logger.log')

    # use GPU if available
    if params["cuda"] and torch.cuda.is_available():
        params["device"] = torch.tensor(get_freer_gpu(), dtype=float)
    else:
        params["device"] = "cpu"

    # Set the random seed for reproducible experiments
    torch.manual_seed(params["random_seed"])
    if params["cuda"]:
        torch.cuda.manual_seed(params["random_seed"])

    # Set up the experiment name (it must contain all the hyper-parameters we are searching over):
    if "name" not in params:
        params["name"] = f'{params["model"]}_' + f'OPTIM={params["optimizer"]}_' + \
                            f'LR={params["learn_rate"]}_'.replace(".", "-") + f'ACT={params["activation"]}_' + \
                            f'ARCH={str(params["architecture"]).replace("[", "").replace("]", "").replace(", ", "-")}_' + \
                            f'POLY={params["polynomials"]}'

    # Create the results folder for that particular experiment:
    if not os.path.exists(f'./results/{params["name"]}'):
        os.mkdir(f'./results/{params["name"]}')

    # Load the data and initialise the optimizer
    data, train_loader, model, loss_fn, optimizer = load_and_intialize(params)

    # Train the NN
    cum_loss = train(model, optimizer, loss_fn, train_loader, params)
    plot_loss(np.asarray(cum_loss), params)

    # Initalise the results dictionary
    results = {}

    # Evaluate
    model.eval()
    results["final_loss"] = cum_loss[-1]
    results["causal_effect"] = causal_effect_estimation_and_plotting(model.to("cpu").float(), params, data)
    results["evaluation"] = evaluate(params, results, data)

    # Log the estimated causal effect
    logging.info(f'The estimated causal effect is: {results["causal_effect"]}')

    # Save the results
    save_dict = {**params, **results}
    # Write the results and architecture in the result.csv file
    save_csv('./results/results.csv', save_dict)


if __name__ == '__main__':

    # Parse the arguments:  
    args = get_args()

    assert os.path.isfile(
        args.yaml_dir), "No YAML configuration file found at {}".format(args.yaml_path)

    with open(args.yaml_dir, 'r') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    main(params)
