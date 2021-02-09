#! ./bootstrap.py
""" Bootstrap estimation of confidence intervals for a specific experiment.

Available functions:
- bootstrap_statistics
- bootstrap_plot
- bootstrap_estimation
"""
import argparse
import numpy as np
import os
import yaml

import torch

from main import causal_effect_estimation_and_plotting, load_and_intialize, save_csv
from plot_utils import plot_non_linear, plot_front_door
from train import train


def bootstrap_statistics(bootstrap_estimate, data, params):
    """ Estimates the bootstrap statistics of  the causal effects.
    
    The current supported bootstrap estimates are:
    - Mean.
    - 10%-90% intervals.
    """

    # Initalise the results dictionary
    results = {}

    bootstrap_estimate = np.array(bootstrap_estimate)
    bootstrap_estimate = np.sort(bootstrap_estimate, axis=0)

    # Compute the mean
    results["bootstrap_mean"] = np.mean(bootstrap_estimate, axis=0)
    lower_idx = int(np.ceil(params["num_bootstrap"]*0.1)) # 10%-90%
    upper_idx = int(np.floor(params["num_bootstrap"]*0.9))

    # Compute the confidence bands
    results["bootstrap_lower"] = bootstrap_estimate[lower_idx]
    results["bootstrap_upper"] = bootstrap_estimate[upper_idx]

    return results

def bootstrap_plot(results, data, params):
    """ Selects the right true values for every type of experiment and plots the bootstrap estimates.
    """

    if params["model"] == "non_linear":
        confounder_linspace = np.linspace(5, 25, len(results["bootstrap_mean"]))
        true_value = (50/(3+confounder_linspace))
        plot_non_linear(results["bootstrap_mean"], true_value, confounder_linspace, data, params, 
                            bootstrap_bands=(results["bootstrap_lower"], results["bootstrap_upper"]))

    elif params["model"] == "mild_unobserved_confounder":
        confounder_linspace = np.linspace(5, 25, len(results["bootstrap_mean"]))
        true_value = (50/(3+confounder_linspace)) + 0.3
        plot_non_linear(results["bootstrap_mean"], true_value, confounder_linspace, data, params, 
                            bootstrap_bands=(results["bootstrap_lower"], results["bootstrap_upper"]))

    elif params["model"] == "strong_unobserved_confounder":
        confounder_linspace = np.linspace(5, 25, len(results["bootstrap_mean"]))
        true_value = (50/(3+confounder_linspace)) + 3.
        plot_non_linear(results["bootstrap_mean"], true_value, confounder_linspace, data, params, 
                            bootstrap_bands=(results["bootstrap_lower"], results["bootstrap_upper"]))

    elif params["model"] == "non_linear_unobserved_confounder":
        confounder_linspace = np.linspace(5, 25, len(results["bootstrap_mean"]))
        true_value = (50/(3+confounder_linspace))
        plot_non_linear(results["bootstrap_mean"], true_value, confounder_linspace, data, params, 
                            bootstrap_bands=(results["bootstrap_lower"], results["bootstrap_upper"]))


def bootstrap_estimation(params):
    """ Runs bootstrap to estimate the confidence intervals of causal effects.

    Args:
        n: Number of bootstrap samples.
        params: Parameters of the experiment.
    """
    # Set up the experiment name (it must contain all the hyper-parameters we are searching over):
    if "name" not in params:
        params["name"] = f'bootstrap_{params["model"]}_' + f'{params["optimizer"]}_' + \
                            f'{params["learn_rate"]}_'.replace(".", "-") + f'{params["activation"]}_' + \
                            f'{str(params["architecture"]).replace("[", "").replace("]", "").replace(", ", "-")}'

    # Create the results folder for that particular experiment:
    if not os.path.exists(f'./results/{params["name"]}'):
        os.mkdir(f'./results/{params["name"]}')

    # Set the random seed for reproducible experiments
    torch.manual_seed(params["random_seed"])
    if params["cuda"]:
        torch.cuda.manual_seed(params["random_seed"])

    params["plot"] = False

    bootstrap_estimate = []
    for b in range(params["num_bootstrap"]):
        params["bootstrap_seed"] = b
        data, train_loader, model, loss_fn, optimizer = load_and_intialize(params)
        _ = train(model, optimizer, loss_fn, train_loader, params)
        bootstrap_estimate.append(causal_effect_estimation_and_plotting(model, params, data))

    results = bootstrap_statistics(bootstrap_estimate, data, params)
    bootstrap_plot(results, dat, params)

    # Save the results
    save_dict = {**params, **results}
    # Write the results and architecture in the result.csv file
    save_csv('./results/bootstrap_results.csv', save_dict)


if __name__ == "__main__":

    # Load the default arameters from yaml file
    parser = argparse.ArgumentParser()
    parser.add_argument('--default_yaml_dir', default='./experiments/default_params.yaml',
                        help="Directory containing default_params.yaml")                     
    args = parser.parse_args()

    assert os.path.isfile(
        args.default_yaml_dir), "No YAML configuration file found at {}".format(args.default_yaml_path)

    with open(args.default_yaml_dir, 'r') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    bootstrap_estimation(params)