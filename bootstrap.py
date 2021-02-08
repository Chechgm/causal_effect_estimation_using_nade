#! ./bootstrap.py
""" Bootstrap estimation of confidence intervals for a specific experiment.
"""
import argparse
import numpy as np
import os
import yaml

import torch

from train import train
from main import causal_effect_estimation_and_plotting, load_and_intialize, save_csv


def bootstrap_aggregation(bootstrap_estimate):
    """ Aggregating the causal effects into bootstrap estimates of intervals.
    """

    # Initalise the results dictionary
    results = {}

    bootstrap_estimate = np.array(bootstrap_estimate)
    bootstrap_estimate.sort(axis=0)

    # Compute the mean
    results["bootstrap_mean"] = np.mean(bootstrap_estimate, axis=0)
    confidence_idx = int(np.ceil(params["num_bootstrap"]*0.05)) # 10%-90%

    # Compute the confidence bands
    results["bootstrap_lower"] = bootstrap_estimate[confidence_idx]
    results["bootstrap_upper"] = bootstrap_estimate[-confidence_idx]

    return results

def no_name_yet():
    """
    """
    if params["plot"]==True:
            confounder_linspace = np.linspace(5, 25, len(causal_effect))
            true_value = (50/(3+confounder_linspace))
            plot_non_linear(causal_effect, true_value, confounder_linspace, data, params)

    plot_non_linear(results["bootstrap_mean"], true_value, confounder_linspace, data, params, 
        bootstrap_bands=(results["bootstrap_lower"], results["bootstrap_upper"]))


def bootstrap_estimation(params):
    """ Runs bootstrap to estimate the confidence intervals of causal effects.

    Args:
        n: Number of bootstrap samples.
        params: Parameters of the experiment.
    """
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

    results = bootstrap_aggregation(bootstrap_estimate)

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