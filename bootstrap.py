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
from tqdm import trange
import yaml

import torch

from main import causal_effect_estimation_and_plotting, load_and_intialize, save_csv
from src.utils.utils import get_freer_gpu
from src.utils.plot_utils import bootstrap_plot
from src.models.train import train


def bootstrap_statistics(bootstrap_estimate):
    """ Estimates the bootstrap statistics of  the causal effects.
    
    The current supported bootstrap estimates are:
    - Mean.
    - 10%-90% intervals.
    """

    # Initalise the results dictionary
    results = {}

    bootstrap_estimate = np.array(bootstrap_estimate)
    idx = np.unique(np.where(~np.isnan(bootstrap_estimate))[0])
    bootstrap_estimate = bootstrap_estimate[:,idx]
    num_bootstrap = len(bootstrap_estimate)

    bootstrap_estimate = np.sort(bootstrap_estimate, axis=0)

    # Compute the mean
    results["bootstrap_mean"] = np.mean(bootstrap_estimate, axis=0)
    lower_idx = int(np.ceil(num_bootstrap*0.1)) # 10%-90%
    upper_idx = int(np.floor(num_bootstrap*0.9))

    # Compute the confidence bands
    results["bootstrap_lower"] = bootstrap_estimate[lower_idx]
    results["bootstrap_upper"] = bootstrap_estimate[upper_idx]

    return results


def bootstrap_estimation(params):
    """ Runs bootstrap to estimate the confidence intervals of causal effects.

    Args:
        n: Number of bootstrap samples.
        params: Parameters of the experiment.
    """
    # Set up the experiment name (it must contain all the hyper-parameters we are searching over):
    if "name" not in params:
        params["name"] = f'bootstrap_{params["model"]}_' + f'OPTIM={params["optimizer"]}_' + \
                            f'LR={params["learn_rate"]}_'.replace(".", "-") + f'ACT={params["activation"]}_' + \
                            f'ARCH={str(params["architecture"]).replace("[", "").replace("]", "").replace(", ", "-")}_' + \
                            f'POLY={params["polynomials"]}'

    # Create the results folder for that particular experiment:
    if not os.path.exists(f'./results/{params["name"]}'):
        os.mkdir(f'./results/{params["name"]}')

    # use GPU if available
    if params["cuda"] and torch.cuda.is_available():
        params["device"] = torch.tensor(get_freer_gpu(), dtype=float)
    else:
        params["device"] = "cpu"

    # Set the random seed for reproducible experiments
    torch.manual_seed(params["random_seed"])
    if params["cuda"]:
        torch.cuda.manual_seed(params["random_seed"])

    params["plot"] = False

    bootstrap_estimate = []
    for b in trange(params["num_bootstrap"], desc="Bootstrap sample"):
        params["bootstrap_seed"] = b
        data, train_loader, model, loss_fn, optimizer = load_and_intialize(params)
        _ = train(model, optimizer, loss_fn, train_loader, params)
        bootstrap_estimate.append(causal_effect_estimation_and_plotting(model, params, data))

    results = bootstrap_statistics(bootstrap_estimate)
    bootstrap_plot(results, data, params)

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