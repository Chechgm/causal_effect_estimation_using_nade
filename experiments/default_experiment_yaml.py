#! default_experiment_yaml.py
""" Script to create a default hyperparameters YAML file for the experiments.
"""
import argparse
import os
from torch import nn
import yaml

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--params_dir', default=".", help="Directory where the info YAML is going to be saved")
    args = parser.parse_args()

    params = {}
    params["batch_size"] = 128
    params["num_epochs"] = 150
    params["learn_rate"] = 1e-2
    params["architecture"] = [4]
    params["activation"] = "linear"  # nn.LeakyReLU(1)
    #params["activation"] = F.relu

    params_dir = os.path.join(args.params_dir, "default_params.yaml")

    with open(params_dir, 'w') as f:
        yaml.dump(params, f)

    print("Done building and saving the Hyperparameters YAML")
