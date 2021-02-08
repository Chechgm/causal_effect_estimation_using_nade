#! default_experiment_yaml.py
""" Script to create a default hyperparameters YAML file for the experiments.
"""
import argparse
import os
import yaml

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--params_dir', default=".", help="Directory where the info YAML is going to be saved")
    args = parser.parse_args()

    params = {}
    params["activation"] = "linear"
    params["architecture"] = [4]
    params["batch_size"] = 128
    params["cuda"] = False
    params["learn_rate"] = 1e-2
    params["model"] = "binary"
    params["n_bootstrap"] = None
    params["num_epochs"] = 75
    params["optimizer"] = "rmsprop"
    params["random_seed"] = 42
    params["save_model"] = False

    params_dir = os.path.join(args.params_dir, "default_params.yaml")

    with open(params_dir, 'w') as f:
        yaml.dump(params, f)

    print("Done building and saving the Hyperparameters YAML")
