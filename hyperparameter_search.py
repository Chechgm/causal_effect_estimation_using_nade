#! ./hyperparameter_search.py
""" Performs hyperparameter search over a predefined space.
"""
import argparse
import os
from subprocess import check_call
import sys
import yaml

from main import main

PYTHON = sys.executable

def launch_job(params):
    """ Launch a job for a specific set of parameters.
    """
    # Set the name for the experiment:
    params["name"] = f'{params["model"]}_' + f'OPTIM={params["optimizer"]}_' + \
                        f'LR={params["learn_rate"]}_'.replace(".", "-") + f'ACT={params["activation"]}_' + \
                        f'ARCH={str(params["architecture"]).replace("[", "").replace("]", "").replace(", ", "-")}_' + \
                        f'POLY={params["polynomials"]}'

    # Create the results folder for that particular experiment, if it doesn't exist:
    if not os.path.exists(f'./results/{params["name"]}'):
        os.mkdir(f'./results/{params["name"]}')
    
    # Save the hyperparameters used in that experiment
    params_dir = os.path.join(f'./results/{params["name"]}', "experiment_params.yaml")

    with open(params_dir, 'w') as f:
        yaml.dump(params, f)

    # Launch training with this config
    cmd = f"{PYTHON} main.py --yaml_dir={params_dir}"
    print(cmd)
    check_call(cmd, shell=True)


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

    # Define the search space for the hyperparameters
    # task_space = ["binary", "continuous_outcome", "continuous_confounder_gamma",
    # "continuous_confounder_logn", "non_linear", "mild_unobserved_confounder", 
    # "strong_unobserved_confounder", "non_linear_unobserved_confounder",
    # "front_door"]
    task_space = ["non_linear", "mild_unobserved_confounder", 
    "strong_unobserved_confounder", "non_linear_unobserved_confounder",
    "front_door"]
    activation_space = ["linear"]#, "relu", "tanh"]
    learn_rate = [1.e-2, 5.e-3, 1.e-3, 5.e-4]
    #optimizer_space = ["sgd", "rmsprop"]
    optimizer_space = ["rmsprop"]
    architecture_space = [
        #[4],
        [8],
        [16],
        [4, 4],
        [8, 8]
    ]

    for ac in activation_space:
        for ar in architecture_space:
            for lr in learn_rate:
                for op in optimizer_space:
                    for tk in task_space:
                        params["activation"] = ac
                        params["architecture"] = ar
                        params["learn_rate"] = lr
                        params["optimizer"] = op
                        params["model"] = tk
                        
                        launch_job(params)
