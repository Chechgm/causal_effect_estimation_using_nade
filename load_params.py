import argparse
import yaml

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--params_dir', default="./experiments/default_params.yaml", help="Directory where the info YAML is going to be saved")
    args = parser.parse_args()

    with open(args.params_dir, 'r') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    print(params)
