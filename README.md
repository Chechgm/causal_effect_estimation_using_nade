# Estimating causal effects using neural autoregressive density estimators

This repository contains the code for the paper "[Estimating causal effects using neural autoregressive density estimators](https://arxiv.org/abs/2008.07283)". The repository is composed of 6 key scripts:
- ```./data/fake_data.py```
- ```./experiments/default_experiment_yaml.py```
- ```./model.py```
- ```./causal_estimates.py```
- ```./train.py```
- ```./main.py```
- ```./utils.py```

And n auxiliary scripts:
- ```./plot_utils.py```

In order to run the experiments, we need to generate the data and run a command line tool. The process is the following:

## Generate fake data

To generate fake data, run in the command line:
```python3 fake_data.py n path```
Where ```n``` is the number of samples you want to generate and ```path``` is the folder where you want to save the fake data.

## Run the experiments

In order to run an experiment, you need a YAML file with the parameters. In the "experiments" folder you can find a tool to create a default parameters YAML. In order to run that script, run in the command line ```python3 ./experiments/default_experiment_yaml.py dir``` where ```dir``` is the directory where you want to save the parameters YAML.

Having the data, and the YAML parameters file, we can run an experiment by running ```python3 main.py data_dir yaml_dir```. The results will be recorded in the "results" directory.

## Reference

In order to cite this code or the paper, use the following bib:
```
@article{garrido2020estimating,
  title={Estimating Causal Effects with the Neural Autoregressive Density Estimator},
  author={Garrido, Sergio and Borysov, Stanislav S and Rich, Jeppe and Pereira, Francisco C},
  journal={arXiv preprint arXiv:2008.07283},
  year={2020}
}
```

### Disclaimer:
This repository is still a work in progress.
