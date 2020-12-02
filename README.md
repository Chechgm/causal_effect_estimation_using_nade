# Estimating causal effects using neural autoregressive density estimators

This repository contains the code for the paper "Estimating causal effects using neural autoregressive density estimators". The repository is composed of N key scripts:
- ```./data/fake_data.py```
- ```./data_loader.py```
- ```./model.py```
- ```./train.py```
- ```./main.py```
- ```./utils.py```

In order to run the experiments, we need to generate the data and run a command line tool. The process is the following:

## Generate fake data

To generate fake data, run in the command line:
```python3 fake_data.py n path```
Where ```n``` is the number of samples you want to generate and ```path``` is the folder where you want to save the fake data.

## Run the experiments

To run the experiments, run in the command line:
