import csv
import os
import re

from src.utils.plot_utils import bootstrap_plot
from main import load_and_intialize

with open('./results/bootstrap_results.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        params = {'model':row['model'],
                    'activation':row['activation'],
                    'name':row['name'],
                    'architecture':eval(row['architecture']),
                    'batch_size':int(row['batch_size']),
                    'bootstrap_seed':None,
                    'cuda':row['cuda'],
                    'device':'cpu',
                    'learn_rate':float(row['learn_rate']),
                    'optimizer':row['optimizer'],
                    'polynomials':False}
        
        results = {'bootstrap_mean':eval(re.sub(r'(?<=[0-9]) ', ',', row['bootstrap_mean'].replace("\n", ","))), 
                    'bootstrap_lower':eval(re.sub(r'(?<=[0-9]) ', ',', row['bootstrap_lower'].replace("\n", ","))),
                    'bootstrap_upper':eval(re.sub(r'(?<=[0-9]) ', ',', row['bootstrap_upper'].replace("\n", ","))),}

        data, train_loader, model, loss_fn, optimizer = load_and_intialize(params)

        if not os.path.exists(f'./results/{params["name"]}'):
            os.mkdir(f'./results/{params["name"]}')

        bootstrap_plot(results, data, params)