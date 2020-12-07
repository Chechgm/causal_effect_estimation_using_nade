#! binary_main.py
""" Main file

TODO: finish loading the data and the Hyperparameters (probably make a class as in the class)
TODO: initialize the model and the optimizer
TODO: save the model results in a csv or something similar

The process that is followed in this file is:
1.
"""
import argparse
import logging
import os

import torch
from torch.utils.data import DataLoader
import torch.optim as optim

# Internal packages
from data_loader import KidneyStoneDataset, ToTensor
from model import binary_ks_net, binary_neg_loglik
from train import train

# Hyperparameters
BATCH_SIZE = 128
EPOCHS = 150
LEARN_R = 1e-2
N_HU = 3


def main(args):
    # Initialize the dataset
    data = KidneyStoneDataset("./data/kidney_data.npy", transform=ToTensor())
    train_loader = DataLoader(data, batch_size=args.BATCH_SIZE)

    # Initialize the model
    model = binary_ks_net(args.N_HU)

    # Optimizers
    #optimizer = optim.SGD(model.parameters(), lr=LEARN_R, weight_decay=0.1)
    optimizer = optim.RMSprop(model.parameters(), lr=LEARN_R)

    cum_loss = train(model, optimizer, binary_neg_loglik, train_loader, EPOCHS)

    cum_loss[-1]


if __name__ == '__main__':

    # Load the parameters from json file
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data/ks_binary_data.npy',
                        help="Directory containing the dataset")
    parser.add_argument('--model_dir', default='experiments/base_model',
                        help="Directory containing params.json")
    parser.add_argument('--restore_file', default=None,
                        help="Optional, name of the file in --model_dir containing weights to reload before \
                        training")  # 'best' or 'train'
    args = parser.parse_args()





    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # use GPU if available
    #params.cuda = torch.cuda.is_available()

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    #if params.cuda:
    #    torch.cuda.manual_seed(230)

    # Set the logger
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))

    # Create the input data pipeline
    logging.info("Loading the datasets...")

    # fetch dataloaders
    dataloaders = data_loader.fetch_dataloader(
        ['train', 'val'], args.data_dir, params)
    train_dl = dataloaders['train']
    val_dl = dataloaders['val']

    logging.info("- done.")

    # Define the model and optimizer
    model = net.Net(params).cuda() if params.cuda else net.Net(params)
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)

    # fetch loss function and metrics
    loss_fn = net.loss_fn
    metrics = net.metrics

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    train_and_evaluate(model, train_dl, val_dl, optimizer, loss_fn, metrics, params, args.model_dir,
                       args.restore_file)
