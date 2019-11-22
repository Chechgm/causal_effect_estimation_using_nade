import argparse
import logging
import os

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

# Internal packages
from data_loader import KidneyStoneDataset, ToTensor
from model import binary_ks_net, binary_neg_loglik
from train import train

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/ks_binary_data.npy',
                    help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Directory containing params.json")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'best' or 'train'

# Hyperparameters
BATCH_SIZE = 128
EPOCHS     = 150
LEARN_R    = 1e-2
N_HU       = 3

# Initialize the dataset
data = KidneyStoneDataset("./data/kidney_data.npy", transform=ToTensor())
train_loader = DataLoader(data, batch_size=BATCH_SIZE)

# Initialize the model
model = binary_ks_net(N_HU)

# Optimizers
#optimizer = optim.SGD(model.parameters(), lr=LEARN_R, weight_decay=0.1)
optimizer = optim.RMSprop(model.parameters(), lr=LEARN_R)

cum_loss = train(model, optimizer, binary_neg_loglik, train_loader, EPOCHS)


import logging
import torch

# Train logger set-up
logging.basicConfig(filename='./logger.log', format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)





def train(model, optimizer, loss_fn, data_iterator, num_epochs):
    """ Train the model for num_epochs times.
    The model instantiated class is modified so we dont need to return anything
    Arguments:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim)
        loss_fn: a function that takes the output of the neural network and the training batch and returns the negative log-likelihood
        data_iterator: (torch DataLoader) a generator that generates batches of data and labels
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """
    cum_loss = []
    for e in range(num_epochs):
        for train_batch in data_iterator:
            # Forward pass of the neural network
            output = model(train_batch)
            loss = loss_fn(output, train_batch)

            # Clear all the previous gradients and estimate the gradients of the loss with respect to the parameters
            optimizer.zero_grad()
            loss.backward()

            # Make a step
            optimizer.step()

            # Save the loss for visualization
            cum_loss.append(loss.data.numpy())

    logger.info('The final loss of the model is: %.2f', cum_loss[-1])

    return cum_loss






if __name__ == '__main__':

    # Load the parameters from json file
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
