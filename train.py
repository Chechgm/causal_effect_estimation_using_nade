# !./train.py
""" Function to train a neural network using PyTorch.
"""
import logging
import numpy as np
from statistics import mean
from tqdm import trange

from causal_estimates import true_front_door_approximation

# Train logger set-up
logging.basicConfig(filename='./results/training_logger.log',
                    format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)


def train(model, optimizer, loss_fn, data_iterator, params):
    """ Train the model for num_epochs times.
    The model instantiated class is modified so we dont need to return anything.

    Args:
        model: (torch.nn.Module).
        optimizer: (torch.optim).
        loss_fn: a function that takes the output of the neural network and the training batch and returns the negative log-likelihood.
        data_iterator: (torch DataLoader) a generator that generates batches of data and labels.
        num_epochs: (int) number of batches to train on, each of size params.batch_size.
        report (float) : percentage of the data at which should report.
    """
    cum_loss = []

    pbar = trange(params["num_epochs"], ascii=True, desc=f'Training {params["model"]} ')

    for i in pbar:
        for train_batch in data_iterator:
            # Forward pass of the neural network
            output = model(train_batch)
            loss = loss_fn(output, train_batch)

            # Clear all the previous gradients and estimate the gradients of the
            #  loss with respect to the parameters
            optimizer.zero_grad()
            loss.backward()

            # Make a step
            optimizer.step()

            # Save the loss for visualization
            cum_loss.append(loss.data.numpy())

        if i%10 == 0:
            pbar.set_postfix(Loss=cum_loss[-1])

    logger.info('The final loss of model: %s, is: %.2f', model.__class__.__name__, cum_loss[-1])

    return cum_loss

def evaluate(params, results, data):
    """ Evaluation of the models against ground-truth.
    """
    causal_effect = results["causal_effect"]

    if params["model"] == "binary":
        evaluation = abs(causal_effect-0.0632)

    elif params["model"] == "continuous_outcome":
        evaluation = abs(causal_effect-4.)

    elif params["model"] == "continuous_confounder_gamma":
        evaluation = [abs(ce-4.) for ce in causal_effect]

    elif params["model"] == "continuous_confounder_logn":
        evaluation = [abs(ce-4.) for ce in causal_effect]

    elif params["model"] == "non_linear":
        confounder_linspace = np.linspace(5, 25, len(causal_effect))
        true_value = (50/(3+confounder_linspace)).tolist()
        evaluation = mean([abs(ce-tv) for ce, tv in zip(causal_effect, true_value)])

    elif params["model"] == "mild_unobserved_confounder":
        confounder_linspace = np.linspace(5, 25, len(causal_effect))
        true_value = (50/(3+confounder_linspace) + 0.3).tolist()
        evaluation = mean([abs(ce-tv) for ce, tv in zip(causal_effect, true_value)])

    elif params["model"] == "strong_unobserved_confounder":
        confounder_linspace = np.linspace(5, 25, len(causal_effect))
        true_value = (50/(3+confounder_linspace) + 3.).tolist()
        evaluation = mean([abs(ce-tv) for ce, tv in zip(causal_effect, true_value)])

    elif params["model"] == "non_linear_unobserved_confounder":
        confounder_linspace = np.linspace(5, 25, len(causal_effect))
        true_value = (50/(3+confounder_linspace)).tolist()
        evaluation = mean([abs(ce-tv) for ce, tv in zip(causal_effect, true_value)])

    elif params["model"] == "front_door":
        mc_interventional_dist_05 = true_front_door_approximation(0.5, data, n_samples=500)
        mc_interventional_dist_0 = true_front_door_approximation(0.0, data, n_samples=500)

        true_value = np.mean(mc_interventional_dist_05) - np.mean(mc_interventional_dist_0)
        evaluation = abs(causal_effect-true_value)

    return evaluation