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
