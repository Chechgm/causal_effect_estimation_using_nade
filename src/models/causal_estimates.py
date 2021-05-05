# ./causal_estimates.py
""" Necessary functions to estimate interventional distributions via do-calculus

Available functions:
- binary_backdoor_adjustment
- continuous_outcome_backdoor_adjustment
- continuous_confounder_and_outcome_backdoor_adjsutment
- continuous_confounder_and_outcome_backdoor_adjsutment_linspace
- frontdoor_adjustment
- true_front_door_approximation
- conditional_estimate
"""
import numpy as np

import torch
from torch.distributions.normal import Normal


def binary_backdoor_adjustment(outcome_model, value_intervention, confounder_model, adjustment_set_values):
    """Estimates the backdoor adjustment for a fully binary model
    """
    estimate = 0
    p_confounder = confounder_model(torch.tensor([1.]).view(-1, 1))
    for adjustment_value in adjustment_set_values:
        input_outcome = torch.tensor([adjustment_value, value_intervention]).view(-1, 2)
        estimate += (outcome_model(input_outcome)*(torch.abs(1-adjustment_value-p_confounder)))

    return estimate.item()


def continuous_outcome_backdoor_adjustment(outcome_model, value_intervention, confounder_model, adjustment_set_values, data):
    """Estimates the backdoor adjustment for a continuous outcome variable
    """
    estimate = 0
    p_confounder = confounder_model(torch.tensor([1.]).view(-1, 1))
    for adjustment_value in adjustment_set_values:
        input_outcome = torch.tensor([adjustment_value, value_intervention]).view(-1, 2)
        mean, _ = outcome_model(input_outcome)
        mean = (mean*data.sd[2])+data.mean[2]
        estimate += (mean*(torch.abs(1-adjustment_value-p_confounder)))

    return estimate.item()


def continuous_confounder_and_outcome_backdoor_adjustment(outcome_model, value_intervention, confounder_model, confounder_dist, n_samples, data):
    """ Estimates the causal effect by backdoor adjustment when confounder and outcome are continuous
    """
    estimate = []
    n_rows = max(n_samples)

    loc_confounder, log_scale_confounder = confounder_model(torch.tensor([1.]).view(-1,1))
    scale_confounder = torch.exp(log_scale_confounder)
    confounder = confounder_dist(loc_confounder, scale_confounder)
    samples_confounder = confounder.sample((n_rows,)).view(n_rows, 1)

    intervention = torch.ones(n_rows, 1)*value_intervention
    input_outcome = torch.cat((samples_confounder, intervention), 1)
    means_outcome, _ = outcome_model(input_outcome.view(-1, 2))
    means_outcome = (means_outcome*data.sd[2])+data.mean[2]

    for n in n_samples:
        estimate.append(torch.mean(means_outcome[:n]).item())

    return estimate


def continuous_confounder_and_outcome_backdoor_adjustment_linspace(outcome_model, min_confounder, max_confounder, value_intervention, data):
    """ Estimates the causal effect for a non-linear model using linspace
    """
    confounder_linspace = torch.arange(min_confounder, max_confounder, 0.1)/data.sd[0]
    n = confounder_linspace.shape[0]

    intervention = torch.ones(n, 1)*value_intervention
    means_outcome, _ = outcome_model(torch.cat((confounder_linspace.view(-1,1), intervention), 1))

    estimate = np.squeeze((means_outcome*data.sd[2]+data.mean[2]).detach().numpy()).tolist()

    return estimate


def front_door_adjustment(model, value_intervention, data, n_samples = 5000):
    """ Estimates an interventional distribution using the front-door adjustment.

    Z is the mediator, X is the confounded treatment and Y is the outcome variable.

    sum(z) P(Z=z | X=int_x) sum(x') P(Y|X=x', Z=z)P(X=x')
    """
    # P(X=x')
    input_x_mlp = torch.tensor([1.]).view(-1,1)
    mu_x, log_sigma_x = model.x_mlp(input_x_mlp)
    sigma_x = torch.exp(log_sigma_x)

    x_dist = Normal(mu_x, sigma_x)
    x_samples = x_dist.sample((n_samples,)).view(n_samples,1)

    # P(Z=z | X=int_x)
    input_z_mlp = torch.tensor([value_intervention/data.sd[0]]).view(-1,1)
    mu_z, log_sigma_z = model.z_mlp(input_z_mlp)
    sigma_z = torch.exp(log_sigma_z)

    z_dist = Normal(mu_z, sigma_z)
    z_samples = z_dist.sample((n_samples,)).view(n_samples,1)

    # P(Y|X=x', Z=z)
    y_aux_input = torch.cat([x_samples, z_samples], dim=1)
    means_outcome, _ = model.y_aux_mlp(y_aux_input)

    estimate = np.squeeze((means_outcome*data.sd[2]).detach().numpy()).tolist()

    return estimate


def true_front_door_approximation(x, data, n_samples=500):
    """
    This function approximates the front-door estimation of the effect of do(X=x) on y
    Since the effects are non-linear, the conditional distributions must be estimated.
    These will be approximated from the data using the Monte Carlo (MC) method. 

    p(y | do(X=x)) = sum_z p(z|x) sum_x' p(y| z, x') p(x')

    The MC method is used on the second sum: sum_x' p(y | z, x') p(x') by sampling
    from the marginal distribution of x', p(x').
    
    Inputs:
        x (float): the intervention value of x
        data (ndarray): the data to be estimated from
        n_samples: The number of samples to be used for the monte carlo integration
    """
    np.random.seed(42)
    data = np.round(data.ks_dataset, 2)
    x = round(x, 2)
    
    # p(z|x)
    z = np.random.normal(1+(-x)**2, 0.1, size=100)
    
    # p(x)
    x_marginal = data[:,0]
    x_interval, x_counts = np.unique(x_marginal, return_counts=True)
    prob_x = x_counts/np.sum(x_counts)
    x_marginal = np.random.choice(x_interval, p=prob_x, size=n_samples)
    
    # p(u|x)
    u = np.zeros_like(x_marginal)
    for i, x_i in enumerate(x_marginal):
        # Memoization mechanism so we don't have to repeat computation
        mem = {}
        if x_i in mem:
            u_interval, u_counts = mem[x_i]
            prob_u_given_x = u_counts/np.sum(u_counts)
        else:
            mask = (data[:,0]==x_i)
            u_given_x = data[mask, 3]
            mem[x_i] = np.unique(u_given_x, return_counts=True)
            u_interval, u_counts = mem[x_i]
            prob_u_given_x = u_counts/np.sum(u_counts)

        # Fill in the sample
        u[i] = np.random.choice(u_interval, p=prob_u_given_x)
        
    # p(y | z, x)
    z_u = np.array(np.meshgrid(z, u)).T.reshape(-1,2)
    y   = np.random.normal(np.sin(z_u[:,1]**2) + 5/(z_u[:,0]), 0.1)
            
    return y


def conditional_estimate(model, conditioning_value, data, n_samples=5000):
    """ Conditional effect estimation for comparison purposes.

    P(Y | X, Z)
    """
    # Mediator (Z) samples
    z_samples = torch.tensor(np.random.choice(data.ks_dataset[:,1], size=n_samples)).float()/data.sd[1]

    # P(Y|X=x', Z=z)
    aux_input = torch.cat([torch.ones([n_samples, 1])*conditioning_value/data.sd[0], z_samples.view(-1,1)], dim=1)
    means_outcome, _ = model.y_aux_mlp(aux_input)

    estimate = np.squeeze((means_outcome*data.sd[2]).detach().numpy()).tolist()

    return estimate
