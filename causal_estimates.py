# ./causal_estimates.py
""" Necessary functions to estimate interventional distributions via do-calculus

Available functions:
- binary_backdoor_adjustment
- continuous_outcome_backdoor_adjustment
- continuous_confounder_and_outcome_backdoor_adjsutment
- continuous_confounder_and_outcome_backdoor_adjsutment_linspace
- frontdoor_adjustment
"""
import numpy as np
import torch


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
