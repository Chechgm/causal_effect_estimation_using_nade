# ./causal_estimates.py
""" Necessary functions to estimate interventional distributions via do-calculus

Available functions:
- backdoor_adjustment
- frontdoor_adjustment
"""
import torch


def backdoor_adjustment(outcome_model, intervention_value, adjustment_set_model, adjustment_set_values):
    """Estimates the backdoor adjustment

    TODO outcome_model can be a partial function of the adjustment set values
    """
    estimate = 0
    for adjustment_value in adjustment_set_values:
        input_outcome = torch.tensor([adjustment_value, intervention_value]).view(-1, 2)
        input_adjustment = torch.tensor([adjustment_value]).view(-1, 1)
        estimate += (outcome_model(input_outcome)*adjustment_set_model(input_adjustment))

    return estimate.item()
