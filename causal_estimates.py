# ./causal_estimates.py
""" Necessary functions to estimate interventional distributions via do-calculus

Available functions:
- binary_backdoor_adjustment
- continuous_outcome_backdoor_adjustment
- frontdoor_adjustment
"""
import torch


def binary_backdoor_adjustment(outcome_model, intervention_value, adjustment_set_model, adjustment_set_values):
    """Estimates the backdoor adjustment for a fully binary model
    """
    estimate = 0
    p_confounder = adjustment_set_model(torch.tensor([1.]).view(-1, 1))
    for adjustment_value in adjustment_set_values:
        input_outcome = torch.tensor([adjustment_value, intervention_value]).view(-1, 2)
        estimate += (outcome_model(input_outcome)*(torch.abs(1-adjustment_value-p_confounder)))

    return estimate.item()


def continuous_outcome_backdoor_adjustment(outcome_model, intervention_value, adjustment_set_model, adjustment_set_values):
    """Estimates the backdoor adjustment for a continuous outcome variable
    """
    estimate = 0
    p_confounder = adjustment_set_model(torch.tensor([1.]).view(-1, 1))
    for adjustment_value in adjustment_set_values:
        input_outcome = torch.tensor([adjustment_value, intervention_value]).view(-1, 2)
        mean, _ = outcome_model(input_outcome)
        estimate += (mean*(torch.abs(1-adjustment_value-p_confounder)))

    return estimate.item()
