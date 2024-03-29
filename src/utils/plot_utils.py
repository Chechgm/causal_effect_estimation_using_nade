# ./plot_utils.py
""" Script where all the plot utilities are coded.

The available functions are:
- plot_non_linear
- plot_front_door
- plot_loss
"""
import torch
import matplotlib.pyplot as plt
import numpy as np
import scipy
import seaborn as sns


def plot_non_linear(estimate, true_value, confounder_linspace, data, params, bootstrap_bands=None):
    """ Utility to plot neural vs. linear treatment effects
    """
    if params["activation"]=="linear":
        label = "Linear"
    else:
        label = "Neural"

    ax = sns.lineplot(x=confounder_linspace, y=estimate, label=f"{label} TE")
    ax = sns.lineplot(x=confounder_linspace, y=true_value, label="True TE")
    ax = sns.histplot(x=data.ks_dataset[:,0]*data.sd[0], element='step', alpha=.5, 
                        color='silver', weights=0.008*np.ones(len(data.ks_dataset)), 
                        bins=35, legend=False)
    if bootstrap_bands is not None:
        ax.fill_between(x=confounder_linspace, y1=bootstrap_bands[0], y2=bootstrap_bands[1], alpha=.5)

    plt.title("Comparison between true and \n estimated conditional Treatment Effects", y=1.10)
    plt.xlabel("Kidney Stone Size")
    plt.ylabel("Treatment Effect")
    ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.10), borderaxespad=0, frameon=False)

    ax.set_xlim(4.1, 25.8)
    #ax.set_ylim(0, 6.5)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.savefig(f'./results/{params["name"]}/{params["name"]}.pdf', ppi=300, bbox_inches='tight');
    plt.close("all");


def plot_front_door(estimate, true_value, value_intervention, params, conditional=False):
    """ Utility to plot the front-door adjustment data.
    """
    if params["activation"]=="linear":
        label = "Linear"
    elif conditional:
        label = "Conditional"
    else:
        label = "Neural"
    
    # Plot of true and estimate (Linear, Neural, Conditional)
    ax = sns.kdeplot(estimate, label=f"{label} $do(X={value_intervention})$")
    ax = sns.kdeplot(true_value, label=f"True $do(X={value_intervention})$")

    plt.title(f"True vs. {label} $do(X={value_intervention})$", y=1.10)
    ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.10), borderaxespad=0, frameon=False)

    ax.text(0.5, 1,"WD: %.2f" % (scipy.stats.wasserstein_distance(true_value, estimate)), fontsize=11)

    ax.set_xlim(0,10)
    ax.set_ylim(0,1.2)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.savefig(f'./results/{params["name"]}/{params["name"]}'+f'_{label}'+
                    f'_{str(value_intervention).replace(".", "")}'+'.pdf', ppi=300, bbox_inches='tight');
    plt.close("all");


def bootstrap_plot(results, data, params):
    """ Selects the right true values for every type of experiment and plots the bootstrap estimates.
    """

    if params["model"] == "non_linear":
        confounder_linspace = np.linspace(5, 25, len(results["bootstrap_mean"]))
        true_value = (50/(3+confounder_linspace))
        plot_non_linear(results["bootstrap_mean"], true_value, confounder_linspace, data, params, 
                            bootstrap_bands=(results["bootstrap_lower"], results["bootstrap_upper"]))

    elif params["model"] == "mild_unobserved_confounder":
        confounder_linspace = np.linspace(5, 25, len(results["bootstrap_mean"]))
        true_value = (50/(3+confounder_linspace)) + 0.3
        plot_non_linear(results["bootstrap_mean"], true_value, confounder_linspace, data, params, 
                            bootstrap_bands=(results["bootstrap_lower"], results["bootstrap_upper"]))

    elif params["model"] == "strong_unobserved_confounder":
        confounder_linspace = np.linspace(5, 25, len(results["bootstrap_mean"]))
        true_value = (50/(3+confounder_linspace)) + 3.
        plot_non_linear(results["bootstrap_mean"], true_value, confounder_linspace, data, params, 
                            bootstrap_bands=(results["bootstrap_lower"], results["bootstrap_upper"]))

    elif params["model"] == "non_linear_unobserved_confounder":
        confounder_linspace = np.linspace(5, 25, len(results["bootstrap_mean"]))
        true_value = (50/(3+confounder_linspace))
        plot_non_linear(results["bootstrap_mean"], true_value, confounder_linspace, data, params, 
                            bootstrap_bands=(results["bootstrap_lower"], results["bootstrap_upper"]))


def plot_loss(loss, params, bootstrap_bands=None):
    """ Utility to plot the neural network loss.
    """
    ax = sns.lineplot(x=range(len(loss)), y=loss, label=f"Training loss")
    if bootstrap_bands is not None:
        ax.fill_between(x=range(len(loss)), y1=bootstrap_bands[0], y2=bootstrap_bands[1], alpha=.5)

    plt.title("Training loss", y=1.10)
    plt.xlabel("Iteration number")

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.savefig(f'./results/{params["name"]}/{params["name"]}_loss.pdf', ppi=300, bbox_inches='tight');
    plt.close("all");
