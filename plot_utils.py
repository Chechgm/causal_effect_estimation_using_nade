# ./plot_utils.py
""" Script where all the plot utilities are coded.

TODO: modify plot_unobserved_confounder_mild for its intended purpose
TODO: test the plot_front_door function

The available functions are:
- plot_non_linear
- plot_unobserved_confounder_mild
- plot_front_door
"""
import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_non_linear(estimate, data, params):
    """ Utility to plot neural vs. linear treatment effects
    """
    if params["activation"]=="linear":
        label = "Linear"
    else:
        label = "Neural"
    confounder_linspace = torch.arange(5, 25, 0.1).numpy()

    ax = sns.lineplot(x=confounder_linspace, y=estimate, label=f"{label} TE")
    ax = sns.lineplot(x=confounder_linspace, y=(50/(3+confounder_linspace)), label="True TE")
    ax = sns.histplot(x=data.ks_dataset[:,0]*data.sd[0], element='step', alpha=.5, color='silver', weights=0.008*np.ones(len(data.ks_dataset)), bins=35)

    plt.title("Comparison between true and \n estimated conditional Treatment Effects", y=1.10)
    ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.10), borderaxespad=0, frameon=False)

    ax.set_xlim(4.1, 25.8)
    ax.set_ylim(0, 6.5)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.savefig(f"./results/{params["name"]}/{params["name"]}.pdf", ppi=300, bbox_inches='tight');

def plot_unobserved_confounder_mild(linear_causal_effect, neural_causal_effect, data):
    """ Utility to plot neural vs. linear treatment effects
    """
    confounder_linspace = torch.arange(5, 25, 0.1).numpy()

    ax = sns.lineplot(x=confounder_linspace, y=linear_causal_effect, label="Linear TE")
    ax = sns.lineplot(x=confounder_linspace, y=neural_causal_effect, label="Neural TE")
    ax = sns.lineplot(x=confounder_linspace, y=(50/(3+confounder_linspace)), label="True TE")
    ax = sns.histplot(x=data.ks_dataset[:,0]*data.sd[0], element='step', alpha=.5, color='silver', weights=0.008*np.ones(len(data.ks_dataset)), bins=35)

    plt.title("Comparison between true and \n estimated conditional Treatment Effects", y=1.10)
    ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.10), borderaxespad=0, frameon=False)

    ax.set_xlim(4.1, 25.8)
    ax.set_ylim(0, 6.5)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.savefig("./results/linear_vs_non-linear_hist.pdf", ppi=300, bbox_inches='tight');


def plot_front_door(estimate, true_value, value_intervention, params):
    """ Utility to plot the front-door adjustment data.
    """
    if params["activation"]=="linear":
        label = "Linear"
    else:
        label = "Neural"
    
    # Plot of true and estimate (Linear, Neural, Conditional)
    ax = sns.distplot(estimate, label=f"{label} $do(X={value_intervention})$")
    ax = sns.distplot(true_value, label="True $do(X={value_intervention})$")

    plt.title(f"True vs. {label} $do(X={value_intervention})$", y=1.10)
    ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.10), borderaxespad=0, frameon=False)

    ax.text(0.5, 1,"WD: %.2f" % (scipy.stats.wasserstein_distance(true_value, estimate)), fontsize=11)

    ax.set_xlim(0,10)
    ax.set_ylim(0,1.2)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.savefig(f"./results/{params["name"]}/{params["name"]}.pdf", ppi=300, bbox_inches='tight');
