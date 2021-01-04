# ./plot_utils.py
""" Script where all the plot utilities are coded.

The available functions are:
- plot_non_linear
"""
import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_non_linear(linear_causal_effect, neural_causal_effect, data):
    """ Helper function to plot neural vs. linear treatment effects
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
