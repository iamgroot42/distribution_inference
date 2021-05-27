import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import os
from utils import flash_utils
import numpy as np
from model_utils import BASE_MODELS_DIR
from data_utils import PROPERTY_FOCUS, SUPPORTED_PROPERTIES
import matplotlib.patches as mpatches
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--darkplot', action="store_true",
                        help='Use dark background for plotting results')
    parser.add_argument('--legend', action="store_true",
                        help='Add legend to plots')
    parser.add_argument('--novtitle', action="store_true",
                        help='Remove Y-axis label')
    parser.add_argument('--filter', choices=SUPPORTED_PROPERTIES,
                        help='name for subfolder to save/load data from')
    args = parser.parse_args()
    flash_utils(args)

    first_cat = " 0.5"

    # Set font size
    plt.rcParams.update({'font.size': 18})

    if args.darkplot:
        # Set dark background
        plt.style.use('dark_background')

    data = []
    columns = [
        r'%s proportion of training data ($\alpha$)' % PROPERTY_FOCUS[args.filter],
        "Accuracy (%)"
    ]

    batch_size = 1000
    num_train = 700
    n_tries = 5

    train_dir_1 = os.path.join(BASE_MODELS_DIR, "victim/%s/" % first_cat)
    test_dir_1 = os.path.join(BASE_MODELS_DIR, "adv/%s/" % first_cat)

    targets = ["0.0", "0.1", "0.2", "0.3",
               "0.4", "0.6", "0.7", "0.8", "0.9", "1.0"]

    # Fill this data in before running script
    if args.filter == "sex":
        raw_data = [[], [], [], [], [], [], [], [], [], []]
    else:
        raw_data = [[], [], [], [], [], [], [], [], [], []]

    for i in range(len(raw_data)):
        for j in range(len(raw_data[i])):
            data.append([float(targets[i]), raw_data[i][j]])

    df = pd.DataFrame(data, columns=columns)
    sns_plot = sns.boxplot(
        x=columns[0], y=columns[1], data=df, color='C0', showfliers=False,)

    # Accuracy range, with space to show good performance
    sns_plot.set(ylim=(45, 101))

    # Add dividing line in centre
    lower, upper = plt.gca().get_xlim()
    midpoint = (lower + upper) / 2
    plt.axvline(x=midpoint, color='black',
                linewidth=1.0, linestyle='--')

    # Map range to numbers to be plotted
    # Fill this data in before running script
    if args.filter == 'race':
        # For race
        baselines = []
        thresholds = [[], [], [], [], [], [], [], [], [], []]
    else:
        # For sex
        baselines = []
        thresholds = [[], [], [], [], [], [], [], [], [], []]

    # Plot baselines
    targets_scaled = range(int((upper - lower)))
    plt.plot(targets_scaled, baselines, color='C1', marker='x', linestyle='--')

    # Plot numbers for threshold-based accuracy
    means, errors = np.mean(thresholds, 1), np.std(thresholds, 1)
    plt.errorbar(targets_scaled, means, yerr=errors, color='C2', linestyle='--')

    # Custom legend
    if args.legend:
        meta_patch = mpatches.Patch(color='C0', label=r'$Acc_{meta-classifier}$')
        baseline_patch = mpatches.Patch(color='C1', label=r'$Acc_{baseline}$')
        threshold_patch = mpatches.Patch(color='C2', label=r'$Acc_{threshold}$')
        plt.legend(handles=[meta_patch, baseline_patch, threshold_patch])

    if args.novtitle:
        plt.ylabel("", labelpad=0)

    # Make sure axis label not cut off
    plt.tight_layout()

    # Save plot
    sns_plot.figure.savefig("./meta_boxplot.png")
