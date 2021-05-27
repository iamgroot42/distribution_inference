import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import argparse
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
    args = parser.parse_args()

    first_cat = " 0.5"

    if args.darkplot:
        # Set dark background style
        plt.style.use('dark_background')

    # Set font size
    plt.rcParams.update({'font.size': 18})

    data = []
    columns = [
        r'Female proportion of training data ($\alpha$)',
        "Accuracy (%)"
    ]

    categories = ["0.2", "0.3", "0.4", "0.6", "0.7", "0.8"]

    # Fill this data in before running script
    raw_data = [[], [], [], [], [], []]

    for i in range(len(raw_data)):
        for j in range(len(raw_data[i])):
            data.append([categories[i], raw_data[i][j]])

    df = pd.DataFrame(data, columns=columns)
    sns_plot = sns.boxplot(
        x=columns[0], y=columns[1], data=df, color='C0', showfliers=False,)

    if args.novtitle:
        plt.ylabel("", labelpad=0)

    # Accuracy range, with space to show good performance
    sns_plot.set(ylim=(45, 101))

    # Add dividing line in centre
    lower, upper = plt.gca().get_xlim()
    midpoint = (lower + upper) / 2
    plt.axvline(x=midpoint, color='black',
                linewidth=1.0, linestyle='--')

    # Fill this data in before running script
    # Map range to numbers to be plotted
    baselines = []
    targets_scaled = range(int((upper - lower)))
    plt.plot(targets_scaled, baselines, color='C1', marker='x', linestyle='--')

    # Fill this data in before running script
    # Plot numbers for threshold-based accuracy
    thresholds = [[], [], [], [], [], []]
    means, errors = np.mean(thresholds, 1), np.std(thresholds, 1)
    plt.errorbar(targets_scaled, means, yerr=errors, color='C2', linestyle='--')

    if args.legend:
        # Custom legend
        meta_patch = mpatches.Patch(color='C0', label=r'$Acc_{meta-classifier}$')
        baseline_patch = mpatches.Patch(color='C1', label=r'$Acc_{baseline}$')
        threshold_patch = mpatches.Patch(color='C2', label=r'$Acc_{threshold}$')
        plt.legend(handles=[meta_patch, baseline_patch, threshold_patch])

    # Make sure axis label not cut off
    plt.tight_layout()

    sns_plot.figure.savefig("./meta_boxplot.png")
