import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200


data = []
columns = [
    r"Mean-degree of training data ($\alpha$)",
    "Accuracy (%)"
]

darkmode = False
add_legend = True

if darkmode:
    # Set dark background style
    plt.style.use('dark_background')

# Set font size
plt.rcParams.update({'font.size': 13})

# Fill this data in before running script
wanted = [9, 10, 11, 12, 14, 15, 16, 17]
numbers = [[], [], [], [], [], [], [], []]
for i, w in enumerate(wanted):
    for num in numbers[i]:
        data.append([w, num])

df = pd.DataFrame(data, columns=columns)
sns_plot = sns.boxplot(
    x=columns[0], y=columns[1], data=df, color='C0', showfliers=False,)
sns_plot.set(ylim=(45, 101))

# Add dividing line in centre
lower, upper = plt.gca().get_xlim()
midpoint = (lower + upper) / 2
plt.axvline(x=midpoint, color='w' if darkmode else 'black',
            linewidth=1.0, linestyle='--')

# Map range to numbers to be plotted
# Fill this data in before running script
baselines = []
targets_scaled = range(int((upper - lower)))
plt.plot(targets_scaled, baselines, color='C1', marker='x', linestyle='--')

# Plot numbers for threshold-based accuracy
# Fill this data in before running script
thresholds = [[], [], [], [], [], [], [], []]
means, errors = np.mean(thresholds, 1), np.std(thresholds, 1)
plt.errorbar(targets_scaled, means, yerr=errors, color='C2', linestyle='--')

if add_legend:
    # Custom legend
    meta_patch = mpatches.Patch(color='C0', label=r'$Acc_{meta-classifier}$')
    baseline_patch = mpatches.Patch(color='C1', label=r'$Acc_{loss test}$')
    threshold_patch = mpatches.Patch(color='C2', label=r'$Acc_{threshold test}$')
    plt.legend(handles=[meta_patch, baseline_patch,
                        threshold_patch], prop={'size': 13})

# Make sure axis label not cut off
plt.tight_layout()

sns_plot.figure.savefig("./meta_boxplot.png")
