from argparse import ArgumentParser
from math import sqrt
import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np

from .results import get_result_dataframe, get_icsbep_dataframe, abbreviated_name


def plot(files, labels=None, plot_type='keff', match=None, show_shaded=True,
         show_uncertainties=True):
    """For all benchmark cases, produce a plot comparing the k-effective mean
    from the calculation to the experimental value along with uncertainties.

    Parameters
    ----------
    files : iterable of str
        Name of a results file produced by the benchmarking script.
    labels: iterable of str
        Labels for each dataset to use in legend
    plot_type : {'keff', 'diff'}
        Type of plot to produce. A 'keff' plot shows the ratio of the
        calculation k-effective mean to the experimental value (C/E). A 'diff'
        plot shows the difference between C/E values for different
        calculations. Default is 'keff'.
    match : str
        Pattern to match benchmark names to
    show_shaded : bool
        Whether to show shaded region indicating uncertainty of mean C/E
    show_uncertainties : bool
        Whether to show uncertainties for individual cases

    Returns
    -------
    matplotlib.axes.Axes
        A matplotlib.axes.Axes object, or None if 'save' is True.

    """
    if labels is None:
        labels = [Path(f).name for f in files]

    # Read data from spreadsheets
    dataframes = {}
    for csvfile, label in zip(files, labels):
        dataframes[label] = get_result_dataframe(csvfile).dropna()

    # Get model keff and uncertainty from ICSBEP
    icsbep = get_icsbep_dataframe()

    # Determine common benchmarks
    base = labels[0]
    index = dataframes[base].index
    for df in dataframes.values():
        index = index.intersection(df.index)

    # Applying matching as needed
    if match is not None:
        cond = index.map(lambda x: fnmatch(x, match))
        index = index[cond]

    # Setup x values (integers) and corresponding tick labels
    n = index.size
    x = np.arange(1, n + 1)
    xticklabels = index.map(abbreviated_name)

    fig, ax = plt.subplots(figsize=(17, 6))

    if plot_type == 'diff':
        # Check that two results files are specified
        if len(files) < 2:
            raise ValueError('Must provide two or more files to create a "diff" plot')

        kwargs = {'mec': 'black', 'mew': 0.15, 'fmt': 'o'}

        keff0 = dataframes[base]['keff'].loc[index]
        stdev0 = 1.96*dataframes[base]['stdev'].loc[index]
        for i, label in enumerate(labels[1:]):
            df = dataframes[label]
            keff_i = df['keff'].loc[index]
            stdev_i = 1.96*df['stdev'].loc[index]

            diff = keff_i - keff0
            err = np.sqrt(stdev_i**2 + stdev0**2)
            kwargs['label'] = labels[i + 1] + ' - ' + labels[0]
            if show_uncertainties:
                ax.errorbar(x, diff, yerr=err, color=f'C{i}', **kwargs)
            else:
                ax.plot(x, diff, color=f'C{i}', **kwargs)

            mu = diff.mean()
            if show_shaded:
                sigma = diff.std() / sqrt(n)
                verts = [(0, mu - sigma), (0, mu + sigma), (n+1, mu + sigma), (n+1, mu - sigma)]
                poly = Polygon(verts, facecolor=f'C{i}', alpha=0.5)
                ax.add_patch(poly)
            else:
                ax.plot([-1, n], [mu, mu], '-', color=f'C{i}', lw=1.5)

        # Define y-axis label
        ylabel = r'$\Delta k_\mathrm{eff}$'

    else:
        for i, (label, df) in enumerate(dataframes.items()):
            # Calculate keff C/E and its standard deviation
            coe = (df['keff'] / icsbep['keff']).loc[index]
            stdev = 1.96 * df['stdev'].loc[index]

            # Plot keff C/E
            kwargs = {'color': f'C{i}', 'mec': 'black', 'mew': 0.15, 'label': label}
            if show_uncertainties:
                ax.errorbar(x, coe, yerr=stdev, fmt='o', **kwargs)
            else:
                ax.plot(x, coe, 'o', **kwargs)

            # Plot mean C/E
            mu = coe.mean()
            sigma = coe.std() / sqrt(n)
            if show_shaded:
                verts = [(0, mu - sigma), (0, mu + sigma), (n+1, mu + sigma), (n+1, mu - sigma)]
                poly = Polygon(verts, facecolor=f'C{i}', alpha=0.5)
                ax.add_patch(poly)
            else:
                ax.plot([-1, n], [mu, mu], '-', color=f'C{i}', lw=1.5)

        # Show shaded region of benchmark model uncertainties
        unc = icsbep['stdev'].loc[index]
        vert = np.block([[x, x[::-1]], [1 + unc, 1 - unc[::-1]]]).T
        poly = Polygon(vert, facecolor='gray', edgecolor=None, alpha=0.2)
        ax.add_patch(poly)

        # Define axes labels and title
        ylabel = r'$k_\mathrm{eff}$ C/E'

    # Configure plot
    ax.set_axisbelow(True)
    ax.set_xlim((0, n+1))
    ax.set_xticks(x)
    ax.set_xticklabels(xticklabels, rotation='vertical')
    ax.tick_params(axis='x', which='major', labelsize=10)
    ax.tick_params(axis='y', which='major', labelsize=14)
    ax.set_xlabel('Benchmark case', fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.grid(True, which='both', color='lightgray', ls='-', alpha=0.7)
    ax.legend(numpoints=1)
    return ax


def main():
    parser = ArgumentParser()
    parser.add_argument('files', nargs='+')
    parser.add_argument('--labels')
    parser.add_argument('--plot-type', choices=['keff', 'diff'], default='keff')
    args = parser.parse_args()

    if args.labels is not None:
        args.labels = args.labels.split(',')

    ax = plot(args.files, labels=args.labels, plot_type=args.plot_type)
    plt.show()
