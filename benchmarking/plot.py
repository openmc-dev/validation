#!/usr/bin/env python3

import csv
import os
import pathlib
import time

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np


def read_uncertainties():
    """Read the benchmark model k-effective means and uncertainties.

    Returns
    -------
    dict of str to tuple of float
        Dictionary whose keys are the benchmark model names and values are
        the k-effective mean and uncertainty

    """
    cwd = pathlib.Path(__file__).parent
    model_keff = {}
    with open(cwd / 'uncertainties.csv', 'r') as csvfile:
        reader = csv.reader(csvfile, skipinitialspace=True)
        for benchmark, case, mean, uncertainty in reader:
            if case:
                name = '{}/{}'.format(benchmark, case)
            else:
                name = benchmark
            model_keff[name] = (float(mean), float(uncertainty))
    return model_keff


def read_results(filename):
    """Read the data from a file produced by the benchmarking script..

    Parameters
    ----------
    filename : str
        Name of a results file produced by the benchmarking script.

    Returns
    -------
    dict
        Dictionary with keys 'coe' (numpy.ndarray of the ratio of the mean
        k-effective values from the calculation to the experimental values),
        'std' (numpy.ndarray of the k-effective standard deviations from the
        calculation), 'unc' (numpy.ndarray of the benchmark model
        uncertainties), and 'label' (list of abbreviated benchmark names).

    """
    # Get the benchmark model k-effective means and uncertainties
    model_keff = read_uncertainties()

    # Read the calculation results
    text = np.loadtxt(filename, dtype=str, delimiter=', ', unpack=True)
    n = text.shape[1]
    name = text[0]
    case = text[1]
    data = {'coe': text[2].astype(float), 'std': text[3].astype(float)*1.96,
            'unc': np.empty(n), 'label': []}

    for i in range(n):
        # Get the abbreviated benchmark names for x-axis tick labels
        volume, form, spectrum, number = name[i].split('-')
        abbreviation = volume[0] + form[0] + spectrum[0]
        short_name = f'{abbreviation}{number}{case[i].replace("case", "")}'
        data['label'].append(short_name)

        # Calculate C/E and get the benchmark model uncertainties
        benchmark = f'{name[i]}/{case[i]}' if case[i] else f'{name[i]}'
        if benchmark in model_keff:
            data['coe'][i] /= model_keff[benchmark][0]
            data['unc'][i] = model_keff[benchmark][1]
    return data


def plot(f1, f2=None, plot_type='keff', output_name=None, output_format='png',
         save=True):
    """For all benchmark cases, produce a plot comparing the k-effective mean
    from the calculation to the experimental value along with uncertainties.

    Parameters
    ----------
    f1 : str
        Name of a results file produced by the benchmarking script.
    f2 : str
        Name of a results file produced by the benchmarking script. This file
        will only be used for a 'diff' plot; for a 'keff' plot, it will be
        ignored.
    plot_type : {'keff', 'diff'}
        Type of plot to produce. A 'keff' plot shows the ratio of the
        calculation k-effective mean to the experimental value (C/E). A 'diff'
        plot shows the difference between C/E values for different
        calculations. Default is 'keff'.
    output_name : str
        The output file name.
    output_format : str
        The output file format, e.g. 'png', 'pdf', ... Default is 'png'.
    save : bool
        If True, the figure will be saved. If False, it will be displayed.

    """
    # Get the data from the results files
    f1 = pathlib.Path(f1)
    r1 = read_results(f1)

    # x values for plot (benchmark number)
    n = len(r1['coe'])
    x = np.arange(1, n+1)

    if plot_type == 'diff':
        # Check that two results files are specified
        if f2 is None:
            msg = ('Unable to create a "diff" plot since only one filename '
                   f'{f1} was provided.')
            raise ValueError(msg)

        # Get the data from the results file
        f2 = pathlib.Path(f2)
        r2 = read_results(f2)

        # Check that number of benchmarks in both files is the same
        if len(r2['coe']) != n:
            msg = f'{f1} and {f2} have different number of benchmarks'
            raise ValueError(msg)

        # Plot mean difference
        mu = sum(r2['coe'] - r1['coe'])/n
        label = f'Average C/E difference = {mu:.4f}'
        plt.plot([0,n+1], [mu, mu], '-', color='#3F5D7D', ls='--', label=label)

        # Plot C/E difference
        kwargs = {'color': '#3F5D7D', 'mec': 'black', 'mew': 0.15}
        err = abs(r2['coe']/r1['coe']) * np.sqrt((r2['std']/r2['coe'])**2 +
                                                 (r1['std']/r2['coe'])**2)
        plt.errorbar(x, r2['coe'] - r1['coe'], yerr=err, fmt='o', **kwargs)

        # Define axes labels and title
        xlabel = 'Benchmark case'
        ylabel = r'$k_{\mathrm{eff}}$ C/E difference'
        f1_time = time.asctime(time.strptime(f1.stem, "%Y-%m-%d-%H%M%S"))
        f2_time = time.asctime(time.strptime(f2.stem, "%Y-%m-%d-%H%M%S"))
        title = f'{f2_time} - {f1_time}'

    else:
        # Show shaded region of benchmark model uncertainties
        vert = np.block([[x, x[::-1]], [1 + r1['unc'], 1 - r1['unc'][::-1]]]).T
        poly = Polygon(vert, facecolor='gray', edgecolor=None, alpha=0.2)
        ax = plt.gca()
        ax.add_patch(poly)

        # Plot mean C/E
        mu = np.mean(r1['coe'])
        label = f'Average C/E = {mu:.4f}'
        plt.plot([0, n+1], [mu, mu], '-', color='#3F5D7D', ls='--', label=label)

        # Plot k-effective C/E
        kwargs = {'color': '#3F5D7D', 'mec': 'black', 'mew': 0.15}
        plt.errorbar(x, r1['coe'], yerr=r1['std'], fmt='o', **kwargs)

        # Define axes labels and title
        xlabel = 'Benchmark case'
        ylabel = r'$k_{\mathrm{eff}}$ C/E'
        title = time.asctime(time.strptime(f1.stem, "%Y-%m-%d-%H%M%S"))

    # Configure plot
    ax = plt.gca()
    ax.set_axisbelow(True)
    plt.setp(ax.get_xticklabels(), fontsize=10)
    plt.setp(ax.get_yticklabels(), fontsize=14)
    plt.xlim((0, n+1))
    plt.xticks(range(1, n+1), r1['label'], rotation='vertical')
    plt.subplots_adjust(bottom=0.15)
    plt.gcf().set_size_inches(17, 6)
    plt.grid(True, which='both', color='lightgray', ls='-', alpha=0.7)
    plt.xlabel(xlabel, fontsize=18)
    plt.ylabel(ylabel, fontsize=18)
    plt.title(title, multialignment='left')
    lgd = plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), numpoints=1)

    # Save or display plot
    if save:
        if output_name is None:
            output_name = f1.stem
            if plot_type == 'diff':
                output_name += f'_{f2.stem}_diff'
            else:
                output_name += '_keff'
        outfile = f1.parent / f'{output_name}.{output_format}'
        plt.savefig(outfile, bbox_inches='tight')
    else:
        plt.show()

    plt.close()
