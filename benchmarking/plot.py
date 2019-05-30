#!/usr/bin/env python3

import csv
import os
import pathlib
import time

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np

mpl.rcParams['ps.useafm'] = True
mpl.rcParams['pdf.use14corefonts'] = True
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [
    r'\usepackage[bitstream-charter]{mathdesign}',
    r'\usepackage{amsmath}',
    r'\usepackage[usenames]{xcolor}'
]
mpl.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})


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


def plot(filename):
    """For each benchmark case, plot the ratio of the k-effective mean from the
    calculation to the experimental value along with uncertainties.

    Parameters
    ----------
    filename : str
        Name of the results file produced by the benchmarking script

    """
    timestamp = pathlib.Path(filename).stem

    # Get the benchmark names, k-effective mean, and k-effective uncertainty
    name, case = np.loadtxt(filename, dtype=str, delimiter=', ',
                            usecols=(0, 1), unpack=True)
    coe, stdev = np.loadtxt(filename, dtype=float, delimiter=', ',
                            usecols=(2, 3), unpack=True)

    # 95% confidence interval
    stdev *= 1.96

    # x values for plot (benchmark number)
    n = len(name)
    x = np.arange(1, n+1)

    # Get the benchmark model k-effective means and uncertainties
    model_keff = read_uncertainties()

    labels = []
    unc = np.empty(n)
    for i in range(n):
        # Get the abbreviated benchmark names for x axis tick labels
        volume, form, spectrum, number = name[i].split('-')
        abbreviation = volume[0] + form[0] + spectrum[0]
        short_name = f'{abbreviation}{number}{case[i].replace("case", "")}'
        labels.append(short_name)

        # Calculate C/E and get the benchmark model uncertainties
        benchmark = f'{name[i]}/{case[i]}' if case[i] else f'{name[i]}'
        if benchmark in model_keff:
            coe[i] /= model_keff[benchmark][0]
            unc[i] = model_keff[benchmark][1]

    # Plot mean C/E
    mu = np.mean(coe)
    label = f'Average C/E = {mu:.4f}'
    plt.plot([0, n+1], [mu, mu], '-', color='#3F5D7D80', lw=1.5, label=label)

    # Plot k-effective C/E
    kwargs = {'color': '#3F5D7D', 'mec': 'black', 'mew': 0.15}
    plt.errorbar(x, coe, yerr=stdev, fmt='o', **kwargs)

    # Show shaded region of benchmark model uncertainties
    verts = np.block([[x, x[::-1]], [1 + unc, 1 - unc[::-1]]]).T
    poly = Polygon(verts, facecolor='gray', edgecolor=None, alpha=0.2)
    ax = plt.gca()
    ax.add_patch(poly)

    # Configure plot
    plt.xticks(range(1, n+1), labels, rotation='vertical')
    plt.xlim((0, n+1))
    plt.subplots_adjust(bottom=0.15)
    plt.setp(ax.get_xticklabels(), fontsize=10)
    plt.setp(ax.get_yticklabels(), fontsize=14)
    plt.gcf().set_size_inches(17, 6)
    plt.xlabel('Benchmark case', fontsize=18)
    plt.ylabel(r'$k_{\text{eff}}$ C/E', fontsize=18)
    plt.grid(True, which='both', color='lightgray', ls='-', alpha=0.7)
    plt.gca().set_axisbelow(True)
    title = time.asctime(time.strptime(timestamp, "%Y-%m-%d-%H%M%S"))
    plt.title(title, multialignment='left')
    lgd = plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), numpoints=1)

    # Save plot
    os.makedirs('results', exist_ok=True)
    plt.savefig(f'results/{timestamp}.png', bbox_inches='tight')
