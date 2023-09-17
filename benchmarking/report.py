from argparse import ArgumentParser
from fnmatch import fnmatch
from math import sqrt
import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np

from .results import get_result_dataframe, get_icsbep_dataframe, abbreviated_name

def num_format(number):
    """Format long decimal values with thin spaces.
    
    Parameters
    ----------
    number  : float
        Float value to be formatted for LaTeX
    
    Returns
    ----------
    tex_number   : str
        TeX-readable number with thin space
    
    """
    number = round(number, 6)
    number = str(number)
    decimal_index = number.find('.') 
    if decimal_index != -1:
        if len(number[decimal_index+4:]) > 0:
            tex_number = number[:decimal_index+4] + '\\,' +number[decimal_index+4:]
        else:
            tex_number = number
    return tex_number

def document(results, file, labels=None, match=None):
    """Fill LaTeX document section with run info and table entries for all 
    benchmark data comparing the calculated and experimental values along 
    with uncertainties.

    Parameters
    ----------
    results : iterable of str
        Name of a results file produced by the benchmarking script.
    file    : str
        Python variable assigned to 
    labels: iterable of str
        Labels for each dataset to use in legend
    match : str
        Pattern to match benchmark names to

    Returns
    -------
    None

    """

    #define document start and end
    document_start = ['\\begin{document}',
                '\\part*{Benchmark Results}']
    
    document_end = ['\\end{document}']

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

    # entries are abbreviated_name, column 1
    # experimental value, uncertainty, column 2,3
    # calculated value, uncertainty, column 4, 5
    
    # Setup x values (integers) and corresponding tick labels
    n = index.size # <-- number of cases we need to issue


    x = np.arange(1, n + 1)
    xticklabels = index.map(abbreviated_name)

    table = ['Table \\ref{tab:tomato} uses (nuclear data set info here) to evaluate ICSBEP benchmarks.',
                '\\begin{longtable}{lcccc}',
                '\\caption{\\label{tab:tomato} Criticality (nuclear data set info here) Benchmark Results}\\\\',
                '\\endfirsthead',
                '\\midrule',
                '\\multicolumn{5}{r}{Continued on Next Page}\\',
                '\\midrule',
                '\\endfoot',
                '\\bottomrule',
                '\\endlastfoot',
                '\\toprule',
                '& Exp. $k_{\\textrm{eff}}$&Exp. unc.& Calc. $k_{\textrm{eff}}$&Calc. unc.\\',
                '\\midrule',
                '\\end{longtable}']

    for i, (label, df) in enumerate(dataframes.items()):
        keff = 
        decimal_index = keff.find('.') 
        if decimal_index != -1:
            
        stdev =

        table.insert(-2, (f'{label}&{icsbep['keff']}&{icsbep['stdev']}&{keff}&{stdev}\\\\'))
        # Calculate keff C/E and its standard deviation
        coe = (df['keff'] / icsbep['keff']).loc[index] # <-- how to get both keff values
        stdev = 1.96 * df['stdev'].loc[index]

        # Plot keff C/E
        kwargs = {'color': f'C{i}', 'mec': 'black', 'mew': 0.15, 'label': label}
        if show_uncertainties:
            ax.errorbar(x, coe, yerr=stdev, fmt='o', **kwargs)
        else:
            ax.plot(x, coe, 'o', **kwargs)

        # Plot mean C/E
        if show_mean:
            mu = coe.mean()
            sigma = coe.std() / sqrt(n)
            if show_shaded:
                verts = [(0, mu - sigma), (0, mu + sigma), (n+1, mu + sigma), (n+1, mu - sigma)]
                poly = Polygon(verts, facecolor=f'C{i}', alpha=0.5)
                ax.add_patch(poly)
            else:
                ax.plot([-1, n], [mu, mu], '-', color=f'C{i}', lw=1.5)

        # Show shaded region of benchmark model uncertainties
        unc = icsbep['stdev'].loc[index] # <-- how to access iscbep stdev (will need to format)
        vert = np.block([[x, x[::-1]], [1 + unc, 1 - unc[::-1]]]).T
        poly = Polygon(vert, facecolor='gray', edgecolor=None, alpha=0.2)
        ax.add_patch(poly)
    

    

    
    
    
    tex.writelines(s + '\n' for s in document)

def main():
    """Produce LaTeX document with tabulated benchmark results"""

    parser = ArgumentParser()
    #parser.add_argument('files', nargs='+', help='Result CSV files')
    parser.add_argument('--labels', help='Comma-separated list of dataset labels')
    #parser.add_argument('--plot-type', choices=['keff', 'diff'], default='keff')
    parser.add_argument('--match', help='Pattern to match benchmark names to')
    #parser.add_argument('--show-mean', action='store_true', help='Show line/bar indicating mean')
    #parser.add_argument('--no-show-mean', dest='show_mean', action='store_false',
    #                    help='Do not show line/bar indicating mean')
    #parser.add_argument('--show-uncertainties', action='store_true',
    #                    help='Show uncertainty bars on individual cases')
    #parser.add_argument('--no-show-uncertainties', dest='show_uncertainties', action='store_false',
    #                    help='Do not show uncertainty bars on individual cases')
    #parser.add_argument('--show-shaded', action='store_true',
    #                    help='Show shaded region indicating uncertainty of mean C/E')
    #parser.add_argument('--no-show-shaded', dest='show_shaded', action='store_false',
    #                    help='Do not show shaded region indicating uncertainty of mean C/E')
    parser.add_argument('-o', '--output', help='Filename to save to')
    #parser.add_argument('-c', '--compile', help='Compile resulting .tex to pdf')
    parser.set_defaults(show_uncertainties=True, show_shaded=True, show_mean=True)
    args = parser.parse_args()
    
    if args.output is not None:
        tex = open(args.output, 'w')
    else:
        tex = open('report.tex', 'w')
    
    if args.labels is not None:
        args.labels = args.labels.split(',')



    
    
    # Testing purposes
    preamble = ['\\documentclass[12pt]{article}', 
                '\\usepackage[letterpaper, margin=1in]{geometry}',
                '\\usepackage{dcolumn}',
                '\\usepackage{tabularx}',
                '\\usepackage{booktabs}',
                '\\usepackage{longtable}',

                '\\usepackage{fancyhdr}',
                '\\setlength\\LTcapwidth{5.55in}',
                '\\setlength\\LTleft{0.5in}',
                '\\setlength\\LTright{0.5in}']
    
    tex.writelines(s + '\n' for s in preamble)
        

    



    tex.close()

