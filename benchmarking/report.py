from argparse import ArgumentParser
from fnmatch import fnmatch
from math import sqrt
import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np

from .results import get_result_dataframe, get_icsbep_dataframe, abbreviated_name

def num_format(num):
    """Format long decimal values with thin spaces.
    
    Parameters
    ----------
    num  : float
        Float value to be formatted for LaTeX

    Returns
    ----------
    tex_num   : str
        TeX-readable num with thin space
    
    """
    num = '{0:.9f}'.format(num)
    dec_pos= num.find('.') 
    if dec_pos != -1:
        if len(num[dec_pos+4:]) > 0:
            tex_number = (num[:dec_pos+4] + '\\,' + num[dec_pos+4:dec_pos+7])
    else:
        tex_num = num
    return num

def write_document(results, output, labels=None, match=None):
    """Write LaTeX document section with preamble, run info, and table entries 
    for all benchmark data comparing the calculated and experimental values 
    along with uncertainties.

    Parameters
    ----------
    results : iterable of str
        Name of a results file produced by the benchmarking script.
    output    : str
        Name of the file to be written, ideally a .tex file
    labels: iterable of str
        Labels for each dataset to use in legend
    match : str
        Pattern to match benchmark names to

    Returns
    -------
    None

    """
    #write file
    if output is None:
        output = 'report.tex'
    tex = open(output, 'w')

    #define document preamble
    preamble = [
                '\\documentclass[12pt]{article}', 
                '\\usepackage[letterpaper, margin=1in]{geometry}',
                '\\usepackage{dcolumn}',
                '\\usepackage{tabularx}',
                '\\usepackage{booktabs}',
                '\\usepackage{longtable}',
                '\\usepackage{fancyhdr}',
                '\\setlength\\LTcapwidth{5.55in}',
                '\\setlength\\LTleft{0.5in}',
                '\\setlength\\LTright{0.5in}'
                ]

    #define document start and end tex
    doc_start = [
                '\\begin{document}',
                '\\part*{Benchmark Results}'
                ]
    
    doc_end = ['\\end{document}']

    if labels is None:
        labels = [Path(f).name for f in results]

    # Read data from spreadsheets
    dataframes = {}
    for csvfile, label in zip(results, labels):
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

    table = [
            'Table \\ref{tab:data} uses (nuclear data set info here) to evaluate ICSBEP benchmarks.',
            '\\begin{longtable}{lcccc}',
            '\\caption{\\label{tab:data} Criticality (nuclear data set info here) Benchmark Results}\\\\',
            '\\endfirsthead',
            '\\midrule',
            '\\multicolumn{5}{r}{Continued on Next Page}\\',
            '\\midrule',
            '\\endfoot',
            '\\bottomrule',
            '\\endlastfoot',
            '\\toprule',
            '& Exp. $k_{\\textrm{eff}}$&Exp. unc.& Calc. $k_{\textrm{eff}}$&Calc. unc.\\\\',
            '\\midrule',
            '\\end{longtable}'
            ]

    for i, (label, df) in enumerate(dataframes.items()):
        keff = df['keff'] #calculated
        keff = num_format(keff)
        stdev = df['stdev'] #calculated?
        stdev = num_format(stdev)

        icsbep_keff = icsbep['keff']
        icsbep_stdev = icsbep['stdev']

        table.insert(-1, (f'{index}&{icsbep_keff}&{icsbep_stdev}&{keff}&{stdev}\\\\'))

    
    tex.writelines(line + '\n' for line in (preamble + doc_start + table + doc_end))
    tex.close()

def main():
    """Produce LaTeX document with tabulated benchmark results"""

    parser = ArgumentParser()
    parser.add_argument('results', nargs='+', help='Result CSV files')
    parser.add_argument('--labels', help='Comma-separated list of dataset labels')
    parser.add_argument('--match', help='Pattern to match benchmark names to')
    parser.add_argument('-o', '--output', help='Filename to save to')
    parser.add_argument('-c', '--compile', help='Compile resulting .tex to pdf')
    parser.set_defaults(show_uncertainties=True, show_shaded=True, show_mean=True)
    args = parser.parse_args()
    
    if args.labels is not None:
        args.labels = args.labels.split(',')

        
    write_document(
        args.results,
        args.output,
        args.labels,
        args.match
        )
    
    if args.compile:
        pass
