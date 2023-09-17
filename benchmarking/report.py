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
            tex_num = (num[:dec_pos+4] + '\\,' + num[dec_pos+4:dec_pos+7])
    else:
        tex_num = num
    return tex_num

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
    # Open and write file
    if output is None:
        output = 'report.tex'
    tex = open(output, 'w')

    # Define document preamble
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

    # Define document start and end snippets

    doc_start = ['\\begin{document}', '\\part*{Benchmark Results}']
    
    doc_end = ['\\end{document}']

    if labels is None:
        labels = [Path(f).name for f in results]
    
    # Read data from spreadsheets
    dataframes = {}
    for csvfile, label in zip(results, labels):
        dataframes[label] = get_result_dataframe(csvfile).dropna() 
        #fills dataframe with a key (label) and value (entire results df)

    # Get model keff and uncertainty from ICSBEP
    icsbep = get_icsbep_dataframe()

    # Determine common benchmarks
    base = labels[0]
    index = dataframes[base].index #get raw icsbep case names 

    for df in dataframes.values(): #get the case/value column of the dataframes
        index = index.intersection(df.index) #gives all of the cases that need to be displayed

    # Applying matching as needed (DEPRECATED)
    if match is not None:
        cond = index.map(lambda x: fnmatch(x, match))
        index = index[cond]

    # Custom Table Caption
    caption = '\\caption{\\label{tab:1} Criticality (' + labels[0] + ') Benchmark Results}\\\\'
    # Define Table Entry
    table = [
            'Table \\ref{tab:1} uses (nuclear data set info here) to evaluate ICSBEP benchmarks.',
            '\\begin{longtable}{lcccc}',
            caption,
            '\\endfirsthead',
            '\\midrule',
            '\\multicolumn{5}{r}{Continued on Next Page}\\\\',
            '\\midrule',
            '\\endfoot',
            '\\bottomrule',
            '\\endlastfoot',
            '\\toprule',
            '& Exp. $k_{\\textrm{eff}}$&Exp. unc.& Calc. $k_{\\textrm{eff}}$&Calc. unc.\\\\',
            '\\midrule',
            '% DATA',

            '\\end{longtable}'
            ]

    for case in index:
        # Obtain and format calculated values
        keff = df['keff'].loc[case]
        keff = num_format(keff)
        
        stdev = df['stdev'].loc[case]
        stdev = num_format(stdev)
        
        # Obtain and format experimental values
        icsbep_keff = '{0:.4f}'.format(icsbep['keff'].loc[case])
        icsbep_stdev = '{0:.4f}'.format(icsbep['stdev'].loc[case])
        
        # Insert data values into table as separate entries
        table.insert(-1, (f'{case}&{icsbep_keff}&{icsbep_stdev}&{keff}&{stdev}\\\\'))

    # Write all accumulated lines
    tex.writelines(line + '\n' for line in (preamble + doc_start + table + doc_end))
    tex.close()
    return

def main():
    """Produce LaTeX document with tabulated benchmark results"""

    parser = ArgumentParser()
    parser.add_argument('results', nargs='+', help='Result CSV file')
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
    
    # TBD
    if args.compile:
        pass
