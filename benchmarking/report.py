from argparse import ArgumentParser
from fnmatch import fnmatch
from math import sqrt
import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
from openmc import __version__

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

def write_document(result, output, match=None):
    """Write LaTeX document section with preamble, run info, and table 
    entries for all benchmark data comparing the calculated and 
    experimental values along with uncertainties.

    Parameters
    ----------
    result : str
        Name of a result csv file produced by the benchmarking script.
    output    : str
        Name of the file to be written, ideally a .tex file
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

    # Convert from list to string
    result = result[0]

    label = Path(result).name
    
    # Read data from spreadsheet
    dataframes = {}
    dataframes[label] = get_result_dataframe(result).dropna()

    # Get model keff and uncertainty from ICSBEP
    icsbep = get_icsbep_dataframe()

    # Determine ICSBEP case names
    base = label
    index = dataframes[base].index

    df = dataframes[label]

    # Applying matching as needed
    if match is not None:
        cond = index.map(lambda x: fnmatch(x, match))
        index = index[cond]

    # Custom Table Description and Caption
    desc = ('Table \\ref{tab:1} uses (nuclear data info here) and openmc ' 
            f'version {__version__} to evaluate ICSBEP benchmarks.')
    caption = ('\\caption{\\label{tab:1} Criticality (' + label + ') Benchmark Results}\\\\')

    # Define Table Entry
    table = [
            desc,
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
    parser.add_argument('result', nargs='+', help='Result CSV file')
    parser.add_argument('--match', help='Pattern to match benchmark names to')
    parser.add_argument('-o', '--output', help='Filename to save to')
    args = parser.parse_args()
        
    write_document(
        args.result,
        args.output,
        args.match
        )
    
