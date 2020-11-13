import csv
from pathlib import Path

import pandas as pd


def abbreviated_name(name):
    """Return short name for ICSBEP benchmark cases

    Parameters
    ----------
    name : str
        ICSBEP benchmark name, e.g. "pu-met-fast-021/case-2"

    Returns
    -------
    str
        Abbreviated name, e.g. "pmf21-2"

    """
    model, *case = name.split('/')
    volume, form, spectrum, number = model.split('-')
    abbreviation = volume[0] + form[0] + spectrum[0]
    if case:
        casenum = case[0].replace('case', '')
    else:
        casenum = ''
    return f'{abbreviation}{int(number)}{casenum}'


def get_result_dataframe(filename):
    """Read the data from a file produced by the benchmarking script.

    Parameters
    ----------
    filename : str
        Name of a results file produced by the benchmarking script.

    Returns
    -------
    pandas.DataFrame
        Dataframe with 'keff' and 'stdev' columns. The benchmark name is used as
        the index in the dataframe.

    """
    return pd.read_csv(
        filename,
        header=None,
        names=['name', 'keff', 'stdev'],
        usecols=[0, 1, 2],
        index_col="name",
    )


def get_icsbep_dataframe():
    """Read the benchmark model k-effective means and uncertainties.

    Returns
    -------
    pandas.DataFrame
        Dataframe with 'keff' and 'stdev' columns. The benchmark name is used as
        the index in the dataframe.

    """
    cwd = Path(__file__).parent
    index = []
    keff = []
    stdev = []
    with open(cwd / 'uncertainties.csv', 'r') as csvfile:
        reader = csv.reader(csvfile, skipinitialspace=True)
        for benchmark, case, mean, uncertainty in reader:
            index.append(f'{benchmark}/{case}' if case else benchmark)
            keff.append(float(mean))
            stdev.append(float(uncertainty))
    return pd.DataFrame({'keff': keff, 'stdev': stdev}, index=index)
