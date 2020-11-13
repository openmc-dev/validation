import pandas as pd


def short_name(name):
    model, *case = name.split('/')
    volume, form, spectrum, number = model.split('-')
    abbreviation = volume[0] + form[0] + spectrum[0]
    if case:
        casenum = case[0].replace('case', '')
    else:
        casenum = ''
    return f'{abbreviation}{int(number)}{casenum}'


def get_result_dataframe(filename):
    return pd.read_csv(
        filename,
        header=None,
        names=['name', 'keff', 'stdev'],
        usecols=[0, 1, 2],
        index_col="name",
    )
