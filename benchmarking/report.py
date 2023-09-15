from argparse import ArgumentParser
from fnmatch import fnmatch
from math import sqrt
import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np

from .results import get_result_dataframe, get_icsbep_dataframe, abbreviated_name


import pylatex as pl


def main():
    """Produce .tex file"""
    