#!/usr/bin/env python3

import argparse
import os
import pathlib
import subprocess
import time

import openmc
from .plot import plot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--benchmarks', type=pathlib.Path,
                        default=pathlib.Path('benchmarks/lists/pst-short'),
                        help='List of benchmarks to run.')
    parser.add_argument('-c', '--cross-sections', type=str,
                        default=os.getenv('OPENMC_CROSS_SECTIONS'),
                        help='Cross sections XML file.')
    parser.add_argument('-p', '--particles', type=int, default=10000,
                        help='Number of source particles.')
    parser.add_argument('-b', '--batches', type=int, default=150,
                        help='Number of batches.')
    parser.add_argument('-i', '--inactive', type=int, default=50,
                        help='Number of inactive batches.')
    parser.add_argument('-m', '--max-batches', type=int, default=10000,
                        help='Maximum number of batches.')
    parser.add_argument('-t', '--threshold', type=float, default=0.0001,
                        help='Value of the standard deviation trigger on '
                        'eigenvalue.')
    args = parser.parse_args()

    current_time = time.localtime()
    timestamp = time.strftime("%Y-%m-%d-%H%M%S", current_time)

    # Create directory and set filename for results
    os.makedirs('results', exist_ok=True)
    outfile = f'results/{timestamp}.csv'

    # Get a copy of the benchmarks repository
    if not pathlib.Path('benchmarks').is_dir():
        repo = 'https://github.com/mit-crpg/benchmarks.git'
        subprocess.run(['git', 'clone', repo], check=True)

    # Get the list of benchmarks to run
    if not args.benchmarks.is_file():
        msg = 'Could not locate the benchmark list {}.'.format(args.benchmarks)
        raise ValueError(msg)
    with open(args.benchmarks) as f:
        benchmarks = f.read().split()

    # Prepare and run benchmarks
    for benchmark in benchmarks:
        openmc.reset_auto_ids()
        path = pathlib.Path('benchmarks') / benchmark

        # Remove old statepoint files
        for f in path.glob('statepoint.*.h5'):
            os.remove(f)

        # Modify settings
        settings = openmc.Settings.from_xml(path / 'settings.xml')
        settings.particles = args.particles
        settings.inactive = args.inactive
        settings.batches = args.batches
        settings.keff_trigger = {'type': 'std_dev',
                                 'threshold': args.threshold}
        settings.trigger_active = True
        settings.trigger_max_batches = args.max_batches
        settings.output = {'tallies': False}
        settings.export_to_xml(path)

        # Set path to cross sections XML
        materials = openmc.Materials.from_xml(path / 'materials.xml')
        materials.cross_sections = args.cross_sections
        materials.export_to_xml(path / 'materials.xml')

        # Create tallies.xml
        tally = openmc.Tally()
        tally.scores = ['total', 'absorption', 'fission']
        tallies = openmc.Tallies([tally])
        tallies.export_to_xml(path / 'tallies.xml')

        # Run benchmark
        openmc.run(cwd=path)

        # Get results
        filename = list(path.glob('statepoint.*.h5'))[0]
        with openmc.StatePoint(filename) as sp:
            mean = sp.k_combined.nominal_value
            stdev = sp.k_combined.std_dev

        # Write results
        words = benchmark.split('/')
        name = words[1]
        case = words[3] if len(words) > 3 else ''
        line = '{}, {}, {}, {}'.format(name, case, mean, stdev)
        if benchmark != benchmarks[-1]:
            line += '\n'
        with open(outfile, 'a') as f:
            f.write(line)

    plot(outfile)
