#!/usr/bin/env python3

import argparse
import os
import pathlib
import re
import shutil
import subprocess
import time

import openmc
from .plot import plot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--benchmarks', type=pathlib.Path,
                        default=pathlib.Path('benchmarks/lists/pst-short'),
                        help='List of benchmarks to run.')
    parser.add_argument('-c', '--code', choices=['openmc', 'mcnp'],
                        default='openmc',
                        help='Code used to run benchmarks.')
    parser.add_argument('-x', '--cross-sections', type=str,
                        default=os.getenv('OPENMC_CROSS_SECTIONS'),
                        help='OpenMC cross sections XML file.')
    parser.add_argument('-s', '--suffix', type=str, default='70c',
                        help='MCNP cross section suffix')
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
    parser.add_argument('-o', '--output-name', type=str,
                        help='Base filename for plot.')
    parser.add_argument('-f', '--output-format', type=str, default='png',
                        help='File format for plot.')
    args = parser.parse_args()

    # Create timestamp
    current_time = time.localtime()
    timestamp = time.strftime("%Y-%m-%d-%H%M%S", current_time)

    # Check that executable exists
    executable = 'mcnp6' if args.code == 'mcnp' else 'openmc'
    if not shutil.which(executable, os.X_OK):
        msg = f'Unable to locate executable {executable} in path.'
        raise IOError(msg)

    # Create directory and set filename for results
    os.makedirs('results', exist_ok=True)
    outfile = f'results/{timestamp}.csv'

    # Get a copy of the benchmarks repository
    if not pathlib.Path('benchmarks').is_dir():
        repo = 'https://github.com/mit-crpg/benchmarks.git'
        subprocess.run(['git', 'clone', repo], check=True)

    # Get the list of benchmarks to run
    if not args.benchmarks.is_file():
        msg = f'Unable to locate benchmark list {args.benchmarks}.'
        raise IOError(msg)
    with open(args.benchmarks) as f:
        benchmarks = f.read().split()

    # Prepare and run benchmarks
    for benchmark in benchmarks:
        path = pathlib.Path('benchmarks') / benchmark

        if args.code == 'openmc':
            openmc.reset_auto_ids()

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

            # Run benchmark
            openmc.run(cwd=path)

            # Read k-effective mean and standard deviation from statepoint
            filename = list(path.glob('statepoint.*.h5'))[0]
            with openmc.StatePoint(filename) as sp:
                mean = sp.k_combined.nominal_value
                stdev = sp.k_combined.std_dev

        elif args.code == 'mcnp':
            # Read input file
            with open(path / 'input', 'r') as f:
                lines = f.readlines()

            # Update criticality source card
            line = f'kcode {args.particles} 1 {args.inactive} {args.batches}\n'
            for i in range(len(lines)):
                if lines[i].strip().startswith('kcode'):
                    lines[i] = line
                    break

            # Update cross section suffix
            match = '(7[0-4]c)|(8[0-6]c)'
            if not re.match(match, args.suffix):
                msg = f'Unsupported cross section suffix {args.suffix}.'
                raise ValueError(msg)
            lines = [re.sub(match, args.suffix, x) for x in lines]

            # Write new input file
            with open(path / 'input', 'w') as f:
                f.write(''.join(lines))

            # Remove old MCNP output files
            for f in ('outp', 'runtpe', 'srctp'):
                try:
                    os.remove(path / f)
                except OSError:
                    pass

            # Run benchmark and capture and print output
            arg_list = [executable, 'inp=input']
            p = subprocess.Popen(
                args=arg_list, cwd=path, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, universal_newlines=True
            )
            while True:
                line = p.stdout.readline()
                if not line and p.poll() is not None:
                    break
                print(line, end='')

            # Read k-effective mean and standard deviation from output
            with open(path / 'outp', 'r') as f:
                line = f.readline()
                while not line.strip().startswith('col/abs/trk len'):
                    line = f.readline()
                words = line.split()
                mean = float(words[2])
                stdev = float(words[3])

        # Write results
        words = benchmark.split('/')
        name = words[1]
        case = words[3] if len(words) > 3 else ''
        line = '{}, {}, {}, {}'.format(name, case, mean, stdev)
        if benchmark != benchmarks[-1]:
            line += '\n'
        with open(outfile, 'a') as f:
            f.write(line)

    plot(outfile, output_name=args.output_name,
         output_format=args.output_format)
