#!/usr/bin/env python3

import argparse

import model


# Define command-line options
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--material', type=str, default='U235',
                    help='Name of the nuclide, e.g. "U235"')
parser.add_argument('-d', '--density', type=float, default=1.,
                    help='Density of the material in g/cm^3')
parser.add_argument('-e', '--energy', type=float, default=1e6,
                    help='Energy of the source in eV')
parser.add_argument('-t', '--electron-treatment', choices=('ttb', 'led'),
                    default='ttb', help='Whether to use local energy'
                    'deposition or thick-target bremsstrahlung treatment '
                    'for electrons and positrons.')
parser.add_argument('-n', '--particles', type=int, default=1000000,
                    help='Number of source particles')
args = parser.parse_args()

m = model.Model(args.material, args.density, [(args.material, 1.)],
                args.energy, args.electron_treatment, args.particles)
m.run()
