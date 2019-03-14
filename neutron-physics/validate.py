#!/usr/bin/env python3

import argparse

import model


# Define command-line options
parser = argparse.ArgumentParser()
parser.add_argument('nuclide', type=str, help='Name of the nuclide, e.g. "U235"')
parser.add_argument('-d', '--density', type=float, default=1.,
                    help='Density of the material in g/cm^3')
parser.add_argument('-e', '--energy', type=float, default=1e6,
                    help='Energy of the source in eV')
parser.add_argument('-p', '--particles', type=int, default=100000,
                    help='Number of source particles')
parser.add_argument('-c', '--code', choices=['mcnp', 'serpent'], default='mcnp',
                    help='Code to validate OpenMC against.')
parser.add_argument('-s', '--suffix', type=str, default='70c',
                    help='MCNP cross section suffix')
parser.add_argument('-l', '--library', type=str,
                    help='Directory containing endf70[a-k] or endf71x MCNP ACE '
                    'data library. If specified, an HDF5 library that can be '
                    'used by OpenMC will be created from the MCNP data.')
parser.add_argument('-o', '--output-name', type=str, help='Name used for output.')
args = parser.parse_args()

m = model.Model(args.nuclide, args.density, args.energy, args.particles,
                args.code, args.suffix, args.library, args.output_name)
m.run()
