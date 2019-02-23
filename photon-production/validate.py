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
parser.add_argument('-p', '--particles', type=int, default=1000000,
                    help='Number of source particles')
parser.add_argument('-t', '--electron-treatment', choices=('ttb', 'led'),
                    default='ttb', help='Whether to use local energy'
                    'deposition or thick-target bremsstrahlung treatment '
                    'for electrons and positrons.')
parser.add_argument('-s', '--suffix', default='70c',
                    help='MCNP cross section suffix')
parser.add_argument('-l', '--library', type=str,
                    help='Directory containing endf70[a-k] or endf71x MCNP ACE '
                    'data library. If specified, an HDF5 library that can be '
                    'used by OpenMC will be created from the MCNP data.')
parser.add_argument('-k', '--photon-library', type=str,
                    help='Directory containing the MCNP ACE photon library '
                    'eprdata12. If specified, an HDF5 library that can be '
                    'used by OpenMC will be created.')
parser.add_argument('-o', '--output-name', type=str, help='Name used for output.')
args = parser.parse_args()

m = model.Model(args.nuclide, args.density, [(args.nuclide, 1.)],
                args.energy, args.particles, args.electron_treatment,
                args.suffix, args.library, args.photon_library, args.output_name)
m.run()
