#!/usr/bin/env python3

import argparse

import model


# Define command-line options
parser = argparse.ArgumentParser()
parser.add_argument('element', type=str, help='Name of the element, e.g. "U"')
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
parser.add_argument('-k', '--photon-library', type=str,
                    help='Directory containing the MCNP ACE photon library '
                    'eprdata12. If specified, an HDF5 library that can be '
                    'used by OpenMC will be created for the given element.')
parser.add_argument('-o', '--output-name', type=str, help='Name used for output.')
args = parser.parse_args()

m = model.Model(args.element, args.density, [(args.element, 1.)], args.energy,
                args.particles, args.electron_treatment, args.photon_library,
                args.output_name)
m.run()
