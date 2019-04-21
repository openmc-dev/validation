#!/usr/bin/env python3

import argparse

from validate_physics.photon_physics import PhotonPhysicsModel


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
parser.add_argument('-c', '--code', choices=['mcnp', 'serpent'], default='mcnp',
                    help='Code to validate OpenMC against.')
parser.add_argument('-s', '--suffix', default='12p',
                    help='Photon cross section suffix')
parser.add_argument('-x', '--xsdir', type=str, help='XSDIR directory file. '
                    'If specified, it will be used to locate the ACE table '
                    'corresponding to the given element and suffix, and an '
                    'HDF5 library that can be used by OpenMC will be created '
                    'from the data.')
parser.add_argument('-g', '--serpent_pdata', type=str, help='Directory '
                    'containing the additional data files needed for photon '
                    'physics in Serpent.')
parser.add_argument('-o', '--output-name', type=str, help='Name used for output.')
args = parser.parse_args()

model = PhotonPhysicsModel(
    args.element, args.density, [(args.element, 1.)], args.energy,
    args.particles, args.electron_treatment, args.code, args.suffix,
    args.xsdir, args.serpent_pdata, args.output_name
)

model.run()
