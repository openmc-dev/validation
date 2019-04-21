#!/usr/bin/env python3

import argparse

from validate_physics.photon_production import PhotonProductionModel


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
parser.add_argument('-c', '--code', choices=['mcnp', 'serpent'], default='mcnp',
                    help='Code to validate OpenMC against.')
parser.add_argument('-s', '--suffix', default='70c',
                    help='Neutron cross section suffix')
parser.add_argument('-k', '--photon-suffix', default='12p',
                    help='Photon cross section suffix')
parser.add_argument('-x', '--xsdir', type=str, help='XSDIR directory file. '
                    'If specified, it will be used to locate the ACE table '
                    'corresponding to the given nuclide and suffix, and an '
                    'HDF5 library that can be used by OpenMC will be created '
                    'from the data.')
parser.add_argument('-g', '--serpent_pdata', type=str, help='Directory '
                    'containing the additional data files needed for photon '
                    'physics in Serpent.')
parser.add_argument('-o', '--output-name', type=str, help='Name used for output.')
args = parser.parse_args()

model = PhotonProductionModel(
    args.nuclide, args.density, [(args.nuclide, 1.)], args.energy,
    args.particles, args.electron_treatment, args.code, args.suffix,
    args.photon_suffix, args.xsdir, args.serpent_pdata, args.output_name
)

model.run()
