# Validation

This repository contains a collection of validation scripts, notebooks, results, etc. that have been used in the preparation of reports and articles. The following scripts are currently available:

- `openmc-run-benchmarks` -- Run a collection of ICSBEP benchmarks with OpenMC or MCNP6
- `openmc-validate-neutron-physics` -- Used for validating neutron transport in OpenMC. The script generates an infinite medium problem of a single nuclide with a point source at a single energy and compares the resulting neutron energy spectra from OpenMC and either MCNP6 or Serpent 2. This script requires that you have MCNP6 or Serpent 2 installed (`mcnp6` or `sss2` executable available).
- `openmc-validate-photon-physics` -- Used for validating photon transport in OpenMC. The script generates an infinite medium problem of a single element with a point source at a single energy and compares the resulting photon energy spectra from OpenMC and either MCNP6 or Serpent 2. This script requires that you have MCNP6 or Serpent 2 installed (`mcnp6` or `sss2` executable available).
- `openmc-validate-photon-production` -- Used for validating photon production in OpenMC. This script generates a broomstick problem where monoenergetic neutrons are shot down a thin, infinitely long cylinder. The energy spectra of photons produced from neutron reactions is tallied along the surface of the cylinder. Comparisons are made between OpenMC and either MCNP6 or Serpent 2. Again, this script requires that MCNP6 or Serpent 2 is properly installed.

## Installation

The validation package can be installed via:

```bash
python setup.py install
```

## Prerequisites

OpenMC, NumPy, Matplotlib, h5py
