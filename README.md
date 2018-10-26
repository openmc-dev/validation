# Validation

This repository contains a collection of validation scripts, notebooks, results, etc. that have been used in the preparation of reports and articles. The following directories are currently available:

- **photon-physics** -- A set of scripts for validating photon transport in OpenMC. The scripts generate an infinite medium problem of a single element with a point source at a single energy and compare the resulting photon energy spectra from both OpenMC and MCNP6. These scripts require that you have MCNP6 installed (`mcnp6` executable available) with cross sections set up appropriately. Namely, the MCNP runs require the eprdata12 library.
- **photon-production** -- A set of scripts for validating photon production in OpenMC. The scripts generate a broomstick problem where monoenergetic neutrons are shot down a thin, infinitely long cylinder. The energy spectra of photons produced from neutron reactions is tallied along the surface of the cylinder. Comparisons are made between OpenMC and MCNP6. Again, these scripts require that MCNP6 is properly installed.
