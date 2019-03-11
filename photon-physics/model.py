#!/usr/bin/env python3

import os
from pathlib import Path
import re
import subprocess

import h5py
from matplotlib import pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline

import openmc


class Model(object):
    """Monoenergetic, isotropic point source in an infinite geometry.

    Parameters
    ----------
    material : str
        Name of the material.
    density : float
        Density of the material in g/cm^3.
    elements : list of tuple
        List in which each item is a 2-tuple consisting of an element string and
        the atom fraction.
    energy : float
        Energy of the source (eV)
    particles : int
        Number of source particles.
    electron_treatment : {'led' or 'ttb'}
        Whether to deposit electron energy locally ('led') or create secondary
        bremsstrahlung photons ('ttb').
    photon_library : str
        Directory containing the MCNP ACE photon library eprdata12. If
        specified, an HDF5 library that can be used by OpenMC will be created.
    name : str
        Name used for output.

    Attributes
    ----------
    material : str
        Name of the material.
    density : float
        Density of the material in g/cm^3.
    elements : list of tuple
        List in which each item is a 2-tuple consisting of an element string and
        the atom fraction.
    energy : float
        Energy of the source (eV)
    particles : int
        Number of source particles.
    electron_treatment : {'led' or 'ttb'}
        Whether to deposit electron energy locally ('led') or create secondary
        bremsstrahlung photons ('ttb').
    photon_library : str
        Directory containing the MCNP ACE photon library eprdata12. If
        specified, an HDF5 library that can be used by OpenMC will be created.
    name : str
        Name used for output.
    energy_mev : float
        Energy of the source (MeV)

    """

    def __init__(self, material, density, elements, energy, particles,
                 electron_treatment, photon_library=None, name=None):
        self.material = material
        self.density = density
        self.elements = elements
        self.energy = energy
        self.particles = particles
        self.electron_treatment = electron_treatment
        if photon_library is not None:
            self.photon_library = Path(photon_library)
            if not self.photon_library.is_dir():
                msg = f'{self.library} is not a directory.'
                raise ValueError(msg)
        else:
            self.photon_library = photon_library
        self.name = name

    @property
    def energy_mev(self):
        return self.energy*1.e-6

    def _create_library(self):
        """Convert the ACE data from the MCNP distribution into an HDF5 library
        that can be used by OpenMC.

        """
        data_lib = openmc.data.DataLibrary()

        for element, fraction in self.elements:
            Z = openmc.data.ATOMIC_NUMBER[element]

            # Get the table from the photon ACE library
            path = self.photon_library / 'eprdata12'
            table = openmc.data.ace.get_table(path, f'{Z}000.12p')

            # Convert cross section data
            data = openmc.data.IncidentPhoton.from_ace(table)

            # Export HDF5 file
            os.makedirs('openmc', exist_ok=True)
            h5_file = Path('openmc') / f'{data.name}.h5'
            data.export_to_hdf5(h5_file, 'w')

            # Register with library
            data_lib.register_file(h5_file)

            # TODO: Currently the neutron libraries are still read in to
            # OpenMC even when doing pure photon transport, so we need to
            # locate them and register them with the library.
            path = os.getenv('OPENMC_CROSS_SECTIONS')
            lib = openmc.data.DataLibrary.from_xml(path)
            element = openmc.Element(element)
            for nuclide, _, _ in element.expand(fraction, 'ao'):
                h5_file = lib.get_by_material(nuclide)['path']
                data_lib.register_file(h5_file)

        data_lib.export_to_xml(Path('openmc') / 'cross_sections.xml')

    def _make_openmc_input(self):
        """Generate the OpenMC input XML

        """
        # Directory from which openmc is run
        os.makedirs('openmc', exist_ok=True)
        
        # Define material
        mat = openmc.Material()
        for element, fraction in self.elements:
            mat.add_element(element, fraction)
        mat.set_density('g/cm3', self.density)
        materials = openmc.Materials([mat])
        if self.photon_library is not None:
            xs_path = (Path('openmc') / 'cross_sections.xml').resolve()
            materials.cross_sections = str(xs_path)
        materials.export_to_xml(Path('openmc') / 'materials.xml')

        # Set up geometry
        sphere = openmc.Sphere(boundary_type='reflective', r=1.e9)
        cell = openmc.Cell()
        cell.fill = mat
        cell.region = -sphere
        geometry = openmc.Geometry([cell])
        geometry.export_to_xml(Path('openmc') / 'geometry.xml')

        # Define source
        source = openmc.Source()
        source.space = openmc.stats.Point((0,0,0))
        source.angle = openmc.stats.Isotropic()
        source.energy = openmc.stats.Discrete([self.energy], [1.])
        source.particle = 'photon'

        # Settings
        settings = openmc.Settings()
        settings.source = source
        settings.particles = self.particles
        settings.run_mode = 'fixed source'
        settings.batches = 1
        settings.photon_transport = True
        settings.electron_treatment = self.electron_treatment
        settings.cutoff = {'energy_photon' : 1000.}
        settings.export_to_xml(Path('openmc') / 'settings.xml')
 
        # Define tallies
        cell_filter = openmc.CellFilter(cell)
        energy_bins = np.logspace(3, np.log10(self.energy), 500)
        energy_filter = openmc.EnergyFilter(energy_bins)
        particle_filter = openmc.ParticleFilter('photon')
        tally = openmc.Tally(name='photon flux')
        tally.filters = [cell_filter, energy_filter, particle_filter]
        tally.scores = ['flux']
        tallies = openmc.Tallies([tally])
        tallies.export_to_xml(Path('openmc') / 'tallies.xml')

    def _make_mcnp_input(self):
        """Generate the MCNP input file

        """
        # Directory from which MCNP will be run
        os.makedirs('mcnp', exist_ok=True)

        # Create the problem description
        lines = ['Point source in infinite geometry']
 
        # Create the cell cards: material 1 inside sphere, void outside
        lines.append('c --- Cell cards ---')
        lines.append(f'1 1 -{self.density} -1 imp:p=1')
        lines.append('2 0 1 imp:p=0')
 
        # Create the surface cards: sphere centered on origin with 1e9 cm
        # radius and  reflective boundary conditions
        lines.append('')
        lines.append('c --- Surface cards ---')
        lines.append('*1 so 1.0e9')
 
        # Create the data cards
        lines.append('')
        lines.append('c --- Data cards ---')
 
        # Materials
        material_card = 'm1'
        for element, fraction in self.elements:
            Z = openmc.data.ATOMIC_NUMBER[element]
            material_card += f' {Z}000.12p -{fraction}'
        lines.append(material_card)

        # Physics: photon transport, 1 keV photon cutoff energy
        if self.electron_treatment == 'led':
            flag = 1
        else:
            flag = 'j'
        lines.append('mode p')
        lines.append(f'phys:p j {flag} j j j')
        lines.append('cut:p j 1.e-3')
 
        # Source definition: isotropic point source at center of sphere
        lines.append(f'sdef cel=1 erg={self.energy_mev}')
 
        # Tallies: photon flux over cell
        lines.append('f4:p 1')
        lines.append(f'e4 1.e-3 498ilog {self.energy_mev}')
 
        # Problem termination: number of particles to transport
        lines.append(f'nps {self.particles}')
 
        # Write the problem
        with open(Path('mcnp') / 'inp', 'w') as f:
            f.write('\n'.join(lines))

    def _plot(self):
        """Extract and plot the results
 
        """
        # Read the results from the OpenMC statepoint
        with openmc.StatePoint(Path('openmc') / 'statepoint.1.h5') as sp:
            t = sp.get_tally(name='photon flux')
            x_openmc = t.find_filter(openmc.EnergyFilter).bins[:,1]*1.e-6
            y_openmc = t.mean[:,0,0]
 
        # Read the results from the MCNP output file
        with open(Path('mcnp') / 'outp', 'r') as f:
            text = f.read()
            p = text.find('1tally')
            p = text.find('energy', p) + 10
            q = text.find('total', p)
            t = np.fromiter(text[p:q].split(), float)
            t.shape = (len(t) // 3, 3)
            x_mcnp = t[1:,0]
            y_mcnp = t[1:,1]
            sd = t[1:,2]
 
        # Normalize the spectra
        y_openmc /= np.diff(np.insert(x_openmc, 0, 1.e-3))*sum(y_openmc)
        y_mcnp /= np.diff(np.insert(x_mcnp, 0, 1.e-3))*sum(y_mcnp)
 
        # Compute the relative error
        err = np.zeros_like(y_mcnp)
        idx = np.where(y_mcnp > 0)
        err[idx] = (y_openmc[idx] - y_mcnp[idx])/y_mcnp[idx]
 
        # Set up the figure
        fig = plt.figure(1, facecolor='w', figsize=(8,8))
        ax1 = fig.add_subplot(111)
 
        # Create a second y-axis that shares the same x-axis, keeping the first
        # axis in front
        ax2 = ax1.twinx()
        ax1.set_zorder(ax2.get_zorder() + 1)
        ax1.patch.set_visible(False)
 
        # Plot the spectra
        ax1.loglog(x_mcnp, y_mcnp, 'r', linewidth=1, label='MCNP')
        ax1.loglog(x_openmc, y_openmc, 'b', linewidth=1, label='OpenMC', linestyle='--')
 
        # Plot the relative error and uncertainties
        ax2.semilogx(x_mcnp, err, color=(0.2, 0.8, 0.0), linewidth=1)
        ax2.semilogx(x_mcnp, 2*sd, color='k', linestyle='--', linewidth=1)
        ax2.semilogx(x_mcnp, -2*sd, color='k', linestyle='--', linewidth=1)
 
        # Set grid and tick marks
        ax1.tick_params(axis='both', which='both', direction='in', length=10)
        ax1.grid(b=False, axis='both', which='both')
        ax2.tick_params(axis='y', which='both', right=False)
        ax2.grid(b=True, which='both', axis='both', alpha=0.5, linestyle='--')
 
        # Set axes labels and limits
        ax1.set_xlim([1.e-3, self.energy_mev])
        ax1.set_xlabel('Energy (MeV)', size=12)
        ax1.set_ylabel('Spectrum', size=12)
        ax1.legend()
        ax2.set_ylabel("Relative error", size=12)
        title = f'{self.material}, {self.energy_mev:.1e} MeV Source'
        plt.title(title)
 
        # Save plot
        os.makedirs('plots', exist_ok=True)
        if self.name is not None:
            name = self.name
        else:
            name = f'{self.material}-{self.energy_mev:.1e}MeV'
        plt.savefig(Path('plots') / f'{name}.png', bbox_inches='tight')
        plt.close()

    def run(self):
        """Generate inputs, run problem, and plot results.
 
        """
        if self.photon_library is not None:
            self._create_library()

        self._make_openmc_input()
        self._make_mcnp_input()

        openmc.run(cwd='openmc')
 
        # Remove old MCNP output files
        for f in ('outp', 'runtpe'):
            try:
                os.remove(Path('mcnp') / f)
            except OSError:
                pass
 
        # Run MCNP and capture and print output
        p = subprocess.Popen('mcnp6', cwd='mcnp', stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, universal_newlines=True)
        while True:
            line = p.stdout.readline()
            if not line and p.poll() is not None:
                break
            print(line, end='')

        self._plot()
