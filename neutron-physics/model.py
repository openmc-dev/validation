#!/usr/bin/env python3

import os
import subprocess

from matplotlib import pyplot as plt
import numpy as np

import openmc
from openmc.data import ATOMIC_NUMBER


class Model(object):
    """Monoenergetic, isotropic point source in an infinite geometry.

    Parameters
    ----------
    nuclide : str
        Name of the nuclide
    density : float
        Density of the material in g/cm^3.
    energy : float
        Energy of the source (eV)
    suffix : str
        Cross section suffix for MCNP
    particles : int
        Number of source particles.

    Attributes
    ----------
    energy_mev : float
        Energy of the source (MeV)

    """

    def __init__(self, nuclide, density, energy, suffix, particles=1000000):
        self.nuclide = nuclide
        self.density = density
        self.energy = energy
        self.suffix = suffix
        self.particles = particles

    @property
    def energy_mev(self):
        return self.energy*1.e-6

    def _build_openmc(self):
        """Generate the OpenMC input XML

        """
        # Directory from which openmc is run
        os.makedirs('openmc', exist_ok=True)
        
        # Define material
        mat = openmc.Material()
        mat.add_nuclide(self.nuclide, 1.0)
        mat.set_density('g/cm3', self.density)
        materials = openmc.Materials([mat])
        materials.export_to_xml(os.path.join('openmc', 'materials.xml'))

        # Set up geometry
        sphere = openmc.Sphere(boundary_type='reflective', R=1.e9)
        cell = openmc.Cell(fill=materials, region=-sphere)
        geometry = openmc.Geometry([cell])
        geometry.export_to_xml(os.path.join('openmc', 'geometry.xml'))

        # Define source
        source = openmc.Source()
        source.space = openmc.stats.Point((0,0,0))
        source.angle = openmc.stats.Isotropic()
        source.energy = openmc.stats.Discrete([self.energy], [1.])

        # Settings
        settings = openmc.Settings()
        settings.source = source
        settings.particles = self.particles
        settings.run_mode = 'fixed source'
        settings.batches = 1
        settings.export_to_xml(os.path.join('openmc', 'settings.xml'))
 
        # Define tallies
        energy_bins = np.logspace(3, np.log10(self.energy), 500)
        energy_filter = openmc.EnergyFilter(energy_bins)
        tally = openmc.Tally(name='neutron flux')
        tally.filters = [energy_filter]
        tally.scores = ['flux']
        tallies = openmc.Tallies([tally])
        tallies.export_to_xml(os.path.join('openmc', 'tallies.xml'))

    def _build_mcnp(self):
        """Generate the MCNP input file

        """
        # Directory from which MCNP will be run
        os.makedirs('mcnp', exist_ok=True)

        # Create the problem description
        lines = ['Point source in infinite geometry']
 
        # Create the cell cards: material 1 inside sphere, void outside
        lines.append('c --- Cell cards ---')
        lines.append('1 1 -{} -1 imp:n=1'.format(self.density))
        lines.append('2 0 1 imp:n=0')
 
        # Create the surface cards: sphere centered on origin with 1e9 cm
        # radius and  reflective boundary conditions
        lines.append('')
        lines.append('c --- Surface cards ---')
        lines.append('*1 so 1.0e9')
 
        # Create the data cards
        lines.append('')
        lines.append('c --- Data cards ---')
 
        # Materials
        Z, A, m = openmc.data.zam(self.nuclide)
        if m > 0:
            A += 400
        material_card = f'm1 {Z}{A:03}.{self.suffix} 1.0'
        lines.append(material_card)

        # Physics: neutron transport
        lines.append('mode n')
 
        # Source definition: isotropic point source at center of sphere
        lines.append('sdef cel=1 erg={}'.format(self.energy_mev))
 
        # Tallies: neutron flux over cell
        lines.append('f4:n 1')
        lines.append('e4 1.e-3 498ilog {}'.format(self.energy_mev))
 
        # Problem termination: number of particles to transport
        lines.append('nps {}'.format(self.particles))
 
        # Write the problem
        with open(os.path.join('mcnp', 'inp'), 'w') as f:
            f.write('\n'.join(lines))

    def _plot(self):
        """Extract and plot the results
 
        """
        # Read the results from the OpenMC statepoint
        with openmc.StatePoint(os.path.join('openmc', 'statepoint.1.h5')) as sp:
            t = sp.get_tally(name='neutron flux')
            x_openmc = t.find_filter(openmc.EnergyFilter).bins[:,1]*1.e-6
            y_openmc = t.mean[:,0,0]
 
        # Read the results from the MCNP output file
        with open(os.path.join('mcnp', 'outp'), 'r') as f:
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
        title = f'{self.nuclide}, {self.energy_mev} MeV Source'
        plt.title(title)
 
        # Save plot
        os.makedirs('plots', exist_ok=True)
        name = f'{self.nuclide}-{self.energy_mev}MeV.png'
        plt.savefig(os.path.join('plots', name), bbox_inches='tight')
        plt.close()

    def run(self):
        """Generate inputs, run problem, and plot results.
 
        """
        # Generate inputs
        self._build_openmc()
        self._build_mcnp()

        # Run Openmc
        openmc.run(cwd='openmc')
 
        # Remove old MCNP output files
        for f in ('outp', 'runtpe'):
            try:
                os.remove(os.path.join('mcnp', f))
            except OSError:
                pass
 
        # Run MCNP
        p = subprocess.Popen('mcnp6', cwd='mcnp', stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, universal_newlines=True)
 
        # Capture and print MCNP output
        while True:
            line = p.stdout.readline()
            if not line and p.poll() is not None:
                break
            print(line, end='')

        # Plot results
        self._plot()
