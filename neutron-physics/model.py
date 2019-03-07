#!/usr/bin/env python3

import os
from pathlib import Path
import re
import subprocess

from matplotlib import pyplot as plt
import numpy as np

import openmc


def zaid(nuclide, suffix):
    """ZA identifier of the nuclide

    Parameters
    ----------
    nuclide : str
        Name of the nuclide
    suffix : str
        Cross section suffix for MCNP

    Returns
    -------
    str
        ZA identifier

    """
    Z, A, m = openmc.data.zam(nuclide)

    # Correct the ground state and first excited state of Am242, which are
    # the reverse of the convention
    if A == 242 and m == 0:
        m = 1
    elif A == 242 and m == 1:
        m = 0

    if m > 0:
        A += 300 + 100*m

    if re.match('(71[0-6]nc)', suffix):
        suffix = f'8{suffix[2]}c'

    return f'{1000*Z + A}.{suffix}'


def szax(nuclide, suffix):
    """SZA identifier of the nuclide

    Parameters
    ----------
    nuclide : str
        Name of the nuclide
    suffix : str
        Cross section suffix for MCNP

    Returns
    -------
    str
        SZA identifier

    """
    Z, A, m = openmc.data.zam(nuclide)

    # Correct the ground state and first excited state of Am242, which are
    # the reverse of the convention
    if A == 242 and m == 0:
        m = 1
    elif A == 242 and m == 1:
        m = 0

    if re.match('(7[0-4]c)|(8[0-6]c)', suffix):
        suffix = f'71{suffix[1]}nc'

    return f'{1000000*m + 1000*Z + A}.{suffix}'


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
    particles : int
        Number of source particles.
    suffix : str
        Cross section suffix for MCNP
    library : str
        Directory containing endf70[a-k] or endf71x MCNP ACE data library. If
        specified, an HDF5 library that can be used by OpenMC will be created
        from the MCNP data.
    name : str
        Name used for output.

    Attributes
    ----------
    nuclide : str
        Name of the nuclide
    density : float
        Density of the material in g/cm^3.
    energy : float
        Energy of the source (eV)
    particles : int
        Number of source particles.
    suffix : str
        Cross section suffix for MCNP
    library : str
        Directory containing endf70[a-k] or endf71x MCNP ACE data library. If
        specified, an HDF5 library that can be used by OpenMC will be created
        from the MCNP data.
    name : str
        Name used for output.
    energy_mev : float
        Energy of the source (MeV)
    temperature : float
        Temperature (Kelvin) of the cross section data

    """

    def __init__(self, nuclide, density, energy, particles, suffix,
                 library=None, name=None):
        self.nuclide = nuclide
        self.density = density
        self.energy = energy
        self.particles = particles
        if not re.match('(7[0-4]c)|(8[0-6]c)|(71[0-6]nc)', suffix):
            msg = f'Unsupported MCNP cross section suffix {suffix}.'
            raise ValueError(msg)
        self.suffix = suffix
        if library is not None:
            self.library = Path(library)
            if not self.library.is_dir():
                msg = f'{self.library} is not a directory.'
                raise ValueError(msg)
        else:
            self.library = library
        self.name = name
        self._temperature = None

    @property
    def energy_mev(self):
        return self.energy*1.e-6

    def _create_library(self):
        """Convert the ACE data from the MCNP distribution into an HDF5 library
        that can be used by OpenMC.

        """
        if re.match('7[0-4]c', self.suffix):
            # Get the table from the ENDF/B-VII.0 neutron ACE files
            name = zaid(self.nuclide, self.suffix)
            for path in self.library.glob('endf70[a-k]'):
                try:
                    table = openmc.data.ace.get_table(path, name)
                except ValueError:
                    pass
        else:
            # Get the table from the ENDF/B-VII.1 neutron ACE files
            Z, A, m = openmc.data.zam(self.nuclide)
            element = openmc.data.ATOMIC_SYMBOL[Z]
            name = szax(self.nuclide, self.suffix)
            path = self.library / 'endf71x' / f'{element}' / f'{name}'
            table = openmc.data.ace.get_table(path, name)

        # Convert cross section data
        data = openmc.data.IncidentNeutron.from_ace(table, 'mcnp')
        self._temperature = data.kTs[0] / openmc.data.K_BOLTZMANN

        # Export HDF5 file
        os.makedirs('openmc', exist_ok=True)
        h5_file = Path('openmc') / f'{data.name}.h5'
        data.export_to_hdf5(h5_file, 'w')

        # Register with library and write cross_sections.xml
        data_lib = openmc.data.DataLibrary()
        data_lib.register_file(h5_file)
        data_lib.export_to_xml(Path('openmc') / 'cross_sections.xml')

    def _make_openmc_input(self):
        """Generate the OpenMC input XML

        """
        # Directory from which openmc is run
        os.makedirs('openmc', exist_ok=True)
        
        # Define material
        mat = openmc.Material()
        mat.add_nuclide(self.nuclide, 1.0)
        mat.set_density('g/cm3', self.density)
        materials = openmc.Materials([mat])
        if self.library is not None:
            xs_path = (Path('openmc') / 'cross_sections.xml').resolve()
            materials.cross_sections = str(xs_path)
        materials.export_to_xml(Path('openmc') / 'materials.xml')

        # Set up geometry
        sphere = openmc.Sphere(boundary_type='reflective', R=1.e9)
        cell = openmc.Cell(fill=materials, region=-sphere)
        geometry = openmc.Geometry([cell])
        geometry.export_to_xml(Path('openmc') / 'geometry.xml')

        # Define source
        source = openmc.Source()
        source.space = openmc.stats.Point((0,0,0))
        source.angle = openmc.stats.Isotropic()
        source.energy = openmc.stats.Discrete([self.energy], [1.])

        # Settings
        settings = openmc.Settings()
        if self._temperature is not None:
            settings.temperature = {'default': self._temperature}
        settings.source = source
        settings.particles = self.particles
        settings.run_mode = 'fixed source'
        settings.batches = 1
        settings.create_fission_neutrons = False
        settings.export_to_xml(Path('openmc') / 'settings.xml')
 
        # Define tallies
        energy_bins = np.logspace(-5, np.log10(self.energy), 500)
        energy_filter = openmc.EnergyFilter(energy_bins)
        tally = openmc.Tally(name='neutron flux')
        tally.filters = [energy_filter]
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
        if self._temperature is not None:
            kT = self._temperature * openmc.data.K_BOLTZMANN * 1e-6
            lines.append(f'1 1 -{self.density} -1 imp:n=1 tmp={kT}')
        else:
            lines.append(f'1 1 -{self.density} -1 imp:n=1')
        lines.append('2 0 1 imp:n=0')
 
        # Create the surface cards: sphere centered on origin with 1e9 cm
        # radius and reflective boundary conditions
        lines.append('')
        lines.append('c --- Surface cards ---')
        lines.append('*1 so 1.0e9')
 
        # Create the data cards
        lines.append('')
        lines.append('c --- Data cards ---')
 
        # Materials
        if re.match('(71[0-6]nc)', self.suffix):
            name = szax(self.nuclide, self.suffix)
        else:
            name = zaid(self.nuclide, self.suffix)
        material_card = f'm1 {name} 1.0'
        lines.append(material_card)
        lines.append('nonu 2')

        # Physics: neutron transport
        lines.append('mode n')
 
        # Source definition: isotropic point source at center of sphere
        lines.append(f'sdef cel=1 erg={self.energy_mev}')
 
        # Tallies: neutron flux over cell
        lines.append('f4:n 1')
        lines.append(f'e4 1.e-11 498ilog {self.energy_mev}')
 
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
            t = sp.get_tally(name='neutron flux')
            x_openmc = t.find_filter(openmc.EnergyFilter).bins[:,1]
            y_openmc = t.mean[:,0,0]
 
        # Read the results from the MCNP output file
        with open(Path('mcnp') / 'outp', 'r') as f:
            text = f.read()
            p = text.find('1tally')
            p = text.find('energy', p) + 10
            q = text.find('total', p)
            t = np.fromiter(text[p:q].split(), float)
            t.shape = (len(t) // 3, 3)
            x_mcnp = t[1:,0] * 1.e6
            y_mcnp = t[1:,1]
            sd = t[1:,2]
 
        # Normalize the spectra
        y_openmc /= np.diff(np.insert(x_openmc, 0, 1.e-5))*sum(y_openmc)
        y_mcnp /= np.diff(np.insert(x_mcnp, 0, 1.e-5))*sum(y_mcnp)
 
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
        ax1.set_xlim([1.e-5, self.energy])
        ax1.set_xlabel('Energy (eV)', size=12)
        ax1.set_ylabel('Spectrum', size=12)
        ax1.legend()
        ax2.set_ylabel("Relative error", size=12)
        title = f'{self.nuclide}, {self.energy:.1e} eV Source'
        plt.title(title)
 
        # Save plot
        os.makedirs('plots', exist_ok=True)
        if self.name is not None:
            name = self.name
        else:
            name = f'{self.nuclide}-{self.energy:.1e}eV'
            if self._temperature is not None:
                name +=  f'-{self._temperature:.1f}K'
        plt.savefig(Path('plots') / (name + '.png'), bbox_inches='tight')
        plt.close()

    def run(self):
        """Generate inputs, run problem, and plot results.
 
        """
        if self.library is not None:
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
