#!/usr/bin/env python3

import os
from pathlib import Path
import re
import shutil
import subprocess

from matplotlib import pyplot as plt
import numpy as np

import openmc
from openmc.data import ATOMIC_NUMBER, NEUTRON_MASS, K_BOLTZMANN
from .utils import create_library, read_results


class PhotonPhysicsModel(object):
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
    code : {'mcnp', 'serpent'}
        Code to validate against
    suffix : str
        Photon cross section suffix
    xsdir : str
        XSDIR directory file. If specified, it will be used to locate the ACE
        table corresponding to the given element and suffix, and an HDF5
        library that can be used by OpenMC will be created from the data.
    serpent_pdata : str
        Directory containing the additional data files needed for photon
        physics in Serpent.
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
    code : {'mcnp', 'serpent'}
        Code to validate against
    suffix : str
        Photon cross section suffix
    xsdir : str
        XSDIR directory file. If specified, it will be used to locate the ACE
        table corresponding to the given element and suffix, and an HDF5
        library that can be used by OpenMC will be created from the data.
    serpent_pdata : str
        Directory containing the additional data files needed for photon
        physics in Serpent.
    name : str
        Name used for output.
    bins : int
        Number of bins in the energy grid
    batches : int
        Number of batches to simulate
    cutoff_energy: float
        Photon cutoff energy (eV)
    openmc_dir : pathlib.Path
        Working directory for OpenMC
    other_dir : pathlib.Path
        Working directory for MCNP or Serpent
    table_names : list of str
        Names of the ACE tables used in the model

    """

    def __init__(self, material, density, elements, energy, particles,
                 electron_treatment, code, suffix, xsdir=None,
                 serpent_pdata=None, name=None):
        self._bins = 500
        self._batches = 100
        self._cutoff_energy = 1.e3
        self._openmc_dir = None
        self._other_dir = None

        self.material = material
        self.density = density
        self.elements = elements
        self.energy = energy
        self.particles = particles
        self.electron_treatment = electron_treatment
        self.code = code
        self.suffix = suffix
        self.xsdir = xsdir
        self.serpent_pdata = serpent_pdata
        self.name = name

    @property
    def energy(self):
        return self._energy

    @property
    def particles(self):
        return self._particles

    @property
    def code(self):
        return self._code

    @property
    def suffix(self):
        return self._suffix

    @property
    def xsdir(self):
        return self._xsdir

    @property
    def serpent_pdata(self):
        return self._serpent_pdata

    @property
    def openmc_dir(self):
        if self._openmc_dir is None:
            self._openmc_dir = Path('openmc')
            os.makedirs(self._openmc_dir, exist_ok=True)
        return self._openmc_dir

    @property
    def other_dir(self):
        if self._other_dir is None:
            self._other_dir = Path(self.code)
            os.makedirs(self._other_dir, exist_ok=True)
        return self._other_dir

    @property
    def table_names(self):
        table_names = []
        for element, _ in self.elements:
            Z = ATOMIC_NUMBER[element]
            table_names.append(f'{1000*Z}.{self.suffix}')
        return table_names

    @energy.setter
    def energy(self, energy):
        if energy <= self._cutoff_energy:
            msg = (f'Energy {energy} eV must be above the cutoff energy '
                   f'{self._cutoff_energy} eV.')
            raise ValueError(msg)
        self._energy = energy

    @particles.setter
    def particles(self, particles):
        if particles % self._batches != 0:
            msg = (f'Number of particles {particles} must be divisible by '
                   f'the number of batches {self._batches}.')
            raise ValueError(msg)
        self._particles = particles

    @code.setter
    def code(self, code):
        if code not in ('mcnp', 'serpent'):
            msg = (f'Unsupported code {code}: code must be either "mcnp" or '
                   '"serpent".')
            raise ValueError(msg)
        executable = 'mcnp6' if code == 'mcnp' else 'sss2'
        if not shutil.which(executable, os.X_OK):
            msg = f'Unable to locate executable {executable} in path.'
            raise ValueError(msg)
        self._code = code

    @suffix.setter
    def suffix(self, suffix):
        if not re.match('0[1-4]p|63p|84p|12p', suffix):
            msg = f'Unsupported cross section suffix {suffix}.'
            raise ValueError(msg)
        self._suffix = suffix

    @xsdir.setter
    def xsdir(self, xsdir):
        if xsdir is not None:
            xsdir = Path(xsdir)
            if not xsdir.is_file():
                msg = f'Could not locate the XSDIR file {xsdir}.'
                raise ValueError(msg)
        self._xsdir = xsdir

    @serpent_pdata.setter
    def serpent_pdata(self, serpent_pdata):
        if self.code == 'serpent':
            if serpent_pdata is None:
                msg = ('Serpent photon data path is required to run a '
                       'calculation with Serpent.')
                raise ValueError(msg)
            serpent_pdata = Path(serpent_pdata).resolve()
            if not serpent_pdata.is_dir():
                msg = (f'Could not locate the Serpent photon data directory '
                       f'{serpent_pdata}.')
                raise ValueError(msg)
        self._serpent_pdata = serpent_pdata

    def _make_openmc_input(self):
        """Generate the OpenMC input XML

        """
        # Define material
        mat = openmc.Material()
        for element, fraction in self.elements:
            mat.add_element(element, fraction)
        mat.set_density('g/cm3', self.density)
        materials = openmc.Materials([mat])
        if self.xsdir is not None:
            xs_path = (self.openmc_dir / 'cross_sections.xml').resolve()
            materials.cross_sections = str(xs_path)
        materials.export_to_xml(self.openmc_dir / 'materials.xml')

        # Set up geometry
        x1 = openmc.XPlane(x0=-1.e9, boundary_type='reflective')
        x2 = openmc.XPlane(x0=+1.e9, boundary_type='reflective')
        y1 = openmc.YPlane(y0=-1.e9, boundary_type='reflective')
        y2 = openmc.YPlane(y0=+1.e9, boundary_type='reflective')
        z1 = openmc.ZPlane(z0=-1.e9, boundary_type='reflective')
        z2 = openmc.ZPlane(z0=+1.e9, boundary_type='reflective')
        cell = openmc.Cell(fill=materials)
        cell.region = +x1 & -x2 & +y1 & -y2 & +z1 & -z2
        geometry = openmc.Geometry([cell])
        geometry.export_to_xml(self.openmc_dir / 'geometry.xml')

        # Define source
        source = openmc.Source()
        source.space = openmc.stats.Point((0,0,0))
        source.angle = openmc.stats.Isotropic()
        source.energy = openmc.stats.Discrete([self.energy], [1.])
        source.particle = 'photon'

        # Settings
        settings = openmc.Settings()
        settings.source = source
        settings.particles = self.particles // self._batches
        settings.run_mode = 'fixed source'
        settings.batches = self._batches
        settings.photon_transport = True
        settings.electron_treatment = self.electron_treatment
        settings.cutoff = {'energy_photon' : self._cutoff_energy}
        settings.export_to_xml(self.openmc_dir / 'settings.xml')
 
        # Define tallies
        energy_bins = np.logspace(np.log10(self._cutoff_energy),
                                  np.log10(1.0001*self.energy), self._bins+1)
        energy_filter = openmc.EnergyFilter(energy_bins)
        particle_filter = openmc.ParticleFilter('photon')
        tally = openmc.Tally(name='tally')
        tally.filters = [energy_filter, particle_filter]
        tally.scores = ['flux']
        tallies = openmc.Tallies([tally])
        tallies.export_to_xml(self.openmc_dir / 'tallies.xml')

    def _make_mcnp_input(self):
        """Generate the MCNP input file

        """
        # Create the problem description
        lines = ['Point source in infinite geometry']

        # Create the cell cards: material 1 inside sphere, void outside
        lines.append('c --- Cell cards ---')
        lines.append(f'1 1 -{self.density} -1 imp:p=1')
        lines.append('2 0 1 imp:p=0')
        lines.append('')

        # Create the surface cards: box centered on origin with 2e9 cm sides`
        # and reflective boundary conditions
        lines.append('c --- Surface cards ---')
        lines.append('*1 rpp -1.e9 1e9 -1.e9 1.e9 -1.e9 1.e9')
        lines.append('')

        # Create the data cards
        lines.append('c --- Data cards ---')

        # Materials
        material_card = 'm1'
        for element, fraction in self.elements:
            Z = openmc.data.ATOMIC_NUMBER[element]
            material_card += f' {Z}000.{self.suffix} -{fraction}'
        lines.append(material_card)

        # Energy in MeV
        energy = self.energy * 1e-6
        cutoff_energy = self._cutoff_energy * 1e-6

        # Physics: photon transport, 1 keV photon cutoff energy
        if self.electron_treatment == 'led':
            flag = 1
        else:
            flag = 'j'
        lines.append('mode p')
        lines.append(f'phys:p j {flag} j j j')
        lines.append(f'cut:p j {cutoff_energy}')

        # Source definition: isotropic point source at center of sphere
        lines.append(f'sdef cel=1 erg={energy}')

        # Tallies: photon flux over cell
        lines.append('f4:p 1')
        lines.append(f'e4 {cutoff_energy} {self._bins-1}ilog {1.0001*energy}')

        # Problem termination: number of particles to transport
        lines.append(f'nps {self.particles}')

        # Write the problem
        with open(self.other_dir / 'inp', 'w') as f:
            f.write('\n'.join(lines))

    def _make_serpent_input(self):
        """Generate the Serpent input file

        """
        # Create the problem description
        lines = ['% Point source in infinite geometry']
        lines.append('')

        # Set the cross section library directory
        if self.xsdir is not None:
            xsdata = (self.other_dir / 'xsdata').resolve()
            lines.append(f'set acelib "{xsdata}"')

        # Set the photon data directory
        lines.append(f'set pdatadir "{self.serpent_pdata}"')
        lines.append('')

        # Create the cell cards: material 1 inside sphere, void outside
        lines.append('% --- Cell cards ---')
        lines.append('cell 1 0 m1 -1')
        lines.append('cell 2 0 outside 1')
        lines.append('')

        # Create the surface cards: box centered on origin with 2e9 cm sides`
        # and reflective boundary conditions
        lines.append('% --- Surface cards ---')
        lines.append('surf 1 cube 0.0 0.0 0.0 1.e9')

        # Reflective boundary conditions
        lines.append('set bc 2')
        lines.append('')

        # Create the material cards
        lines.append('% --- Material cards ---')
        lines.append(f'mat m1 -{self.density}')

        # Add element data
        for element, fraction in self.elements:
            Z = ATOMIC_NUMBER[element]
            name = f'{1000*Z}.{self.suffix}'
            lines.append(f'{name} {fraction}')

        # Turn on unresolved resonance probability treatment
        lines.append('set ures 1')

        # Set electron treatment
        if self.electron_treatment == 'led':
            lines.append('set ttb 0')
        else:
            lines.append('set ttb 1')

        # Energy in MeV
        energy = self.energy * 1e-6
        cutoff_energy = self._cutoff_energy * 1e-6

        # Set cutoff energy
        lines.append(f'set ecut 0 {cutoff_energy}')
        lines.append('')

        # External source mode with isotropic point source at center of sphere
        lines.append('% --- Set external source mode ---')
        lines.append(f'set nps {self.particles} {self._batches}')
        lines.append(f'src 1 g se {energy} sp 0.0 0.0 0.0')
        lines.append('')

        # Detector definition: flux energy spectrum
        lines.append('% --- Detector definition ---')
        lines.append('det 1 de 1 dc 1')

        # Energy grid definition: equal lethargy spacing
        lines.append(f'ene 1 3 {self._bins} {cutoff_energy} {1.0001*energy}')
        lines.append('')

        # Write the problem
        with open(self.other_dir / 'input', 'w') as f:
            f.write('\n'.join(lines))

    def _plot(self):
        """Extract and plot the results
 
        """
        # Read results
        path = self.openmc_dir / f'statepoint.{self._batches}.h5'
        x1, y1, _ = read_results('openmc', path)
        if self.code == 'serpent':
            path = self.other_dir / 'input_det0.m'
        else:
            path = self.other_dir / 'outp'
        x2, y2, sd = read_results(self.code, path)

        # Normalize the spectra
        cutoff_energy = self._cutoff_energy * 1e-6
        y1 /= np.diff(np.insert(x1, 0, cutoff_energy))*sum(y1)
        y2 /= np.diff(np.insert(x2, 0, cutoff_energy))*sum(y2)

        # Compute the relative error
        err = np.zeros_like(y2)
        idx = np.where(y2 > 0)
        err[idx] = (y1[idx] - y2[idx])/y2[idx]
 
        # Set up the figure
        fig = plt.figure(1, facecolor='w', figsize=(8,8))
        ax1 = fig.add_subplot(111)
 
        # Create a second y-axis that shares the same x-axis, keeping the first
        # axis in front
        ax2 = ax1.twinx()
        ax1.set_zorder(ax2.get_zorder() + 1)
        ax1.patch.set_visible(False)
 
        # Plot the spectra
        label = 'Serpent' if self.code == 'serpent' else 'MCNP'
        ax1.loglog(x2, y2, 'r', linewidth=1, label=label)
        ax1.loglog(x1, y1, 'b', linewidth=1, label='OpenMC', linestyle='--')
 
        # Plot the relative error and uncertainties
        ax2.semilogx(x2, err, color=(0.2, 0.8, 0.0), linewidth=1)
        ax2.semilogx(x2, 2*sd, color='k', linestyle='--', linewidth=1)
        ax2.semilogx(x2, -2*sd, color='k', linestyle='--', linewidth=1)
 
        # Set grid and tick marks
        ax1.tick_params(axis='both', which='both', direction='in', length=10)
        ax1.grid(b=False, axis='both', which='both')
        ax2.tick_params(axis='y', which='both', right=False)
        ax2.grid(b=True, which='both', axis='both', alpha=0.5, linestyle='--')
 
        # Energy in MeV
        energy = self.energy * 1e-6

        # Set axes labels and limits
        ax1.set_xlim([cutoff_energy, energy])
        ax1.set_xlabel('Energy (MeV)', size=12)
        ax1.set_ylabel('Spectrum', size=12)
        ax1.legend()
        ax2.set_ylabel("Relative error", size=12)
        title = f'{self.material}, {energy:.1e} MeV Source'
        plt.title(title)
 
        # Save plot
        os.makedirs('plots', exist_ok=True)
        if self.name is not None:
            name = self.name
        else:
            name = f'{self.material}-{energy:.1e}MeV'
        plt.savefig(Path('plots') / f'{name}.png', bbox_inches='tight')
        plt.close()

    def run(self):
        """Generate inputs, run problem, and plot results.
 
        """
        # Create the HDF5 library
        if self.xsdir is not None:
            path = self.other_dir if self.code == 'serpent' else None
            create_library(self.xsdir, self.table_names, self.openmc_dir, path)

            # TODO: Currently the neutron libraries are still read in to OpenMC
            # even when doing pure photon transport, so we need to locate them and
            # register them with the library.
            path = os.getenv('OPENMC_CROSS_SECTIONS')
            lib = openmc.data.DataLibrary.from_xml(path)

            path = self.openmc_dir / 'cross_sections.xml'
            data_lib = openmc.data.DataLibrary.from_xml(path)

            for element, fraction in self.elements:
                element = openmc.Element(element)
                for nuclide, _, _ in element.expand(fraction, 'ao'):
                    h5_file = lib.get_by_material(nuclide)['path']
                    data_lib.register_file(h5_file)

            data_lib.export_to_xml(path)

        # Generate input files
        self._make_openmc_input()

        if self.code == 'serpent':
            self._make_serpent_input()
            args = ['sss2', 'input']
        else:
            self._make_mcnp_input()
            args = ['mcnp6']
            if self.xsdir is not None:
                args.append(f'XSDIR={self.xsdir}')

        # Remove old MCNP output files
        for f in ('outp', 'runtpe'):
            try:
                os.remove(self.other_dir / f)
            except OSError:
                pass
 
        # Run code and capture and print output
        p = subprocess.Popen(
            args, cwd=self.other_dir, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, universal_newlines=True
        )

        while True:
            line = p.stdout.readline()
            if not line and p.poll() is not None:
                break
            print(line, end='')

        openmc.run(cwd=self.openmc_dir)

        self._plot()
