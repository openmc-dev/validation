#!/usr/bin/env python3

import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess

import h5py
from matplotlib import pyplot as plt
import numpy as np

import openmc
from openmc.data import K_BOLTZMANN
from .utils import zaid, szax, create_library, read_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('nuclide', type=str,
                        help='Name of the nuclide, e.g. "U235"')
    parser.add_argument('-d', '--density', type=float, default=1.,
                        help='Density of the material in g/cm^3')
    parser.add_argument('-e', '--energy', type=float, default=1e6,
                        help='Energy of the source in eV')
    parser.add_argument('-p', '--particles', type=int, default=100000,
                        help='Number of source particles')
    parser.add_argument('-c', '--code', choices=['mcnp', 'serpent'],
                        default='mcnp',
                        help='Code to validate OpenMC against.')
    parser.add_argument('-s', '--suffix', type=str, default='70c',
                        help='MCNP cross section suffix')
    parser.add_argument('-x', '--xsdir', type=str, help='XSDIR directory '
                        'file. If specified, it will be used to locate the '
                        'ACE table corresponding to the given nuclide and '
                        'suffix, and an HDF5 library that can be used by '
                        'OpenMC will be created from the data.')
    parser.add_argument('-t', '--thermal', type=str, help='ZAID of the '
                        'thermal scattering data, e.g. "grph.10t". If '
                        'specified, thermal scattering data will be assigned '
                        'to the material.')
    parser.add_argument('-o', '--output-name', type=str,
                        help='Name used for output.')
    args = parser.parse_args()

    model = NeutronPhysicsModel(
        args.nuclide, args.density, args.energy, args.particles, args.code,
        args.suffix, args.xsdir, args.thermal, args.output_name
    )
    model.run()


class NeutronPhysicsModel(object):
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
    code : {'mcnp', 'serpent'}
        Code to validate against
    suffix : str
        Cross section suffix
    xsdir : str
        XSDIR directory file. If specified, it will be used to locate the ACE
        table corresponding to the given nuclide and suffix, and an HDF5
        library that can be used by OpenMC will be created from the data.
    thermal : str
        ZAID of the thermal scattering data. If specified, thermal scattering
        data will be assigned to the material.
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
    code : {'mcnp', 'serpent'}
        Code to validate against
    suffix : str
        Cross section suffix for MCNP
    xsdir : str
        XSDIR directory file. If specified, it will be used to locate the ACE
        table corresponding to the given nuclide and suffix, and an HDF5
        library that can be used by OpenMC will be created from the data.
    thermal : str
        ZAID of the thermal scattering data. If specified, thermal scattering
        data will be assigned to the material.
    name : str
        Name used for output.
    temperature : float
        Temperature (Kelvin) of the cross section data
    bins : int
        Number of bins in the energy grid
    batches : int
        Number of batches to simulate
    min_energy : float
        Lower limit of energy grid (eV)
    openmc_dir : pathlib.Path
        Working directory for OpenMC
    other_dir : pathlib.Path
        Working directory for MCNP or Serpent
    table_names : list of str
        Names of the ACE tables used in the model

    """

    def __init__(self, nuclide, density, energy, particles, code, suffix,
                 xsdir=None, thermal=None, name=None):
        self._temperature = None
        self._bins = 500
        self._batches = 100
        self._min_energy = 1.e-5
        self._openmc_dir = None
        self._other_dir = None

        self.nuclide = nuclide
        self.density = density
        self.energy = energy
        self.particles = particles
        self.code = code
        self.suffix = suffix
        self.xsdir = xsdir
        self.thermal = thermal
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
        table_names = [zaid(self.nuclide, self.suffix)]
        if self.thermal is not None:
            table_names.append(self.thermal)
        return table_names

    @energy.setter
    def energy(self, energy):
        if energy <= self._min_energy:
            msg = (f'Energy {energy} eV must be above the minimum energy '
                   f'{self._min_energy} eV.')
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
        match = '(7[0-4]c)|(8[0-6]c)|(71[0-6]nc)|[0][3,6,9]c|[1][2,5,8]c'
        if not re.match(match, suffix):
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

    def _make_openmc_input(self):
        """Generate the OpenMC input XML

        """
        # Define material
        mat = openmc.Material()
        mat.add_nuclide(self.nuclide, 1.0)
        if self.thermal is not None:
            name, suffix = self.thermal.split('.')
            thermal_name = openmc.data.thermal.get_thermal_name(name)
            mat.add_s_alpha_beta(thermal_name)
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

        # Settings
        settings = openmc.Settings()
        if self._temperature is not None:
            settings.temperature = {'default': self._temperature}
        settings.source = source
        settings.particles = self.particles // self._batches
        settings.run_mode = 'fixed source'
        settings.batches = self._batches
        settings.create_fission_neutrons = False
        settings.export_to_xml(self.openmc_dir / 'settings.xml')
 
        # Define tallies
        energy_bins = np.logspace(np.log10(self._min_energy),
                                  np.log10(1.0001*self.energy), self._bins+1)
        energy_filter = openmc.EnergyFilter(energy_bins)
        tally = openmc.Tally(name='tally')
        tally.filters = [energy_filter]
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
        if self._temperature is not None:
            kT = self._temperature * K_BOLTZMANN * 1e-6
            lines.append(f'1 1 -{self.density} -1 imp:n=1 tmp={kT}')
        else:
            lines.append(f'1 1 -{self.density} -1 imp:n=1')
        lines.append('2 0 1 imp:n=0')
        lines.append('')

        # Create the surface cards: box centered on origin with 2e9 cm sides`
        # and reflective boundary conditions
        lines.append('c --- Surface cards ---')
        lines.append('*1 rpp -1.e9 1e9 -1.e9 1.e9 -1.e9 1.e9')
        lines.append('')

        # Create the data cards
        lines.append('c --- Data cards ---')

        # Materials
        if re.match('(71[0-6]nc)', self.suffix):
            name = szax(self.nuclide, self.suffix)
        else:
            name = zaid(self.nuclide, self.suffix)
        lines.append(f'm1 {name} 1.0')
        if self.thermal is not None:
            lines.append(f'mt1 {self.thermal}')
        lines.append('nonu 2')

        # Physics: neutron transport
        lines.append('mode n')

        # Source definition: isotropic point source at center of sphere
        energy = self.energy * 1e-6
        lines.append(f'sdef cel=1 erg={energy}')

        # Tallies: neutron flux over cell
        lines.append('f4:n 1')
        min_energy = self._min_energy * 1e-6
        lines.append(f'e4 {min_energy} {self._bins-1}ilog {1.0001*energy}')

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
        name = zaid(self.nuclide, self.suffix)
        if self.thermal is not None:
            Z, A, m = openmc.data.zam(self.nuclide)
            lines.append(f'mat m1 -{self.density} moder t1 {1000*Z + A}')
        else:
            lines.append(f'mat m1 -{self.density}')
        lines.append(f'{name} 1.0')

        # Add thermal scattering library associated with the nuclide
        if self.thermal is not None:
            lines.append(f'therm t1 {self.thermal}')
        lines.append('')

        # External source mode with isotropic point source at center of sphere
        lines.append('% --- Set external source mode ---')
        lines.append(f'set nps {self.particles} {self._batches}')
        energy = self.energy * 1e-6
        lines.append(f'src 1 n se {energy} sp 0.0 0.0 0.0')
        lines.append('')

        # Detector definition: flux energy spectrum
        lines.append('% --- Detector definition ---')
        lines.append('det 1 de 1 dc 1')

        # Energy grid definition: equal lethargy spacing
        min_energy = self._min_energy * 1e-6
        lines.append(f'ene 1 3 {self._bins} {min_energy} {1.0001*energy}')
        lines.append('')

        # Treat fission as capture
        lines.append('set nphys 0')

        # Turn on unresolved resonance probability treatment
        lines.append('set ures 1')

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

        # Convert energies to eV
        x1 *= 1e6
        x2 *= 1e6

        # Normalize the spectra
        y1 /= np.diff(np.insert(x1, 0, self._min_energy))*sum(y1)
        y2 /= np.diff(np.insert(x2, 0, self._min_energy))*sum(y2)

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
 
        # Set axes labels and limits
        ax1.set_xlim([self._min_energy, self.energy])
        ax1.set_xlabel('Energy (eV)', size=12)
        ax1.set_ylabel('Spectrum', size=12)
        ax1.legend()
        ax2.set_ylabel("Relative error", size=12)
        title = f'{self.nuclide}'
        if self.thermal is not None:
            name, suffix = self.thermal.split('.')
            thermal_name = openmc.data.thermal.get_thermal_name(name)
            title += f' + {thermal_name}'
        title += f', {self.energy:.1e} eV Source'
        plt.title(title)
 
        # Save plot
        os.makedirs('plots', exist_ok=True)
        if self.name is not None:
            name = self.name
        else:
            name = f'{self.nuclide}'
            if self.thermal is not None:
                name += f'-{thermal_name}'
            name += f'-{self.energy:.1e}eV'
            if self._temperature is not None:
                name +=  f'-{self._temperature:.1f}K'
        plt.savefig(Path('plots') / f'{name}.png', bbox_inches='tight')
        plt.close()

    def run(self):
        """Generate inputs, run problem, and plot results.
 
        """
        # Create HDF5 cross section library and Serpent XSDATA file
        if self.xsdir is not None:
            path = self.other_dir if self.code == 'serpent' else None
            create_library(self.xsdir, self.table_names, self.openmc_dir, path)

            # Get the temperature of the cross section data
            f = h5py.File(self.openmc_dir / (self.nuclide + '.h5'), 'r')
            temperature = list(f[self.nuclide]['kTs'].values())[0][()]
            self._temperature = temperature / K_BOLTZMANN

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
        p = subprocess.Popen(args, cwd=self.code, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, universal_newlines=True)
        while True:
            line = p.stdout.readline()
            if not line and p.poll() is not None:
                break
            print(line, end='')

        openmc.run(cwd='openmc')

        self._plot()
