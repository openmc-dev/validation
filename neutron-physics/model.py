#!/usr/bin/env python3

import os
from pathlib import Path
import re
import shutil
import subprocess

from matplotlib import pyplot as plt
import numpy as np

import openmc
from openmc.data import K_BOLTZMANN, NEUTRON_MASS


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
    code : {'mcnp', 'serpent'}
        Code to validate against
    suffix : str
        Cross section suffix for MCNP
    library : str
        XSDIR directory file. If specified, it will be used to locate the ACE
        table corresponding to the given nuclide and suffix, and an HDF5
        library that can be used by OpenMC will be created from the data.
    name : str
        Name used for output.
    thermal : str
        ZAID of the thermal scattering data. If specified, thermal scattering
        data will be assigned to the material.

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
    library : str
        XSDIR directory file. If specified, it will be used to locate the ACE
        table corresponding to the given nuclide and suffix, and an HDF5
        library that can be used by OpenMC will be created from the data.
    name : str
        Name used for output.
    thermal : str
        ZAID of the thermal scattering data. If specified, thermal scattering
        data will be assigned to the material.
    temperature : float
        Temperature (Kelvin) of the cross section data
    bins : int
        Number of bins in the energy grid
    batches : int
        Number of batches to simulate
    min_energy : float
        Lower limit of energy grid (eV)

    """

    def __init__(self, nuclide, density, energy, particles, code, suffix,
                 library=None, name=None, thermal=None):
        self._temperature = None
        self._bins = 500
        self._batches = 100
        self._min_energy = 1.e-5

        self.nuclide = nuclide
        self.density = density
        self.energy = energy
        self.particles = particles
        self.code = code
        self.suffix = suffix
        self.library = library
        self.name = name
        self.thermal = thermal

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
    def library(self):
        return self._library

    @property
    def zaid(self):
        Z, A, m = openmc.data.zam(self.nuclide)

        # Serpent metastable convention
        if re.match('[0][3,6,9]c|[1][2,5,8]c', self.suffix):
            # Increase mass number above 300
            if m > 0:
                while A < 300:
                    A += 100

        # MCNP metastable convention
        else:
            # Correct the ground state and first excited state of Am242, which
            # are the reverse of the convention
            if A == 242 and m == 0:
                m = 1
            elif A == 242 and m == 1:
                m = 0

            if m > 0:
                A += 300 + 100*m

        if re.match('(71[0-6]nc)', self.suffix):
            suffix = f'8{self.suffix[2]}c'
        else:
            suffix = self.suffix

        return f'{1000*Z + A}.{suffix}'

    @property
    def szax(self):
        Z, A, m = openmc.data.zam(self.nuclide)

        # Correct the ground state and first excited state of Am242, which are
        # the reverse of the convention
        if A == 242 and m == 0:
            m = 1
        elif A == 242 and m == 1:
            m = 0

        if re.match('(7[0-4]c)|(8[0-6]c)', self.suffix):
            suffix = f'71{self.suffix[1]}nc'
        else:
            suffix = self.suffix

        return f'{1000000*m + 1000*Z + A}.{suffix}'

    @energy.setter
    def energy(self, energy):
        if energy <= self._min_energy:
            msg = (f'Energy {energy} eV is not above the minimum energy '
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
        if code not in ['mcnp', 'serpent']:
            msg = (f'Unsupported code {code}: code must be either "mcnp" or '
                   f'"serpent".')
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

    @library.setter
    def library(self, library):
        if library is not None:
            library = Path(library)
            if not library.is_file():
                msg = f'XSDIR {library} is not a file.'
                raise ValueError(msg)
        self._library = library

    def _create_library(self):
        """Convert the ACE data from the MCNP or Serpent distribution into an
        HDF5 library that can be used by OpenMC.

        """
        datapath = None
        entry = None
        thermal_entry = None

        # Get the name of the ACE table
        name = self.zaid

        # Get the location of the table from the XSDIR directory file
        with open(self.library) as f:
            # Read the datapath if it is specified
            line = f.readline()
            tokens = re.split('\s|=', line)
            if tokens[0].lower() == 'datapath':
                datapath = Path(tokens[1])

            line = f.readline()
            while line:
                # Handle continuation lines
                while line[-2] == '+':
                    line += f.readline()
                    line = line.replace('+\n', '')

                tokens = line.split()

                # Locate the entry for the table
                if tokens[0] == name:
                    entry = tokens

                # Locate the entry for the S(alpha, beta) table if needed
                elif self.thermal is not None and tokens[0] == self.thermal:
                    thermal_entry = tokens

                # Check if we have found all entries
                if ((self.thermal is None or thermal_entry is not None)
                    and entry is not None):
                    break

                line = f.readline()

        # Check that we found the library
        if entry is None:
            msg = f'Could not locate table {name} in XSDIR {self.library}.'
            raise ValueError(msg)

        # Get the access route if it is specified; otherwise, set the parent
        # directory of XSDIR as the datapath
        if datapath is None:
            if entry[3] != '0':
                datapath = Path(entry[3])
            else:
                datapath = self.library.parent

        # Get the full path to the ace library
        path = datapath / entry[2]
        if not path.is_file():
            msg = f'ACE file {path} does not exist.'
            raise ValueError(msg)

        if self.thermal is not None:
            # Check that we found the thermal library
            if thermal_entry is None:
                msg = (f'Could not locate thermal scattering table '
                       f'{self.thermal} in XSDIR {self.library}.')
                raise ValueError(msg)

            # Get the full path to the thermal ace library
            thermal_path = datapath / thermal_entry[2]
            if not thermal_path.is_file():
                msg = f'ACE file {thermal_path} does not exist.'
                raise ValueError(msg)

        # Create the Serpent XSDATA directory file
        if self.code == 'serpent':
            if entry[4] != '1':
                msg = f'File type {entry[4]} not supported for {name}.'
                raise ValueError(msg)
            atomic_weight = float(entry[1]) * NEUTRON_MASS
            temperature = float(entry[9]) / K_BOLTZMANN * 1e6
            Z, A, m = openmc.data.zam(self.nuclide)
            lines = [f'{name} {name} 1 {1000*Z + A} {m} {atomic_weight} '
                     f'{temperature} 0 {path}']

            # Add thermal data
            if self.thermal is not None:
                atomic_weight = float(thermal_entry[1]) * NEUTRON_MASS
                lines.append(f'{self.thermal} {self.thermal} 3 0 0 '
                             f'{atomic_weight} {temperature} 0 {thermal_path}')

            # Write the XSDATA file
            with open(Path('serpent') / 'xsdata', 'w') as f:
                f.write('\n'.join(lines))

        if re.match('(8[0-6]c)|(71[0-6]nc)', self.suffix):
            name = self.szax

        # Get the ACE table
        print(f'Converting table {name} from library {path}...')
        table = openmc.data.ace.get_table(path, name)

        # Convert cross section data
        if re.match('(7[0-4]c)|(8[0-6]c)|(71[0-6]nc)', self.suffix):
            data = openmc.data.IncidentNeutron.from_ace(table, 'mcnp')
        else:
            data = openmc.data.IncidentNeutron.from_ace(table, 'nndc')
        self._temperature = data.kTs[0] / K_BOLTZMANN

        # Create data library and directory for HDF5 files
        data_lib = openmc.data.DataLibrary()
        os.makedirs('openmc', exist_ok=True)

        # Export HDF5 files and register with library
        h5_file = Path('openmc') / f'{data.name}.h5'
        data.export_to_hdf5(h5_file, 'w')
        data_lib.register_file(h5_file)

        if self.thermal is not None:
            # Get the thermal scattering ACE table
            print(f'Converting table {self.thermal} from library '
                  f'{thermal_path}...')
            table = openmc.data.ace.get_table(thermal_path, self.thermal)

            # Convert the thermal scattering data
            data = openmc.data.ThermalScattering.from_ace(table)

            # Export HDF5 files and register with library
            h5_file = Path('openmc') / f'{data.name}.h5'
            data.export_to_hdf5(h5_file, 'w')
            data_lib.register_file(h5_file)

        # Write cross_sections.xml
        data_lib.export_to_xml(Path('openmc') / 'cross_sections.xml')

    def _make_openmc_input(self):
        """Generate the OpenMC input XML

        """
        # Directory from which openmc is run
        os.makedirs('openmc', exist_ok=True)
        
        # Define material
        mat = openmc.Material()
        mat.add_nuclide(self.nuclide, 1.0)
        if self.thermal is not None:
            name, suffix = self.thermal.split('.')
            thermal_name = openmc.data.thermal.get_thermal_name(name)
            mat.add_s_alpha_beta(thermal_name)
        mat.set_density('g/cm3', self.density)
        materials = openmc.Materials([mat])
        if self.library is not None:
            xs_path = (Path('openmc') / 'cross_sections.xml').resolve()
            materials.cross_sections = str(xs_path)
        materials.export_to_xml(Path('openmc') / 'materials.xml')

        # Set up geometry
        sphere = openmc.Sphere(boundary_type='vacuum', r=1.e9)
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
        settings.particles = self.particles // self._batches
        settings.run_mode = 'fixed source'
        settings.batches = self._batches
        settings.create_fission_neutrons = False
        settings.export_to_xml(Path('openmc') / 'settings.xml')
 
        # Define tallies
        energy_bins = np.logspace(np.log10(self._min_energy),
                                  np.log10(1.0001*self.energy), self._bins+1)
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
            kT = self._temperature * K_BOLTZMANN * 1e-6
            lines.append(f'1 1 -{self.density} -1 imp:n=1 tmp={kT}')
        else:
            lines.append(f'1 1 -{self.density} -1 imp:n=1')
        lines.append('2 0 1 imp:n=0')
 
        # Create the surface cards: sphere centered on origin with 1e9 cm
        # radius and vacuum boundary conditions
        lines.append('')
        lines.append('c --- Surface cards ---')
        lines.append('+1 so 1.e9')
 
        # Create the data cards
        lines.append('')
        lines.append('c --- Data cards ---')
 
        # Materials
        if re.match('(71[0-6]nc)', self.suffix):
            name = self.szax
        else:
            name = self.zaid
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
        with open(Path('mcnp') / 'inp', 'w') as f:
            f.write('\n'.join(lines))

    def _make_serpent_input(self):
        """Generate the Serpent input file

        """
        # Directory from which Serpent will be run
        os.makedirs('serpent', exist_ok=True)

        # Create the problem description
        lines = ['% Point source in infinite geometry']
        lines.append('')

        # Set the cross section library directory
        if self.library is not None:
            xsdata = (Path('serpent') / 'xsdata').resolve()
            lines.append(f'set acelib "{xsdata}"')
            lines.append('')
 
        # Create the cell cards: material 1 inside sphere, void outside
        lines.append('% --- Cell cards ---')
        lines.append('cell 1 0 m1 -1')
        lines.append('cell 2 0 outside 1')

        # Create the surface cards: sphere centered on origin with 1e9 cm
        # radius and vacuum boundary conditions
        lines.append('')
        lines.append('% --- Surface cards ---')
        lines.append('surf 1 sph 0.0 0.0 0.0 1.e9')

        # Create the material cards
        lines.append('')
        lines.append('% --- Material cards ---')
        name = self.zaid
        if self.thermal is not None:
            Z, A, m = openmc.data.zam(self.nuclide)
            lines.append(f'mat m1 -{self.density} moder t1 {1000*Z + A}')
        else:
            lines.append(f'mat m1 -{self.density}')
        lines.append(f'{name} 1.0')

        # Add thermal scattering library associated with the nuclide
        if self.thermal is not None:
            lines.append(f'therm t1 {self.thermal}')

        # External source mode with isotropic point source at center of sphere
        lines.append('')
        lines.append('% --- Set external source mode ---')
        lines.append(f'set nps {self.particles} {self._batches}')
        energy = self.energy * 1e-6
        lines.append(f'src 1 n se {energy} sp 0.0 0.0 0.0')

        # Detector definition: flux energy spectrum
        lines.append('')
        lines.append('% --- Detector definition ---')
        lines.append('det 1 de 1 dc 1')

        # Energy grid definition: equal lethargy spacing
        min_energy = self._min_energy * 1e-6
        lines.append(f'ene 1 3 {self._bins} {min_energy} {1.0001*energy}')

        # Treat fission as capture
        lines.append('')
        lines.append('set nphys 0')

        # Turn on unresolved resonance probability treatment
        lines.append('set ures 1')

        # Write the problem
        with open(Path('serpent') / 'input', 'w') as f:
            f.write('\n'.join(lines))

    def _read_openmc_results(self):
        """Extract the results from the OpenMC statepoint

        """
        # Read the results from the OpenMC statepoint
        path = Path('openmc') / f'statepoint.{self._batches}.h5'
        with openmc.StatePoint(path) as sp:
            t = sp.get_tally(name='neutron flux')
            x = t.find_filter(openmc.EnergyFilter).bins[:,1]
            y = t.mean[:,0,0]
            sd = t.std_dev[:,0,0]

        # Normalize the spectrum
        y /= np.diff(np.insert(x, 0, self._min_energy))*sum(y)

        return x, y, sd

    def _read_mcnp_results(self):
        """Extract the results from the MCNP output file

        """
        with open(Path('mcnp') / 'outp', 'r') as f:
            text = f.read()
            p = text.find('1tally')
            p = text.find('energy', p) + 10
            q = text.find('total', p)
            t = np.fromiter(text[p:q].split(), float)
            t.shape = (len(t) // 3, 3)
            x = t[1:,0] * 1.e6
            y = t[1:,1]
            sd = t[1:,2]
 
        # Normalize the spectrum
        y /= np.diff(np.insert(x, 0, self._min_energy))*sum(y)

        return x, y, sd

    def _read_serpent_results(self):
        """Extract the results from the Serpent output file

        """
        with open(Path('serpent') / 'input_det0.m', 'r') as f:
            text = f.read().split()
            n = self._bins
            t = np.fromiter(text[3:3+12*n], float).reshape(n, 12)
            e = np.fromiter(text[7+12*n:7+15*n], float).reshape(n, 3)
            x = e[:,1] * 1.e6
            y = t[:,10]
            sd = t[:,11]

        # Normalize the spectrum
        y /= np.diff(np.insert(x, 0, self._min_energy))*sum(y)

        return x, y, sd

    def _plot(self):
        """Extract and plot the results
 
        """
        # Read results
        x1, y1, _ = self._read_openmc_results()
        if self.code == 'serpent':
            x2, y2, sd = self._read_serpent_results()
        else:
            x2, y2, sd = self._read_mcnp_results()

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
        if self.library is not None:
            self._create_library()

        # Generate input files
        if self.code == 'serpent':
            self._make_serpent_input()
            args = ['sss2', 'input']
        else:
            self._make_mcnp_input()
            args = ['mcnp6']
            if self.library is not None:
                args.append(f'XSDIR={self.library}')

            # Remove old MCNP output files
            for f in ('outp', 'runtpe'):
                try:
                    os.remove(Path('mcnp') / f)
                except OSError:
                    pass

        self._make_openmc_input()

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
