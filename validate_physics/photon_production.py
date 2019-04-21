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
from .utils import zaid, szax, XSDIR


class PhotonProductionModel(object):
    """Monoenergetic, monodirectional neutron source directed down a thin,
    infinitely long cylinder ('Broomstick' problem).

    Parameters
    ----------
    material : str
        Name of the material.
    density : float
        Density of the material in g/cm^3.
    nuclides : list of tuple
        List in which each item is a 2-tuple consisting of a nuclide string and
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
        Neutron cross section suffix
    photon_suffix : str
        Photon cross section suffix
    xsdir : str
        XSDIR directory file. If specified, it will be used to locate the ACE
        table corresponding to the given nuclide and suffix, and an HDF5
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
    nuclides : list of tuple
        List in which each item is a 2-tuple consisting of a nuclide string and
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
        Neutron cross section suffix
    photon_suffix : str
        Photon cross section suffix
    xsdir : str
        XSDIR directory file. If specified, it will be used to locate the ACE
        table corresponding to the given nuclide and suffix, and an HDF5
        library that can be used by OpenMC will be created from the data.
    serpent_pdata : str
        Directory containing the additional data files needed for photon
        physics in Serpent.
    name : str
        Name used for output.
    temperature : float
        Temperature (Kelvin) of the cross section data
    bins : int
        Number of bins in the energy grid
    batches : int
        Number of batches to simulate
    max_energy : float
        Upper limit of energy grid (eV)
    cutoff_energy: float
        Photon cutoff energy (eV)
    directory_openmc : pathlib.Path
        Working directory for OpenMC
    directory_other : pathlib.Path
        Working directory for MCNP or Serpent

    """

    def __init__(self, material, density, nuclides, energy, particles,
                 electron_treatment, code, suffix, photon_suffix, xsdir=None,
                 serpent_pdata=None, name=None):
        self._temperature = None
        self._bins = 500
        self._batches = 100
        self._cutoff_energy = 1.e3
        self._directory_openmc = None
        self._directory_other = None

        self.material = material
        self.density = density
        self.nuclides = nuclides
        self.energy = energy
        self.particles = particles
        self.electron_treatment = electron_treatment
        self.code = code
        self.suffix = suffix
        self.photon_suffix = photon_suffix
        self.xsdir = xsdir
        self.serpent_pdata = serpent_pdata
        self.name = name

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
    def photon_suffix(self):
        return self._photon_suffix

    @property
    def xsdir(self):
        return self._xsdir

    @property
    def serpent_pdata(self):
        return self._serpent_pdata

    @property
    def max_energy(self):
        if self.energy < 1.e6:
            return 1.e7
        else:
            return self.energy * 10

    @property
    def directory_openmc(self):
        if self._directory_openmc is None:
            self._directory_openmc = Path('openmc')
            os.makedirs(self._directory_openmc, exist_ok=True)
        return self._directory_openmc

    @property
    def directory_other(self):
        if self._directory_other is None:
            self._directory_other = Path(self.code)
            os.makedirs(self._directory_other, exist_ok=True)
        return self._directory_other

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

    @photon_suffix.setter
    def photon_suffix(self, photon_suffix):
        if not re.match('0[1-4]p|63p|84p|12p', photon_suffix):
            msg = f'Unsupported photon cross section suffix {photon_suffix}.'
            raise ValueError(msg)
        self._photon_suffix = photon_suffix

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

    def _create_library(self):
        """Convert the ACE data from the MCNP or Serpent distribution into an
        HDF5 library that can be used by OpenMC.

        """
        # Create data library
        data_lib = openmc.data.DataLibrary()

        # Get names of the ACE tables for all nuclides in model
        table_names = set()
        for nuclide, _ in self.nuclides:
            # Name of neutron cross section table
            table_names.add(zaid(nuclide, self.suffix))

            # Name of photon cross section table
            Z, A, m = openmc.data.zam(nuclide)
            table_names.add(f'{1000*Z}.{self.photon_suffix}')

        # Load the XSDIR directory file
        xsdir = XSDIR(self.xsdir)

        # Get the ACE cross section tables
        tables = xsdir.get_tables(table_names)

        for table in tables:
            # Convert cross section data
            if table.name[-1] == 'c':
                match = '(7[0-4]c)|(8[0-6]c)|(71[0-6]nc)'
                scheme = 'mcnp' if re.match(match, self.suffix) else 'nndc'
                data = openmc.data.IncidentNeutron.from_ace(table, scheme)
                if self._temperature is None:
                    self._temperature = data.kTs[0] / K_BOLTZMANN
            else:
                data = openmc.data.IncidentPhoton.from_ace(table)

            # Export HDF5 files and register with library
            h5_file = self.directory_openmc / f'{data.name}.h5'
            data.export_to_hdf5(h5_file, 'w')
            data_lib.register_file(h5_file)

        # Write cross_sections.xml
        data_lib.export_to_xml(self.directory_openmc / 'cross_sections.xml')

        # Write the Serpent XSDATA file
        if self.code == 'serpent':
            xsdir.export_to_xsdata(self.directory_other / 'xsdata', table_names)

    def _make_openmc_input(self):
        """Generate the OpenMC input XML

        """
        # Define material
        mat = openmc.Material()
        for nuclide, fraction in self.nuclides:
            mat.add_nuclide(nuclide, fraction)
        mat.set_density('g/cm3', self.density)
        materials = openmc.Materials([mat])
        if self.xsdir is not None:
            xs_path = (self.directory_openmc / 'cross_sections.xml').resolve()
            materials.cross_sections = str(xs_path)
        materials.export_to_xml(self.directory_openmc / 'materials.xml')
 
        # Instantiate surfaces
        cyl = openmc.XCylinder(boundary_type='vacuum', r=1.e-6)
        px1 = openmc.XPlane(boundary_type='vacuum', x0=-1.)
        px2 = openmc.XPlane(boundary_type='transmission', x0=1.)
        px3 = openmc.XPlane(boundary_type='vacuum', x0=1.e9)
 
        # Instantiate cells
        inner_cyl_left = openmc.Cell()
        inner_cyl_right = openmc.Cell()
        outer_cyl = openmc.Cell()
 
        # Set cells regions and materials
        inner_cyl_left.region = -cyl & +px1 & -px2
        inner_cyl_right.region = -cyl & +px2 & -px3
        outer_cyl.region = ~(-cyl & +px1 & -px3)
        inner_cyl_right.fill = mat
 
        # Create root universe and export to XML
        geometry = openmc.Geometry([inner_cyl_left, inner_cyl_right, outer_cyl])
        geometry.export_to_xml(self.directory_openmc / 'geometry.xml')
 
        # Define source
        source = openmc.Source()
        source.space = openmc.stats.Point((0,0,0))
        source.angle = openmc.stats.Monodirectional()
        source.energy = openmc.stats.Discrete([self.energy], [1.])
        source.particle = 'neutron'
 
        # Settings
        settings = openmc.Settings()
        if self._temperature is not None:
            settings.temperature = {'default': self._temperature}
        settings.source = source
        settings.particles = self.particles // self._batches
        settings.run_mode = 'fixed source'
        settings.batches = self._batches
        settings.photon_transport = True
        settings.electron_treatment = self.electron_treatment
        settings.cutoff = {'energy_photon' : self._cutoff_energy}
        settings.export_to_xml(self.directory_openmc / 'settings.xml')
 
        # Define filters
        surface_filter = openmc.SurfaceFilter(cyl)
        particle_filter = openmc.ParticleFilter('photon')
        energy_bins = np.logspace(np.log10(self._cutoff_energy),
                                  np.log10(self.max_energy), self._bins+1)
        energy_filter = openmc.EnergyFilter(energy_bins)
 
        # Create tallies and export to XML
        tally = openmc.Tally(name='photon current')
        tally.filters = [surface_filter, energy_filter, particle_filter]
        tally.scores = ['current']
        tallies = openmc.Tallies([tally])
        tallies.export_to_xml(self.directory_openmc / 'tallies.xml')

    def _make_mcnp_input(self):
        """Generate the MCNP input file

        """
        # Create the problem description
        lines = ['Broomstick problem']

        # Create the cell cards: material 1 inside cylinder, void outside
        lines.append('c --- Cell cards ---')
        if self._temperature is not None:
            kT = self._temperature * openmc.data.K_BOLTZMANN * 1e-6
            lines.append(f'1 1 -{self.density} -4 6 -7 imp:n,p=1 tmp={kT}')
        else:
            lines.append(f'1 1 -{self.density} -4 6 -7 imp:n,p=1')
        lines.append('2 0 -4 5 -6 imp:n,p=1')
        lines.append('3 0 #(-4 5 -7) imp:n,p=0')
        lines.append('')

        # Create the surface cards: cylinder with radius 1e-6 cm along x-axis
        lines.append('c --- Surface cards ---')
        lines.append('4 cx 1.0e-6')
        lines.append('5 px -1.0')
        lines.append('6 px 1.0')
        lines.append('7 px 1.0e9')
        lines.append('')

        # Create the data cards
        lines.append('c --- Data cards ---')

        # Materials
        material_card = 'm1'
        for nuclide, fraction in self.nuclides:
            if re.match('(71[0-6]nc)', self.suffix):
                name = szax(nuclide, self.suffix)
            else:
                name = zaid(nuclide, self.suffix)
            material_card += f' {name} -{fraction} plib={self.photon_suffix}'
        lines.append(material_card)

        # Energy in MeV
        energy = self.energy * 1e-6
        max_energy = self.max_energy * 1e-6
        cutoff_energy = self._cutoff_energy * 1e-6

        # Physics: neutron and neutron-induced photon, 1 keV photon cutoff energy
        if self.electron_treatment == 'led':
            flag = 1
        else:
            flag = 'j'
        lines.append('mode n p')
        lines.append(f'phys:p j {flag} j j j')
        lines.append(f'cut:p j {cutoff_energy}')

        # Source definition: point source at origin monodirectional along
        # positive x-axis
        lines.append(f'sdef cel=2 erg={energy} vec=1 0 0 dir=1 par=1')

        # Tallies: Photon current over surface
        lines.append('f1:p 4')
        lines.append(f'e1 {cutoff_energy} {self._bins-1}ilog {max_energy}')

        # Problem termination: number of particles to transport
        lines.append(f'nps {self.particles}')

        # Write the problem
        with open(self.directory_other / 'inp', 'w') as f:
            f.write('\n'.join(lines))

    def _make_serpent_input(self):
        """Generate the Serpent input file

        """
        # Create the problem description
        lines = ['% Broomstick problem']
        lines.append('')
 
        # Set the cross section library directory
        if self.xsdir is not None:
            xsdata = (self.directory_other / 'xsdata').resolve()
            lines.append(f'set acelib "{xsdata}"')

        # Set the photon data directory
        lines.append(f'set pdatadir "{self.serpent_pdata}"')
        lines.append('')

        # Create the cell cards: material 1 inside cylinder, void outside
        lines.append('% --- Cell cards ---')
        lines.append('cell 1 0 m1 -1 3 -4')
        lines.append('cell 2 0 void -1 2 -3')
        lines.append('cell 3 0 outside 1')
        lines.append('cell 4 0 outside -2')
        lines.append('cell 5 0 outside 4')
        lines.append('')

        # Create the surface cards: cylinder with radius 1e-6 cm along x-axis
        lines.append('% --- Surface cards ---')
        lines.append('surf 1 cylx 0.0 0.0 1.0e-6')
        lines.append('surf 2 px -1.0')
        lines.append('surf 3 px 1.0')
        lines.append('surf 4 px 1.0e9')
        lines.append('')

        # Create the material cards
        lines.append('% --- Material cards ---')
        lines.append(f'mat m1 -{self.density}')
        elements = {}
        for nuclide, fraction in self.nuclides:
            # Add nuclide data
            name = zaid(nuclide, self.suffix)
            lines.append(f'{name} {fraction}')

            # Sum element fractions
            Z, A, m = openmc.data.zam(nuclide)
            name = f'{1000*Z}.{self.photon_suffix}'
            if name not in elements:
                elements[name] = fraction
            else:
                elements[name] += fraction

        # Add element data
        for name, fraction in elements.items():
                lines.append(f'{name} {fraction}')
        lines.append('')

        # Turn on unresolved resonance probability treatment
        lines.append('set ures 1')

        # Set electron treatment
        if self.electron_treatment == 'led':
            lines.append('set ttb 0')
        else:
            lines.append('set ttb 1')

        # Turn on Doppler broadening of Compton scattered photons (on by
        # default)
        lines.append('set cdop 1')

        # Coupled neutron-gamma calculations (0 is off, 1 is analog, 2 is
        # implicit)
        lines.append('set ngamma 1')

        # Energy in MeV
        energy = self.energy * 1e-6
        max_energy = self.max_energy * 1e-6
        cutoff_energy = self._cutoff_energy * 1e-6

        # Set cutoff energy
        lines.append(f'set ecut 0 {cutoff_energy}')
        lines.append('')

        # External source mode with isotropic point source at center of sphere
        lines.append('% --- Set external source mode ---')
        lines.append(f'set nps {self.particles} {self._batches}')
        lines.append(f'src 1 n se {energy} sp 0.0 0.0 0.0 sd 1.0 0.0 0.0')
        lines.append('')

        # Detector definition: photon current over surface
        lines.append('% --- Detector definition ---')
        lines.append('det 1 p de 1 ds 1 1')

        # Energy grid definition: equal lethargy spacing
        lines.append(f'ene 1 3 {self._bins} {cutoff_energy} {max_energy}')
        lines.append('')

        # Write the problem
        with open(self.directory_other / 'input', 'w') as f:
            f.write('\n'.join(lines))

    def _read_openmc_results(self):
        """Extract the results from the OpenMC statepoint

        """
        # Read the results from the OpenMC statepoint
        path = self.directory_openmc / f'statepoint.{self._batches}.h5'

        with openmc.StatePoint(path) as sp:
            t = sp.get_tally(name='photon current')
            x = t.find_filter(openmc.EnergyFilter).bins[:,1] * 1e-6
            y = t.mean[:,0,0]
            sd = t.std_dev[:,0,0]

        # Normalize the spectrum
        cutoff_energy = self._cutoff_energy * 1e-6
        y /= np.diff(np.insert(x, 0, cutoff_energy))*sum(y)

        return x, y, sd

    def _read_mcnp_results(self):
        """Extract the results from the MCNP output file

        """
        with open(self.directory_other / 'outp', 'r') as f:
            text = f.read()
            p = text.find(f'1tally{1: >9}')
            p = text.find('energy', p) + 10
            q = text.find('total', p)
            t = np.fromiter(text[p:q].split(), float)
            t.shape = (len(t) // 3, 3)
            x = t[1:,0]
            y = t[1:,1]
            sd = t[1:,2]
 
        # Normalize the spectrum
        cutoff_energy = self._cutoff_energy * 1e-6
        y /= np.diff(np.insert(x, 0, cutoff_energy))*sum(y)

        return x, y, sd

    def _read_serpent_results(self):
        """Extract the results from the Serpent output file

        """
        with open(self.directory_other / 'input_det0.m', 'r') as f:
            text = f.read().split()
            n = self._bins
            t = np.fromiter(text[3:3+12*n], float).reshape(n, 12)
            e = np.fromiter(text[7+12*n:7+15*n], float).reshape(n, 3)
            x = e[:,1]
            y = t[:,10]
            sd = t[:,11]

        # Normalize the spectrum
        cutoff_energy = self._cutoff_energy * 1e-6
        y /= np.diff(np.insert(x, 0, cutoff_energy))*sum(y)

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

        # Energy in MeV
        energy = self.energy * 1e-6
        max_energy = self.max_energy * 1e-6
        cutoff_energy = self._cutoff_energy * 1e-6

        # Set axes labels and limits
        ax1.set_xlim([cutoff_energy, max_energy])
        ax1.set_xlabel('Energy (MeV)', size=12)
        ax1.set_ylabel('Particle Current', size=12)
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
            if self._temperature is not None:
                name +=  f'-{self._temperature:.1f}K'
        plt.savefig(Path('plots') / f'{name}.png', bbox_inches='tight')
        plt.close()

    def run(self):
        """Generate inputs, run problem, and plot results.
 
        """
        if self.xsdir is not None:
            self._create_library()

        # Generate input files
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
                    os.remove(self.directory_other / f)
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
