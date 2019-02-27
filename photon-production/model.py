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
    suffix : str
        Neutron cross section suffix for MCNP
    library : str
        Directory containing endf70[a-k] or endf71x MCNP ACE data library. If
        specified, an HDF5 library that can be used by OpenMC will be created
        from the MCNP data.
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
    suffix : str
        Neutron cross section suffix for MCNP
    library : str
        Directory containing endf70[a-k] or endf71x MCNP ACE data library. If
        specified, an HDF5 library that can be used by OpenMC will be created
        from the MCNP data.
    photon_library : str
        Directory containing the MCNP ACE photon library eprdata12. If
        specified, an HDF5 library that can be used by OpenMC will be created.
    name : str
        Name used for output.
    energy_mev : float
        Energy of the source (MeV)
    temperature : float
        Temperature (Kelvin) of the cross section data

    """

    def __init__(self, material, density, nuclides, energy, particles,
                 electron_treatment, suffix, library=None,
                 photon_library=None, name=None):
        self.material = material
        self.density = density
        self.nuclides = nuclides
        self.energy = energy
        self.particles = particles
        self.electron_treatment = electron_treatment
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
        if photon_library is not None:
            self.photon_library = Path(photon_library)
            if not self.photon_library.is_dir():
                msg = f'{self.library} is not a directory.'
                raise ValueError(msg)
        else:
            self.photon_library = photon_library
        self.name = name
        self._temperature = None

    @property
    def energy_mev(self):
        return self.energy*1.e-6

    def _create_library(self):
        """Convert the ACE data from the MCNP distribution into an HDF5 library
        that can be used by OpenMC.

        """
        data_lib = openmc.data.DataLibrary()

        for nuclide, _ in self.nuclides:
            Z, A, m = openmc.data.zam(nuclide)

            if self.library is not None:
                element = openmc.data.ATOMIC_SYMBOL[Z]

                if re.match('7[0-4]c', self.suffix):
                    # Get the table from the ENDF/B-VII.0 neutron ACE library
                    name = zaid(nuclide, self.suffix)
                    for path in self.library.glob('endf70[a-k]'):
                        try:
                            table = openmc.data.ace.get_table(path, name)
                        except ValueError:
                            pass
                else:
                    # Get the table from the ENDF/B-VII.1 neutron ACE library
                    name = szax(nuclide, self.suffix)
                    path = self.library / 'endf71x' / f'{element}' / f'{name}'
                    table = openmc.data.ace.get_table(path, name)

                # Convert cross section data
                data = openmc.data.IncidentNeutron.from_ace(table, 'mcnp')
                self._temperature = data.kTs[0] / openmc.data.K_BOLTZMANN

                # Export HDF5 file
                os.makedirs('openmc', exist_ok=True)
                h5_file = Path('openmc') / f'{data.name}.h5'
                data.export_to_hdf5(h5_file, 'w')

                # Register with library
                data_lib.register_file(h5_file)

                # If only the neutron library is generated from MCNP data,
                # locate the photon HDF5 file for this element and register it
                # with the library
                if self.photon_library is None:
                    path = os.getenv('OPENMC_CROSS_SECTIONS')
                    lib = openmc.data.DataLibrary.from_xml(path)
                    h5_file = lib.get_by_material(element)['path']
                    data_lib.register_file(h5_file)

            if self.photon_library is not None:
                # Get the table from the photon ACE library
                path = self.photon_library / 'eprdata12'
                table = openmc.data.ace.get_table(path, f'{Z}000.12p')

                # Convert cross section data
                data = openmc.data.IncidentPhoton.from_ace(table)

                # Add stopping powers for thick-target bremsstrahlung
                # approximation used in OpenMC
                data_path = Path(openmc.data.__file__).parent
                if Z < 99:
                    path = data_path / 'stopping_powers.h5'
                    with h5py.File(path, 'r') as f:
                        # Units are in MeV; convert to eV
                        data.stopping_powers['energy'] = f['energy'].value*1.e6

                        # Units are in MeV cm^2/g; convert to eV cm^2/g
                        group = f[f'{Z:03}']
                        data.stopping_powers.update({
                            'I': group.attrs['I'],
                            's_collision': group['s_collision'].value*1.e6,
                            's_radiative': group['s_radiative'].value*1.e6
                        })

                # Add bremsstrahlung cross sections used in OpenMC
                path = data_path / 'BREMX.DAT'
                brem = open(path, 'r').read().split()

                # Incident electron kinetic energy grid in eV
                data.bremsstrahlung['electron_energy'] = np.logspace(3, 9, 200)
                log_energy = np.log(data.bremsstrahlung['electron_energy'])

                # Get number of tabulated electron and photon energy values
                n = int(brem[37])
                k = int(brem[38])

                # Index in data
                p = 39

                # Get log of incident electron kinetic energy values, used for
                # cubic spline interpolation in log energy. Units are in MeV, so
                # convert to eV.
                logx = np.log(np.fromiter(brem[p:p+n], float, n)*1.e6)
                p += n

                # Get reduced photon energy values
                data.bremsstrahlung['photon_energy'] = np.fromiter(brem[p:p+k], float, k)
                p += k

                # Get the scaled cross section values for each electron energy
                # and reduced photon energy for this Z. Units are in mb, so
                # convert to b.
                p += (Z - 1)*n*k
                y = np.reshape(np.fromiter(brem[p:p+n*k], float, n*k), (n, k))*1.0e-3

                data.bremsstrahlung['dcs'] = np.empty([len(log_energy), k])
                for j in range(k):
                    # Cubic spline interpolation in log energy and linear DCS
                    cs = CubicSpline(logx, y[:,j])

                    # Get scaled DCS values (millibarns) on new energy grid
                    data.bremsstrahlung['dcs'][:,j] = cs(log_energy)

                # Export HDF5 file
                os.makedirs('openmc', exist_ok=True)
                h5_file = Path('openmc') / f'{data.name}.h5'
                data.export_to_hdf5(h5_file, 'w')

                # Register with library
                data_lib.register_file(h5_file)

                # If only the photon library is generated from MCNP data,
                # locate the neutron HDF5 file for this nuclide and register it
                # with the library
                if self.library is None:
                    path = os.getenv('OPENMC_CROSS_SECTIONS')
                    lib = openmc.data.DataLibrary.from_xml(path)
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
        for nuclide, fraction in self.nuclides:
            mat.add_nuclide(nuclide, fraction)
        mat.set_density('g/cm3', self.density)
        materials = openmc.Materials([mat])
        if self.library is not None or self.photon_library is not None:
            xs_path = (Path('openmc') / 'cross_sections.xml').resolve()
            materials.cross_sections = str(xs_path)
        materials.export_to_xml(Path('openmc') / 'materials.xml')
 
        # Instantiate surfaces
        cyl = openmc.XCylinder(boundary_type='vacuum', R=1.e-6)
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
        geometry.export_to_xml(Path('openmc') / 'geometry.xml')
 
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
        settings.particles = self.particles
        settings.run_mode = 'fixed source'
        settings.batches = 1
        settings.photon_transport = True
        settings.electron_treatment = self.electron_treatment
        settings.cutoff = {'energy_photon' : 1000.}
        settings.export_to_xml(Path('openmc') / 'settings.xml')
 
        # Define filters
        surface_filter = openmc.SurfaceFilter(cyl)
        particle_filter = openmc.ParticleFilter('photon')
        energy_bins = np.logspace(3, np.log10(2*self.energy), 500)
        energy_filter = openmc.EnergyFilter(energy_bins)
 
        # Create tallies and export to XML
        tally = openmc.Tally(name='photon current')
        tally.filters = [surface_filter, energy_filter, particle_filter]
        tally.scores = ['current']
        tallies = openmc.Tallies([tally])
        tallies.export_to_xml(Path('openmc') / 'tallies.xml')

    def _make_mcnp_input(self):
        """Generate the MCNP input file

        """
        # Directory from which MCNP will be run
        os.makedirs('mcnp', exist_ok=True)

        # Create the problem description
        lines = ['Broomstick problem']
 
        # Create the cell cards: material 1 inside cylinder, void outside
        lines.append('c --- Cell cards ---')
        lines.append(f'1 1 -{self.density} -4 6 -7 imp:n,p=1')
        lines.append('2 0 -4 5 -6 imp:n,p=1')
        lines.append('3 0 #(-4 5 -7) imp:n,p=0')
 
        # Create the surface cards: cylinder with radius 1e-6 cm along x-axis
        lines.append('')
        lines.append('c --- Surface cards ---')
        lines.append('4 cx 1.0e-6')
        lines.append('5 px -1.0')
        lines.append('6 px 1.0')
        lines.append('7 px 1.0e9')
 
        # Create the data cards
        lines.append('')
        lines.append('c --- Data cards ---')
 
        # Materials
        material_card = 'm1'
        for nuclide, fraction in self.nuclides:
            if re.match('(71[0-6]nc)', self.suffix):
                name = szax(nuclide, self.suffix)
            else:
                name = zaid(nuclide, self.suffix)
            material_card += f' {name} -{fraction} plib=12p'
        lines.append(material_card)
 
        # Physics: neutron and neutron-induced photon, 1 keV photon cutoff energy
        if self.electron_treatment == 'led':
            flag = 1
        else:
            flag = 'j'
        lines.append('mode n p')
        lines.append(f'phys:p j {flag} j j j')
        lines.append('cut:p j 1.e-3')
 
        # Source definition: point source at origin monodirectional along
        # positive x-axis
        lines.append(f'sdef cel=2 erg={self.energy_mev} vec=1 0 0 dir=1 par=1')
 
        # Tallies: Photon current over surface
        lines.append('f1:p 4')
        lines.append(f'e1 1.e-3 498ilog {2*self.energy_mev}')
 
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
            t = sp.get_tally(name='photon current')
            x_openmc = t.find_filter(openmc.EnergyFilter).bins[:,1]*1.e-6
            y_openmc = t.mean[:,0,0]
 
        # Read the results from the MCNP output file
        with open(Path('mcnp') / 'outp', 'r') as f:
            text = f.read()
            p = text.find(f'1tally{1: >9}')
            p = text.find('energy', p) + 10
            q = text.find('total', p)
            t = np.fromiter(text[p:q].split(), float)
            t.shape = (len(t) // 3, 3)
            x_mcnp = t[1:,0]
            y_mcnp = t[1:,1]
            sd = t[1:,2]
 
        # Normalize the spectra
        y_openmc /= np.diff(np.insert(x_openmc, 0, 1.e-3))
        y_mcnp /= np.diff(np.insert(x_mcnp, 0, 1.e-3))
 
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
        ax1.set_xlim([1.e-3, 2*self.energy_mev])
        ax1.set_xlabel('Energy (MeV)', size=12)
        ax1.set_ylabel('Particle Current', size=12)
        ax1.legend()
        ax2.set_ylabel("Relative error", size=12)
        title = f'{self.material}, {self.energy_mev} MeV Source'
        plt.title(title)
 
        # Save plot
        os.makedirs('plots', exist_ok=True)
        if self.name is None:
            name = f'{self.material}.png'
        else:
            name = f'{self.name}.png'
        plt.savefig(Path('plots') / name, bbox_inches='tight')
        plt.close()

    def run(self):
        """Generate inputs, run problem, and plot results.
 
        """
        if self.library is not None or self.photon_library is not None:
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
