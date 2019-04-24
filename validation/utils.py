from pathlib import Path
import re

import numpy as np

import openmc.data
from openmc.data import K_BOLTZMANN, NEUTRON_MASS


class XSDIR:
    """XSDIR directory file

    Parameters
    ----------
    filename : str
        Path of the XSDIR file to load.

    Attributes
    ----------
    filename : str
        Path of the XSDIR file.
    datapath : str
        Directory where the data libraries are stored.
    atomic_weight_ratio : dict of int to double
        Dictionary whose keys are ZAIDs and values are atomic weight ratios.
    directory : dict of str to XSDIRTable
        Dictionary whose keys are table names and values the entries in an
        XSDIR cross section table description.

    """

    def __init__(self, filename):
        self.filename = filename
        self.datapath = None
        self.atomic_weight_ratio = {}
        self.directory = {}

        self._read()

    def _read(self):
        """Read the XSDIR directory file.

        """
        with open(self.filename) as f:
            # First section: read the datapath if it is specified
            line = f.readline()
            tokens = re.split('\s|=', line)
            if tokens[0].lower() == 'datapath':
                self.datapath = tokens[1]

            line = f.readline()
            while line.strip().lower() != 'atomic weight ratios':
                line = f.readline()

            # Second section: read the ZAID/atomic weight ratio pairs
            line = f.readline()
            while line.strip().lower() != 'directory':
                tokens = line.split()
                if len(tokens) > 1:
                    items = {int(tokens[i]): float(tokens[i+1])
                             for i in range(0, len(tokens), 2)}
                    self.atomic_weight_ratio.update(items)

                line = f.readline()

            # Third section: read the available data tables
            line = f.readline()
            while line:
                # Handle continuation lines
                while line[-2] == '+':
                    line += f.readline()
                    line = line.replace('+\n', '')

                # Store the entry if we need this table
                tokens = line.split()
                self.directory[tokens[0]] = XSDIRTable(line)

                line = f.readline()

    def export_to_xsdata(self, path='xsdata', table_names=None):
        """Create a Serpent XSDATA directory file.

        Parameters
        ----------
        path : str
            Path to file to write. Defaults to 'xsdata'.
        table_names : None, str, or iterable, optional
            Tables from the XSDIR file to write to the XSDATA file. If None,
            all of the entries are written. If str or iterable, only the
            entries matching the table names are written.
 
        """
        if table_names is None:
            table_names = self.directory.keys()
        else:
            table_names = set(table_names)

        # Classes of data included in the XSDATA file (continuous-energy
        # neutron, neutron dosimetry, thermal scattering, and continuous-energy
        # photoatomic)
        data_classes = {'c': 1, 'y': 2, 't': 3, 'p': 5}

        lines = []
        for name in table_names:
            table = self.directory.get(name)
            if table is None:
                msg = f'Could not find table {name} in {self.filename}.'
                raise ValueError(msg)

            # Check file format
            if table.file_type != 'ascii':
                msg = f'Unsupported file type {table.file_type} for {name}.'
                raise ValueError(msg)

            if self.datapath is None:
                # Set the access route as the datapath if it is specified;
                # otherwise, set the parent directory of XSDIR as the datapath
                if table.access_route is not None:
                    datapath = Path(table.access_route)
                else:
                    datapath = Path(self.filename).parent
            else:
                datapath = Path(self.datapath)

            # Get the full path to the ace library
            ace_path = datapath / table.file_name
            if not ace_path.is_file():
                raise ValueError(f'Could not find ACE file {ace_path}.')

            zaid, suffix = name.split('.')

            # Skip this table if it is not one of the data classes included in
            # XSDATA
            if suffix[-1] not in data_classes:
                continue

            # Get information about material and type of cross section data
            data_class = data_classes[suffix[-1]]
            if data_class == 3:
                ZA = 0
                m = 0
            else:
                zaid = int(zaid)
                _, element, Z, A, m = openmc.data.get_metadata(zaid, 'nndc')
                ZA = 1000*Z + A
                alias = f'{element}-'
                if A == 0:
                    alias += 'nat.'
                elif m == 0:
                    alias += f'{A}.'
                else:
                    alias += f'{A}m.'
                alias += suffix

            # Calculate the atomic weight
            if zaid in self.atomic_weight_ratio:
                atomic_weight = self.atomic_weight_ratio[zaid] * NEUTRON_MASS
            else:
                atomic_weight = table.atomic_weight_ratio * NEUTRON_MASS

            # Calculate the temperature in Kelvin
            temperature = table.temperature / K_BOLTZMANN * 1e6

            # Entry in the XSDATA file
            lines.append(f'{name} {name} {data_class} {ZA} {m} '
                         f'{atomic_weight:.8f} {temperature:.1f} 0 {ace_path}')

            # Also write an entry with the alias if this is not a thermal
            # scattering table
            if data_class != 3:
                lines.append(f'{alias} {name} {data_class} {ZA} {m} '
                             f'{atomic_weight:.8f} {temperature:.1f} 0 {ace_path}')

        # Write the XSDATA file
        with open(path, 'w') as f:
            f.write('\n'.join(lines))


    def get_tables(self, table_names):
        """Read ACE cross section tables from an XSDIR directory file.

        Parameters
        ----------
        table_names : str or iterable
            Names of the ACE tables to load

        Returns
        -------
        list of openmc.data.ace.Table
            ACE cross section tables

        """
        if isinstance(table_names, str):
            table_names = [table_names]
        else:
            table_names = set(table_names)

        tables = []
        for name in table_names:
            table = self.directory.get(name)
            if table is None:
                msg = f'Could not find table {name} in {self.filename}.'
                raise ValueError(msg)

            if self.datapath is None:
                # Set the access route as the datapath if it is specified;
                # otherwise, set the parent directory of XSDIR as the datapath
                if table.access_route is not None:
                    datapath = Path(table.access_route)
                else:
                    datapath = Path(self.filename).parent
            else:
                datapath = Path(self.datapath)

            # Get the full path to the ace library
            ace_path = datapath / table.file_name
            if not ace_path.is_file():
                raise ValueError(f'Could not find ACE file {ace_path}.')

            zaid, suffix = name.split('.')
            if re.match('(8[0-6]c)|(71[0-6]nc)', suffix):
                nuclide, _, _, _, _ = openmc.data.get_metadata(int(zaid))
                name = szax(nuclide, suffix)

            # Get the ACE table
            print(f'Converting table {name} from library {ace_path}...')
            tables.append(openmc.data.ace.get_table(ace_path, name))

        return tables


class XSDIRTable:
    """XSDIR description of a cross section table

    Parameters
    ----------
    line : str
        Cross section table description from an XSDIR directory file.

    Attributes
    ----------
    name : str
        ZAID of the table.
    atomic_weight_ratio : float
        Atomic mass ratio of the target nuclide.
    file_name : str
        Name of the library that contains the table.
    access_route : str
        Path to the library.
    file_type : {'ascii', 'binary'}
        File format.
    address : int
        For type 1 files the address is the line number in the file where the
        table starts. For type 2 files it is the record number of the first
        record of the table.
    table_length : int
        Length (total number of words) of the table.
    record_length : int
        For type 1 files the record length is unused. For type 2 files it is a
        multiple of the number of entries per record.
    entries_per_record : int
        For type 1 files this is unused. For type 2 files it is the number of
        entries per record.
    temperature : float
        Temperature in MeV at which a neutron table is processed. This is used
        only for neutron data.
    ptables : bool
        If true, it indicates a continuous-energy neutron nuclide has
        unresolved resonance range probability tables.

    """
    def __init__(self, line):
        entries = line.split()
        num_entries = len(entries)

        self.name = entries[0]
        self.atomic_weight_ratio = float(entries[1])
        self.file_name = entries[2]
        if entries[3] != '0':
            self.access_route = entries[3]
        else:
            self.access_route = None
        if entries[4] == '1':
            self.file_type = 'ascii'
        else:
            self.file_type = 'binary'
        self.address = int(entries[5])
        self.table_length = int(entries[6])
        if num_entries > 7:
            self.record_length = int(entries[7])
        else:
            self.record_length = 0
        if num_entries > 8:
            self.entries_per_record = int(entries[8])
        else:
            self.entries_per_record = 0
        if num_entries > 9:
            self.temperature = float(entries[9])
        else:
            self.temperature =  0.0
        if num_entries > 10:
            self.ptables = entries[10].lower() == 'ptable'
        else:
            self.ptables = False


def zaid(nuclide, suffix):
    """Return ZAID for a given nuclide and cross section suffix.

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

    # Serpent metastable convention
    if re.match('[0][3,6,9]c|[1][2,5,8]c', suffix):
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

    if re.match('(71[0-6]nc)', suffix):
        suffix = f'8{suffix[2]}c'

    return f'{1000*Z + A}.{suffix}'


def szax(nuclide, suffix):
    """Return SZAX for a given nuclide and cross section suffix.

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


def create_library(xsdir, table_names, hdf5_dir, xsdata_dir=None):
    """Convert the ACE data from the MCNP or Serpent distribution into an
    HDF5 library that can be used by OpenMC and create and XSDATA directory
    file for use with Serpent.

    Parameters
    ----------
    xsdir : str
        Path of the XSDIR directory file
    table_names : str or iterable
        Names of the ACE tables to convert
    hdf5_dir : str
        Directory to write the HDF5 library to
    xsdata_dir : str
        If specified, an XSDATA directory file containing entries for each of
        the table names provided will be written to this directory.

    """
    # Create data library
    data_lib = openmc.data.DataLibrary()

    # Load the XSDIR directory file
    xsdir = XSDIR(xsdir)

    # Get the ACE cross section tables
    tables = xsdir.get_tables(table_names)

    for table in tables:
        zaid, suffix = table.name.split('.')

        # Convert cross section data
        if suffix[-1] == 'c':
            match = '(7[0-4]c)|(8[0-6]c)|(71[0-6]nc)'
            scheme = 'mcnp' if re.match(match, suffix) else 'nndc'
            data = openmc.data.IncidentNeutron.from_ace(table, scheme)
        elif suffix[-1] == 'p':
            data = openmc.data.IncidentPhoton.from_ace(table)
        elif suffix[-1] == 't':
            data = openmc.data.ThermalScattering.from_ace(table)
        else:
            msg = ('Unknown data class: cannot convert cross section data '
                   f'from table {table.name}')
            raise ValueError(msg)

        # Export HDF5 files and register with library
        h5_file = Path(hdf5_dir) / f'{data.name}.h5'
        data.export_to_hdf5(h5_file, 'w')
        data_lib.register_file(h5_file)

    # Write cross_sections.xml
    data_lib.export_to_xml(Path(hdf5_dir) / 'cross_sections.xml')

    # Write the Serpent XSDATA file
    if xsdata_dir is not None:
        xsdir.export_to_xsdata(Path(xsdata_dir) / 'xsdata', table_names)


def read_results(code, filename):
    """Read the energy, mean, and standard deviation from the output

    Parameters
    ----------
    code : {'openmc', 'mcnp', 'serpent'}
        Code which produced the output file
    filename : str
        Path to the output file

    Returns
    -------
    energy : numpy.ndarray
        Energy bin values [MeV]
    mean : numpy.ndarray
        Sample mean of the tally
    std_dev : numpy.ndarray
        Sample standard deviation of the tally

    """
    if code == 'openmc':
        with openmc.StatePoint(filename) as sp:
            t = sp.get_tally(name='tally')
            energy = t.find_filter(openmc.EnergyFilter).bins[:,1]*1e-6
            mean = t.mean[:,0,0]
            std_dev = t.std_dev[:,0,0]

    elif code == 'mcnp':
        with open(filename, 'r') as f:
            text = f.read()
            p = text.find('1tally')
            p = text.find('energy', p) + 10
            q = text.find('total', p)
            t = np.fromiter(text[p:q].split(), float)
            t.shape = (len(t) // 3, 3)
            energy = t[1:,0]
            mean = t[1:,1]
            std_dev = t[1:,2]

    elif code == 'serpent':
        with open(filename, 'r') as f:
            text = re.split('\[|\]', f.read())
            t = np.fromiter(text[1].split(), float)
            t = t.reshape(len(t) // 12, 12)
            e = np.fromiter(text[3].split(), float)
            e = e.reshape(len(e) // 3, 3)
            energy = e[:,1]
            mean = t[:,10]
            std_dev = t[:,11]

    return energy, mean, std_dev
