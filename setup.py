#!/usr/bin/env python

from setuptools import setup


setup(
    name='validation',
    author='The OpenMC Development Team',
    author_email='openmc-dev@googlegroups.com',
    description='A collection of validation scripts for OpenMC',
    url='https://github.com/openmc-dev/validation',
    packages=['validation'],
    entry_points={
        'console_scripts': [
            'openmc-validate-neutron-physics=validation.neutron_physics:main',
            'openmc-validate-photon-physics=validation.photon_physics:main',
            'openmc-validate-photon-production=validation.photon_production:main'
        ]
    }
)
