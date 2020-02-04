#!/usr/bin/env python
from distutils.core import setup
#from setuptools import setup

#setup(name="feasibgs",
#        description="package for investigating the feasibility of the DESI-BGS", 
#        packages=["feasibgs"])

__version__ = '0.1'

setup(name = 'eMaNu',
      version = __version__,
      description = 'Umulating MAssive NeUtrinos in galaxy clustering',
      author='ChangHoon Hahn',
      author_email='hahn.changhoon@gmail.com',
      url='',
      package_data={'emanu': ['dat/quijote_header_lookup.dat']},
      platforms=['*nix'],
      license='GPL',
      requires = ['numpy', 'matplotlib', 'scipy'],
      provides = ['emanu'],
      packages = ['emanu', 'emanu.sims']
      )
