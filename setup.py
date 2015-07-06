#!/usr/bin/env python

__author__ = 'Aleksandar Savkov'

from distutils.core import setup

setup(name='CRFSuiteTagger',
      version='0.1',
      description='A multi-purpose sequential tagger wrapped around CRFSuite',
      author='Aleksandar Savkov',
      author_email='aleksandar@savkov.eu',
      url='https://github.com/savkov/CRFSuiteTagger',
      package_dir={'': 'src'},
      packages=['crfsuitetagger']
     )