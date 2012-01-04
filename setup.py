# -*- coding: utf-8 -*-
#_____________________________________________________________________________
#
# Copyright (C) 2011 by Philipp Meier, Felix Franke and
# Berlin Institute of Technology
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
# ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#_____________________________________________________________________________
#
# Affiliation:
#   Bernstein Center for Computational Neuroscience (BCCN) Berlin
#     and
#   Neural Information Processing Group
#   School for Electrical Engineering and Computer Science
#   Berlin Institute of Technology
#   FR 2-1, Franklinstrasse 28/29, 10587 Berlin, Germany
#   Tel: +49-30-314 26756
#_____________________________________________________________________________
#
# Acknowledgements:
#   This work was supported by Deutsche Forschungs Gemeinschaft (DFG) with
#   grant GRK 1589/1
#     and
#   Bundesministerium für Bildung und Forschung (BMBF) with grants 01GQ0743
#   and 01GQ0410.
#_____________________________________________________________________________
#


"""setuptools script"""
__docformat__ = 'restructuredtext'


##--- IMPORTS

from ez_setup import use_setuptools

use_setuptools()
from setuptools import setup, find_packages

##--- SETUP

setup(
    # basic info
    name='SpikePy',
    version='0.1dev',
    description='A python package for spike sorting and '\
                'electro-physiological data processing',
    long_description="""yet to come :/""",
    # PyPI info
    author='Philipp Meier',
    author_email='pmeier82@googlemail.com',
    url='http://www.ni.tu-berlin.de/'\
        'menue/research/projects/and/spike_sorting_and_spike_train_analysis',
    license='MIT-ish',
    # dependancies
    packages=['spikepy', 'spikepy.nodes', 'spikepy.ntrode', 'spikepy.common',
              'spikepy.common.datafile'],
    package_data={
        '':['*.txt'],
        'doc':['*.*']
    },

    install_requires=[
        'scipy',
        'scikits.learn',
        'mdp',
        'tables'
    ],
    zip_safe=False)
