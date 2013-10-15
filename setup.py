# -*- coding: utf-8 -*-
#_____________________________________________________________________________
#
# Copyright (c) 2012-2013, Berlin Institute of Technology
# All rights reserved.
#
# Developed by:	Philipp Meier <pmeier82@gmail.com>
#
#               Neural Information Processing Group (NI)
#               School for Electrical Engineering and Computer Science
#               Berlin Institute of Technology
#               MAR 5-6, Marchstr. 23, 10587 Berlin, Germany
#               http://www.ni.tu-berlin.de/
#
# Repository:   https://github.com/pmeier82/BOTMpy
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to
# deal with the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimers.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimers in the documentation
#   and/or other materials provided with the distribution.
# * Neither the names of Neural Information Processing Group (NI), Berlin
#   Institute of Technology, nor the names of its contributors may be used to
#   endorse or promote products derived from this Software without specific
#   prior written permission.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# WITH THE SOFTWARE.
#_____________________________________________________________________________
#
# Acknowledgements:
#   Philipp Meier <pmeier82@gmail.com>
#_____________________________________________________________________________
#
# Changelog:
#   * <iso-date> <identity> :: <description>
#_____________________________________________________________________________
#

__docformat__ = "restructuredtext"

## IMPORTS

# setup tools and cython (order matters!)
from setuptools import setup
from distutils.extension import Extension

try:
    from Cython.Distutils import build_ext
except ImportError:
    build_ext = None

# other imports
import numpy

## HELPERS

def find_version():
    """read version from botmpy.__init__"""

    try:
        with open("./botmpy/__init__.py", 'r') as fp:
            for line in fp:
                if line.startswith("__version__"):
                    return line.split()[-1][1:-1]
    except:
        return "0.0-null"

## CYTHON

ext_mod_list = []
if build_ext is not None:
    ext_mod_list.append(
        Extension(
            "botmpy.common.mcfilter.mcfilter_cy",
            ["botmpy/common/mcfilter/mcfilter_cy.pyx"],
            include_dirs=[numpy.get_include()]))

## MAIN

if __name__ == "__main__":
    setup(
        #main
        name="BOTMpy",
        version=find_version(),
        packages=["botmpy",
                  "botmpy.common",
                  "botmpy.common.datafile",
                  "botmpy.common.mcfilter",
                  "botmpy.nodes"],
        requires=["numpy", "scipy", "mdp", "sklearn"],
        zip_safe=False,

        # metadata
        author="Philipp Meier",
        author_email="pmeier82@gmail.com",
        maintainer="Philipp Meier",
        maintainer_email="pmeier82@gmail.com",
        description=open("README", 'r').readline().strip(),
        long_description=open("README", 'r').read(),
        license="University of Illinois/NCSA Open Source License",
        url="http://www.ni.tu-berlin.de",
        classifiers=[
            "Development Status :: 4 - Beta",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: University of Illinois/NCSA Open Source License",
            "Natural Language :: English",
            "Operating System :: OS Independent",
            "Programming Language :: Python",
            "Programming Language :: Cython",
            "Topic :: Scientific/Engineering :: Bio-Informatics",
            "Topic :: Scientific/Engineering :: Information Analysis"],

        # cython
        cmdclass={"build_ext": build_ext},
        ext_modules=ext_mod_list)

## EOF
