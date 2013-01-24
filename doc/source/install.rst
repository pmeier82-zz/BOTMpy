.. _sec_install:

Download and Installation
=========================

BOTMpy can be used as a pure Python_ package. There are some additional Cython_
implementations for parts of the code that benefit from it, the Cython_ part
is optional but recommended, BOTMpy will run as a pure Python_ if desired.

Requirements
------------

For the basic Python_ version this stack is required:

* Python_ >= 2.7.3
* Scipy_ >= 0.9.0
* Numpy_ >= 1.6.1
* scikit-learn_ >= 0.12.1
* mdp_ >= 3.3

Please follow the links to the respective websites for instructions on how
to install them if they are not present on your computer. For Python_, Scipy_
and Numpy_ it is advised to use install mechanism appropriate to your operating
system, and not use the python packaging mechanism like pip_ and easy_install_.

Download
--------

The easiest way to get BOTMpy is from the Python Package Index (PyPI_).
If you have pip_ installed:
::

  $ pip install botmpy

Alternatively, if you have setuptools_::

  $ easy_install botmpy

Alternatively, you can get the latest version directly from GitHub at
https://github.com/pmeier82/BOTMpy.::

  $ git clone https://github.com/pmeier82/BOTMpy.git

The master branch (selected by default) always contains the current stable
version. All the other branches will be feature branches or integration
branches and are not recommended for general usage.

Installation
------------

After you have acquired a copy of the package from GitHub you need to install
it on your system by calling::

  $ python setup.py install

This is not necessary when you install with either pip_ or easy_install_.

Cython Speedup
--------------

To use the Cython_ implementations you will need a current Cython_ package
(>= 0.15.1) and the sources for Python_ and Numpy_ to build the extention
modules. The relevant packages for debian are called *pythonX.Y-dev* and
*python-numpy-dev*.

If the required packages and Cython_ is present on your system during
installation, the cython extensions will be build automatically. There is no
change to the interface specific to the cython implementations that the user
needs to take care of.

.. ############################################################################
.. link targets

.. _python: http://python.org
.. _cython: http://cython.org
.. _scipy: http://scipy.org
.. _numpy: http://numpy.org
.. _mdp: http://mdp-toolkit.sourceforge.net
.. _scikit-learn: http://scikit-learn.org/stable
.. _sklearn: http://scikit-learn.org/stable
.. _pip: http://www.pip-installer.org
.. _pypi: http://pypi.python.org
.. _setuptools: http://pypi.python.org/pypi/setuptools
.. _easy_install: http://packages.python.org/distribute/easy_install.html
