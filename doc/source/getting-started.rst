.. _installation:

Installation
============

BOTMpy can be used as a pure Python_ package. There are some additional Cython_
implementations for parts of the code that benefit from it. The Cython_ part
is optional but recommended, BOTMpy will run as a pure Python_ if desired.

Requirements
------------

For the basic Python_ version this stack is required:

* Python_ >= 2.7.3
* Scipy_ >= 0.9.0
* Numpy_ >= 1.6.1
* scikit-learn_ >= 0.10
* mdp_ >= 3.3

Please follow the links to the respective websites for instructions on how
to install them if they are not present on your computer. For Python_, Scipy_
and Numpy_ it is advised to use install mechanism appropriate to your operating
system, and not use the python packaging mechanism like pip_ and easy_install_.

To use the Cython_ implementations you will need a current Cython_ package
(>= 0.15.1) and the sources for Python_ and Numpy_ to build the extention
modules. For details on an installation with Cython_ enabled please read
:ref:`Cython Speedup <cython_enabled>`.

Download and Installation
=========================
The easiest way to get BOTMpy is from the Python Package Index.
If you have pip_ installed:
::

  $ pip install botmpy

Alternatively, if you have setuptools_::

  $ easy_install botmpy

Alternatively, you can get the latest version directly from GitHub at
https://github.com/pmeier82/BOTMpy.

The master branch (selected by default) always contains the current stable
version. All the other branches will be feature branches or integration
branches and are not recommended for general usage.

After you have acquired a copy of the package from GitHub you need to install
it on your system by calling::

  $ python setup.py install

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
.. _setuptools: http://pypi.python.org/pypi/setuptools
.. _easy_install: http://packages.python.org/distribute/easy_install.html
