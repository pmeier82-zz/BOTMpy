Requirements
============
BOTMpy can be used as a pure Python_ package. There are also some Cython_
implementations for parts of the code that benefit from it. The Cython_ part
is optional, BOTMpy will run as a pure PYthon_ package with the following
requirements:

* Python_ >= 2.7.3
* scipy_ >= 0.9.0
* numpy_ >= 1.6.1
* sklearn_ >= 0.10
* mdp_ >= 3.3

To use the Cython_ implementations you will need a current Cython_ package
and the sources for Python_ and numpy_. For details on an installation with
Cython_ enables please read :ref:`cython_install`
requirements and install
For a significant speedup of the signal processing algorithms :mod:`cython`
implementations are provided. These are optional but recommended. To use them
you will further need:


Please see the respective websites for instructions on how to install them if
they are not present on your computer.

.. note::
    The current version of Neo in the Python Package Index contains
    some bugs that prevent it from working properly with spykeutils in some
    situations. Please install the latest version directly from GitHub:
    https://github.com/rproepp/python-neo

    You can download the repository from the GitHub page or clone it using
    git and then install from the resulting folder::

    $ python setup.py install

Download and Installation
=========================
The easiest way to get spykeutils is from the Python Package Index.
If you have pip_ installed::

$ pip install spykeutils

Alternatively, if you have setuptools_::

$ easy_install spykeutils

Alternatively, you can get the latest version directly from GitHub at
https://github.com/rproepp/spykeutils.

The master branch (selected by default) always contains the current stable
version. If you want the latest development version (not recommended unless
you need some features that do not exist in the stable version yet), select
the develop branch. You can download the repository from the GitHub page
or clone it using git and then install from the resulting folder::

$ python setup.py install
