.. BOTMpy documentation master file, created by
   sphinx-quickstart on Wed Jan  9 14:48:25 2013.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

######################################################################
BOTMpy : spike sorting using Bayes Optimal Template Matching in Python
######################################################################

.. sidebar:: Details

    :Release: |release|
    :Date: |today|
    :Authors: **Philipp Meier**
    :Target: users and developers
    :status: mature

This package implements spike sorting with the BOTM algorithm. It is intended
as a tool for scientists in neuroscience to process time series data. It can be
used to detect and classify action potentials of distinct single cells in
voltage traces.

For further information on the general topic of spike sorting:
:doc:`spike-sorting`

For information on the details of the linear filters and the BOTM method:
:doc:`botm`

For pointers how to get started read the :ref:`` section.


Contents:
---------

.. toctree::
   :maxdepth: 3

   install
   cython
   spike-sorting
   botm


APIDocs:
--------

.. toctree::
   :maxdepth: 6

   apidoc/botmpy

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
