# -*- coding: utf-8 -*-
#_____________________________________________________________________________
#
# Copyright (c) 2012 Berlin Institute of Technology
# All rights reserved.
#
# Developed by:	Philipp Meier <pmeier82@gmail.com>
#               Neural Information Processing Group (NI)
#               School for Electrical Engineering and Computer Science
#               Berlin Institute of Technology
#               MAR 5-6, Marchstr. 23, 10587 Berlin, Germany
#               http://www.ni.tu-berlin.de/
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

"""abstract base classes derived from MDP nodes"""
__docformat__ = 'restructuredtext'
__all__ = ['Node', 'ResetNode', 'TrainingResetMixin', 'PCANode']

##---IMPORTS

# MPD environ settings to stop it from frantically importing all sorts of
# packages

import os

os.environ['MDP_DISABLE_PARALLEL_PYTHON'] = True
os.environ['MDP_DISABLE_MONKEYPATCH_PP'] = True
os.environ['MDP_DISABLE_SHOGUN'] = True
os.environ['MDP_DISABLE_LIBSVM'] = True
os.environ['MDP_DISABLE_JOBLIB'] = True
os.environ['MDP_DISABLE_SKLEARN'] = True

# MPD DONE

from mdp import Node
from mdp.nodes import PCANode

##---CLASSES

class TrainingResetMixin(object):
    """allows :py:class:`mdp.Node` to reset to training state

    This is a mixin class for subclasses of :py:class:`mdp.Node`. To use it
    inherit from :py:class:`mdp.Node` and put this mixin as the first
    superclass.

    node is a mdp.signal_node.Cumulator that can have its training phase
    reinitialised once a batch of cumulated data has been processed on. This
    is useful for online algorithms that derive parameters from the batch of
    data currently under consideration (Ex.: stochastic thresholding).
    """

    ## additional interface

    def reset(self):
        """reset handler, calls the reset hook and resets to training phase"""

        # reset training capability
        self._train_phase = 0
        self._train_phase_started = False
        self._training = True
        self._reset()

    def _reset(self):
        pass


class ResetNode(TrainingResetMixin, Node):
    pass

##---MAIN

if __name__ == '__main__':
    pass
