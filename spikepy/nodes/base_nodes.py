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


"""abstract base classes derived from MDP nodes"""
__docformat__ = 'restructuredtext'
__all__ = ['TrainingResetMixin', 'ResetNode']

##---IMPORTS

from mdp import Node, IsNotTrainableException, TrainingFinishedException

##---CLASSES

class TrainingResetMixin(object):
    """mixin class to allow mdp nodes to reset to training state after
    execution

    This is a mixin for subclasses of mdp.Node. Instance the class and put the

    node is a mdp.signal_node.Cumulator that can have its training phase
    reinitialised once a batch of cumulated data has been processed on. This
     is
    useful for online algorithms that derive parameters from the batch of data
    currently under consideration (Ex.: stochastic thresholding).
    """

    ## additional interface

    def reset(self):
        """reset handler, calling the reset hook and reseting to training
        phase"""

        # reset training capability
        self._train_phase = 0
        self._train_phase_started = False
        self._training = True

    def _reset(self):
        """reset hook, this is called each time a new training cycle begins"""

        pass

    ## mdp interface

    def execute(self, x, *args, **kwargs):
        """Process the data contained in 'x'.

        If the object is still in the training phase, the function
        'stop_training' will be called.
        'x' is a matrix having different variables on different columns
        and observations on the rows.

        By default, subclasses should overwrite _execute to implement
        their execution phase. The docstring of the '_execute' method
        overwrites this docstring.
        """

        try:
            self._pre_execution_checks(x)
            return self._execute(self._refcast(x), *args, **kwargs)
        finally:
            if self.is_trainable() is True:
                self.reset()

    def train(self, x, *args, **kwargs):
        """Update the internal structures according to the input data 'x'.

        'x' is a matrix having different variables on different columns
        and observations on the rows.

        By default, subclasses should overwrite _train to implement their
        training phase. The docstring of the '_train' method overwrites this
        docstring.

        Note: a subclass supporting multiple training phases should implement
        the *same* signature for all the training phases and document the
        meaning of the arguments in the '_train' method doc-string. Having
        consistent signatures is a requirement to use the node in a flow.
        """

        if not self.is_trainable():
            raise IsNotTrainableException("This node is not trainable.")

        if not self.is_training():
            err_str = "The training phase has already finished."
            raise TrainingFinishedException(err_str)

        # BEGIN - TRAINING RESET MIXIN ADDITION
        if self._train_phase_started is False:
            self._reset()
            # END - TRAINING RESET MIXIN ADDITION

        self._check_input(x)
        self._check_train_args(x, *args, **kwargs)

        self._train_phase_started = True
        self._train_seq[self._train_phase][0](self._refcast(x), *args,
                                              **kwargs)


class ResetNode(TrainingResetMixin, Node):
    pass

##---MAIN

if __name__ == '__main__':
    pass
