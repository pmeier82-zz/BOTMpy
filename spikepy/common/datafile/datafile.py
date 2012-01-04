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


"""interfaces for reading (multichanneled) data from files"""
__docformat__ = 'restructuredtext'
__all__ = ['DataFileError', 'DataFileMetaclass', 'DataFile']

##---IMPORTS

import scipy as sp
import inspect as _inspect

##---CLASSES

class DataFileError(Exception):
    pass


class DataFileMetaclass(type):
    """This Metaclass is meant to overwrite doc strings of methods with
    those defined in the corresponding private methods. Corresponding means
    there is a public interface entry method and a private implementation
    method. For the instances object, the call to the public method should
    present the private methods docstring.

    This makes it possible for subclasses implementing the interface to
    document the usage of public methods, without the need to overwrite
    the ancestor's methods.
    """

    # methods that can overwrite doc:
    DOC_METHODS = ['_close', '_closed', '_filename', '_get_data']

    def __new__(cls, classname, bases, members):
        # select private methods that can overwrite the docstring
        wrapper_names = []
        priv_infos = []
        for privname in cls.DOC_METHODS:
            if privname in members:
                priv_info = cls._get_infodict(members[privname])
                # if the docstring is empty, don't overwrite it
                if not priv_info['doc']:
                    continue
                    # get the name of the corresponding public method
                pubname = privname[1:]
                # If the public method has been overwritten in this
                # subclass, then keep it.
                # This is also important because we use super in the wrapper
                # (so the public method in this class would be missed).
                if pubname not in members:
                    wrapper_names.append(pubname)
                    priv_infos.append(priv_info)
        new_cls = super(DataFileMetaclass, cls).__new__(cls, classname,
                                                        bases, members)
        # now add the wrappers
        for wrapper_name, priv_info in zip(wrapper_names, priv_infos):
            # Note: super works because we never wrap in the defining class
            wrapped_method = getattr(super(new_cls, new_cls), wrapper_name)
            wrapped_info = cls._get_infodict(wrapped_method)
            priv_info['name'] = wrapper_name
            # Preserve the signature only if it does not end with kwargs
            # (this is important for binodes).
            # Note that this relies on the exact name 'kwargs', if this causes
            # problems we could switch to looking for ** in the signature.
            if not wrapped_info['argnames'][-1] == "kwargs":
                priv_info['signature'] = wrapped_info['signature']
                priv_info['argnames'] = wrapped_info['argnames']
                priv_info['defaults'] = wrapped_info['defaults']
            setattr(new_cls, wrapper_name,
                    cls._wrap_method(priv_info, new_cls))
        return new_cls

    # The next two functions (originally called get_info, wrapper)
    # are adapted versions of functions in the
    # decorator module by Michele Simionato
    # Version: 2.3.1 (25 July 2008)
    # Download page: http://pypi.python.org/pypi/decorator
    # Note: Moving these functions to utils would cause circular import.

    @staticmethod
    def _get_infodict(func):
        """
        Returns an info dictionary containing:
        - name (the name of the function : str)
        - argnames (the names of the arguments : list)
        - defaults (the values of the default arguments : tuple)
        - signature (the signature : str)
        - doc (the docstring : str)
        - module (the module name : str)
        - dict (the function __dict__ : str)

        >>> def f(self, x=1, y=2, *args, **kw): pass

        >>> info = getinfo(f)

        >>> info["name"]
        'f'
        >>> info["argnames"]
        ['self', 'x', 'y', 'args', 'kw']

        >>> info["defaults"]
        (1, 2)

        >>> info["signature"]
        'self, x, y, *args, **kw'
        """
        regargs, varargs, varkwargs, defaults = _inspect.getargspec(func)
        argnames = list(regargs)
        if varargs:
            argnames.append(varargs)
        if varkwargs:
            argnames.append(varkwargs)
        signature = _inspect.formatargspec(regargs,
                                           varargs,
                                           varkwargs,
                                           defaults,
                                           formatvalue=lambda value:"")[1:-1]
        return dict(name=func.__name__, argnames=argnames,
                    signature=signature,
                    defaults=func.func_defaults, doc=func.__doc__,
                    module=func.__module__, dict=func.__dict__,
                    globals=func.func_globals, closure=func.func_closure)

    @staticmethod
    def _wrap_function(original_func, wrapper_infodict):
        """Return a wrapped version of func.

        original_func -- The function to be wrapped.
        wrapper_infodict -- The infodict to be used for constructing the
            wrapper.
        """
        src = ("lambda %(signature)s: _original_func_(%(signature)s)" %
               wrapper_infodict)
        wrapped_func = eval(src, dict(_original_func_=original_func))
        wrapped_func.__name__ = wrapper_infodict['name']
        wrapped_func.__doc__ = wrapper_infodict['doc']
        wrapped_func.__module__ = wrapper_infodict['module']
        wrapped_func.__dict__.update(wrapper_infodict['dict'])
        wrapped_func.func_defaults = wrapper_infodict['defaults']
        wrapped_func.undecorated = wrapper_infodict
        return wrapped_func

    @staticmethod
    def _wrap_method(wrapper_infodict, cls):
        """Return a wrapped version of func.

        wrapper_infodict -- The infodict to be used for constructing the
            wrapper.
        cls -- Class to which the wrapper method will be added, this is used
            for the super call.
        """
        src = ("lambda %(signature)s: " % wrapper_infodict +
               "super(_wrapper_class_, _wrapper_class_)." +
               "%(name)s(%(signature)s)" % wrapper_infodict)
        wrapped_func = eval(src, {"_wrapper_class_":cls})
        wrapped_func.__name__ = wrapper_infodict['name']
        wrapped_func.__doc__ = wrapper_infodict['doc']
        wrapped_func.__module__ = wrapper_infodict['module']
        wrapped_func.__dict__.update(wrapper_infodict['dict'])
        wrapped_func.func_defaults = wrapper_infodict['defaults']
        wrapped_func.undecorated = wrapper_infodict
        return wrapped_func


class DataFile(object):
    """abstract data file interface

    This is an abstract datafile interface. All implementations should
    implement
    the whole private interface to prive and handlde tha data! The datafile
    manages an file handle to the native file and reads in the whole file into
    memory, where it is buffered to easy further access on the data.

    All public methods are mainly relays to their private counter parts, which
    should implement the functionality as necessary.
    """

    __metaclass__ = DataFileMetaclass

    ## constructor

    def __init__(self, filename=None, dtype=sp.float32, **kwargs):
        """
        :Parameters:
            filename : str
                Path to the file to load.
            dtype : dtype
                A numpy dtype
        """

        # members
        self.dtype = sp.dtype(dtype)
        self.fp = None

        # initialize
        self._initialize_file(filename, **kwargs)

    ## public interface

    def close(self):
        """close the datafile"""
        self._close()

    def closed(self):
        """return the closed status"""
        return self._closed()

    def filename(self):
        """return the filename"""
        return self._filename()

    def get_data(self, **kwargs):
        """returns a numpy array of the data with samples on the rows and
        channels on the columns. channels may be selected via the channels
        parameter.
        """

        if self.closed():
            raise DataFileError('Archive is closed!')
        return self._get_data(**kwargs)

    ## private interface - to be implemented in subclasses

    def _initialize_file(self, filename, **kwargs):
        """initialize the file wrapper

        :Parameters:
            filename : str
                Valid path on the local filesystem.
            kwargs : dict
                Keywords for subclasses
        """

        raise NotImplementedError

    def _close(self):
        """close the underlying file handle"""

        raise NotImplementedError

    def _closed(self):
        """return the underlying file status"""

        raise NotImplementedError

    def _filename(self):
        """return the filename of the underlying file"""

        raise NotImplementedError

    def _get_data(self, **kwargs):
        """return ndarray of data according to keywords passed"""

        raise NotImplementedError

    ## special methods

    def __str__(self):
        return '%s[%s]' % (self.__class__.__name__, self.filename())

    def __enter__(self):
        return self

    def __exit__(self, t, v, tb):
        if t is not None:
            # exeption occured
            pass
        self.close()

    def __del__(self):
        if self.fp is not None:
            self.close()

##---MAIN

if __name__ == '__main__':
    pass
