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


"""interfaces for reading (multi-channeled) data from various file formats"""

__docformat__ = "restructuredtext"
__all__ = ["DataFileError", "DataFileMetaclass", "DataFile"]

## IMPORTS

import scipy as sp
import inspect as _inspect

## CLASSES

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
    DOC_METHODS = ["_close", "_closed", "_filename", "_get_data"]

    def __new__(cls, classname, bases, members):
        # select private methods that can overwrite the docstring
        wrapper_names = []
        priv_infos = []
        for privname in cls.DOC_METHODS:
            if privname in members:
                priv_info = cls._get_infodict(members[privname])
                # if the docstring is empty, don't overwrite it
                if not priv_info["doc"]:
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
            priv_info["name"] = wrapper_name
            # Preserve the signature only if it does not end with kwargs
            # (this is important for binodes).
            # Note that this relies on the exact name 'kwargs', if this causes
            # problems we could switch to looking for ** in the signature.
            if not wrapped_info["argnames"][-1] == "kwargs":
                priv_info["signature"] = wrapped_info["signature"]
                priv_info["argnames"] = wrapped_info["argnames"]
                priv_info["defaults"] = wrapped_info["defaults"]
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
                                           formatvalue=lambda value: "")[1:-1]
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
        wrapped_func = eval(src, {"_wrapper_class_": cls})
        wrapped_func.__name__ = wrapper_infodict['name']
        wrapped_func.__doc__ = wrapper_infodict['doc']
        wrapped_func.__module__ = wrapper_infodict['module']
        wrapped_func.__dict__.update(wrapper_infodict['dict'])
        wrapped_func.func_defaults = wrapper_infodict['defaults']
        wrapped_func.undecorated = wrapper_infodict
        return wrapped_func


class DataFile(object):
    """abstract data file interface

    This is an abstract datafile interface. Implementations should implement
    the private interface as required for that file format! A `DataFile`
    manages a file handle to the physical file and should try to hold
    its contents in memory, and use buffering to reduce filesystem load.

    All public methods belong to the interface and relay to their private
    counter parts, which are provided by the file format implementation.
    """

    __metaclass__ = DataFileMetaclass

    ## constructor

    def __init__(self, filename=None, dtype=None, **kwargs):
        """
        :type filename: str
        :param filename: Path on the filesystem to the physical file.
        :type dtype: dtype resolveable
        :param dtype: A numpy dtype
            Default=float32
        """

        # members
        self.dtype = sp.dtype(dtype or sp.float32)
        self.fp = None

        # initialize
        self._initialize_file(filename, **kwargs)

    ## public interface

    def close(self):
        """close this `DataFile`"""

        self._close()

    def closed(self):
        """return the `closed` status"""

        return self._closed()

    def filename(self):
        """return the filename of the physical file"""

        return self._filename()

    def get_data(self, **kwargs):
        """returns a numpy array of the data with samples on the rows and
        channels on the columns. channels may be selected via the channels
        parameter.

        :rtype: ndarray
        :returns: requested data
        """

        if self._closed():
            raise DataFileError('Archive is closed!')
        return self._get_data(**kwargs)

    ## private interface - to be implemented in subclasses

    def _initialize_file(self, filename, **kwargs):
        """initialize the file handle

        :type filename: str
        :param filename: path on the local filesystem
        :keyword kwargs: keywords for subclass
        """

        raise NotImplementedError

    def _close(self):
        """close the file handle"""

        raise NotImplementedError

    def _closed(self):
        """return the file handle status

        :rtype: bool
        :returns: file handle close state
        """

        raise NotImplementedError

    def _filename(self):
        """return the filesystem path of the file handle

        :rtype: str
        :returns: filesystem path of the file handle
        """

        raise NotImplementedError

    def _get_data(self, **kwargs):
        """return ndarray of data according to keywords passed

        :keyword kwargs: keywords for subclass
        :rtype: ndarray
        :returns: requested data
        """

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
