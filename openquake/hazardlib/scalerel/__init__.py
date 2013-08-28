# The Hazard Library
# Copyright (C) 2012 GEM Foundation
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
Package :mod:`openquake.hazardlib.scalerel` contains base classes and
implementations of magnitude-area and area-magnitude scaling relationships.
"""
import os
import inspect
import importlib
from collections import OrderedDict
from openquake.hazardlib.scalerel.base import BaseMSR, BaseASR


from openquake.hazardlib.scalerel.peer import PeerMSR
from openquake.hazardlib.scalerel.wc1994 import WC1994


def _get_available_class(base_class):
    '''
    Return an ordered dictionary with the available classes in the
    scalerel submodule with classes that derives from `base_class`,
    keyed by class name.
    '''
    gsims = {}
    for fname in os.listdir(os.path.dirname(__file__)):
        if fname.endswith('.py'):
            modname, _ext = os.path.splitext(fname)
            mod = importlib.import_module(
                'openquake.hazardlib.scalerel.' + modname)
            for cls in mod.__dict__.itervalues():
                if inspect.isclass(cls) and issubclass(cls, base_class) \
                        and cls != base_class:
                    gsims[cls.__name__] = cls
    return OrderedDict((k, gsims[k]) for k in sorted(gsims))


def get_available_magnitude_scalerel():
    '''
    Return an ordered dictionary with the available Magnitude ScaleRel
    classes, keyed by class name.
    '''
    return _get_available_class(BaseMSR)


def get_available_area_scalerel():
    '''
    Return an ordered dictionary with the available Magnitude ScaleRel
    classes, keyed by class name.
    '''
    return _get_available_class(BaseASR)


def get_available_scalerel():
    '''
    Return an ordered dictionary with the available ScaleRel classes,
    keyed by class name.
    '''
    ret = get_available_area_scalerel()
    ret.update(get_available_magnitude_scalerel())

    return ret
