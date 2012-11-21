# nhlib: A New Hazard Library
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
from nhlib.gsim.atkinson_boore_2006 import AtkinsonBoore2006
from nhlib.gsim.base import SitesContext, RuptureContext, DistancesContext
from nhlib.imt import PGA
from nhlib.const import StdDev

from tests.gsim.utils import BaseGSIMTestCase

import numpy


class AtkinsonBoore2006TestCase(BaseGSIMTestCase):
    GSIM_CLASS = AtkinsonBoore2006
    
    # Test data generated from Fortran implementation
    # of Dave Boore (http://www.daveboore.com/pubs_online.html)

    def test_mean(self):
        self.check('AB06/AB06_MEAN.csv',
                    max_discrep_percentage=0.9)
                    
    def test_std_total(self):
        self.check('AB06/AB06_STD_TOTAL.csv',
                    max_discrep_percentage=0.1)

    def test_zero_distance(self):
        # test the calculation in case of zero rrup distance (for rrup=0
        # the equations have a singularity). In this case the
        # method should return values equal to the ones obtained by
        # replacing 0 values with 1
        sctx = SitesContext()
        rctx = RuptureContext()
        dctx = DistancesContext()
        setattr(sctx, 'vs30', numpy.array([500.0, 2500.0]))
        setattr(rctx, 'mag', 5.0)
        setattr(dctx, 'rrup', numpy.array([0.0, 0.2]))
        mean_0, stds_0 = self.GSIM_CLASS().get_mean_and_stddevs(
            sctx, rctx, dctx, PGA(), [StdDev.TOTAL])
        setattr(dctx, 'rrup', numpy.array([1.0, 0.2]))
        mean_01, stds_01 = self.GSIM_CLASS().get_mean_and_stddevs(
            sctx, rctx, dctx, PGA(), [StdDev.TOTAL])
        numpy.testing.assert_array_equal(mean_0, mean_01)
        numpy.testing.assert_array_equal(stds_0, stds_01)
