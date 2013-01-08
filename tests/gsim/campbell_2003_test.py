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
from nhlib.gsim.campbell_2003 import Campbell2003, Campbell2003SHARE

from tests.gsim.utils import BaseGSIMTestCase

import numpy

# Test data generated from OpenSHA implementation.

class Campbell2003TestCase(BaseGSIMTestCase):
    GSIM_CLASS = Campbell2003

    def test_mean(self):
        self.check('C03/C03_MEAN.csv',
                   max_discrep_percentage=0.1)

    def test_std_total(self):
        self.check('C03/C03_STD_TOTAL.csv',
                   max_discrep_percentage=0.1)

class Campbell2003SHARETestCase(BaseGSIMTestCase):
    GSIM_CLASS = Campbell2003SHARE

    def test_mean(self):
        self.check('C03/C03SHARE_MEAN.csv',
                   max_discrep_percentage=0.1)

    def test_std_total(self):
        self.check('C03/C03SHARE_STD_TOTAL.csv',
                   max_discrep_percentage=0.1)
