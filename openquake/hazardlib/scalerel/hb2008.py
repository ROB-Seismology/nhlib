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
"""
Module :mod:`nhlib.scalerel.hb2008` implements :class:`HB2008`.
"""


from openquake.hazardlib.scalerel.base import BaseMSR


class HB2008(BaseMSR):
	"""
	Hanks, T. C., and W. H. Bakun, M-log A observations of recent large
	earthquakes, Bull. Seismol. Soc. Am., 98 , 490, 2008.
	
	Hanks, T. C., and W. H. Bakun, A blinear source-scaling model for M-log A
	observations of continental earthquakes, Bull. Seismol. Soc. Am., 92 , 1841,
	2002.
	
	Implements magnitude scaling relationship
	"""
	def get_median_area(self, mag, rake):
		if mag <= 6.71:
			return 10**(mag-3.98)
		else:
			return 10**(3/4*(mag-3.07))

