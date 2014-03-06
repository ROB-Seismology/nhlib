## The Hazard Library
## Copyright (C) 2012 GEM Foundation
##
## This program is free software: you can redistribute it and/or modify
## it under the terms of the GNU Affero General Public License as
## published by the Free Software Foundation, either version 3 of the
## License, or (at your option) any later version.
##
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU Affero General Public License for more details.
##
## You should have received a copy of the GNU Affero General Public License
## along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
Module exports :class:`RietbrockEtAl2013`.
"""
from __future__ import division

import numpy as np

from scipy.constants import g

from openquake.hazardlib.gsim.base import GMPE, CoeffsTable
from openquake.hazardlib import const
from openquake.hazardlib.imt import PGA, PGV, SA


class RietbrockEtAl2013(GMPE):
	"""
	Implements GMPE developed by Rietbrock et al. (2013). Based on weak-motion
	events (2 < Ml < 4.7). Model1 (self-similar) for the stress parameter is used.
	"""
	
	#: Supported tectonic region type is 'Stable Shallow Crust' because the
	#: equations have been derived from data from the UK.
	DEFINED_FOR_TECTONIC_REGION_TYPE = const.TRT.STABLE_CONTINENTAL

	#: Set of :mod:`intensity measure types <openquake.hazardlib.imt>`
	#: this GSIM can calculate. A set should contain classes from module
	#: :mod:`openquake.hazardlib.imt`.
	DEFINED_FOR_INTENSITY_MEASURE_TYPES = set([
		PGA,
		PGV,
		SA
	])
	
	#: Supported intensity measure component is the geometric mean of two
	#: horizontal components
	DEFINED_FOR_INTENSITY_MEASURE_COMPONENT = const.IMC.AVERAGE_HORIZONTAL

	#: Supported standard deviation types are inter-event, intra-event
	#: and total.
	DEFINED_FOR_STANDARD_DEVIATION_TYPES = set([
		const.StdDev.TOTAL,
		const.StdDev.INTER_EVENT,
		const.StdDev.INTRA_EVENT
	])
	
	REQUIRES_SITES_PARAMETERS = set()

	#: Required rupture parameters are magnitude (eq. 10, page 63).
	REQUIRES_RUPTURE_PARAMETERS = set(('mag',))

	#: Required distance measure is Rjb (eq. 11, page 63).
	REQUIRES_DISTANCES = set(('rjb', ))

	def get_mean_and_stddevs(self, sites, rup, dists, imt, stddev_types):
		"""
		See :meth:`superclass method
		<.base.GroundShakingIntensityModel.get_mean_and_stddevs>`
		for spec of input and result values.
		"""
		# extracting dictionary of coefficients specific to required
		# intensity measure type.
		C = self.COEFFS[imt]
		
		R = self._get_r(C, dists)
		
		log10_mean = (
			self._compute_1(C, rup) +
			self._compute_2(C, rup, R) +
			self._compute_3(C, R)
			)

		# Convert units to g,
		# but only for PGA and SA (not PGV):
		if isinstance(imt, (PGA, SA)):
			mean = np.log((10.0 ** (log10_mean - 2.0)) / g)
		else:
			# PGV:
			mean = np.log(10.0 ** log10_mean)

		istddevs = self._get_stddevs(C, stddev_types,
									 num_sites=len(dists.rjb))

		stddevs = np.log(10 ** np.array(istddevs))

		return mean, stddevs
	
	def _get_stddevs(self, C, stddev_types, num_sites):
		"""
		Return standard deviations.
		"""
		stddevs = []
		for stddev_type in stddev_types:
			assert stddev_type in self.DEFINED_FOR_STANDARD_DEVIATION_TYPES
			if stddev_type == const.StdDev.TOTAL:
				stddevs.append(C['st'] + np.zeros(num_sites))
			elif stddev_type == const.StdDev.INTRA_EVENT:
				stddevs.append(C['sw'] + np.zeros(num_sites))
			elif stddev_type == const.StdDev.INTER_EVENT:
				stddevs.append(C['sb'] + np.zeros(num_sites))
		return stddevs
	
	def _compute_1(self, C, rup):
		"""
		Compute part 1 of equation 10 described on p. 63.
		"""
		return C['c1'] + (C['c2'] * rup.mag) + (C['c3'] * (rup.mag ** 2))
	
	def _compute_2(self, C, rup, R):
		"""
		Compute part 2 of equation 10 described on p. 63:
		"""
		return (
			(C['c4'] + C['c5'] * rup.mag) * self._get_f0(R) +
			(C['c6'] + C['c7'] * rup.mag) * self._get_f1(R) +
			(C['c8'] + C['c9'] * rup.mag) * self._get_f2(R)
			)
	
	def _compute_3(self, C, R):
		"""
		Compute part 3 of equation 10 described on p. 63.
		"""
		return C['c10'] * R
	
	def _get_f0(self, R):
		"""
		Compute f0 from equation 12a described on p. 63.
		"""
		r0 = 10
		f0 = np.where(R <= r0, np.log10(r0 / R), 0)
		return f0
	
	def _get_f1(self, R):
		"""
		Compute f2 from equation 12b described on p. 63.
		"""
		r1 = 50
		f1 = np.where(R <= r1, np.log10(R), np.log10(r1))
		return f1
	
	def _get_f2(self, R):
		"""
		Compute f2 from equation 12c described on p. 63.
		"""
		r2 = 100
		f2 = np.where(R <= r2, 0, np.log10(R / r2))
		return f2
	
	def _get_r(self, C, dists):
		"""
		Compute r from equation 11 described on p. 63.
		"""
		return np.sqrt(dists.rjb**2 + C['c11']**2)
	
	## table for model 1 (self-similar) for stress parameter, p. 64.
	COEFFS = CoeffsTable(sa_damping=5, table="""\
	IMT	 c1     c2     c3      c4      c5     c6      c7     c8      c9     c10       c11    st    sb    sw
	PGV -2.9598 0.9039 -0.0434 -1.6243 0.1987 -1.6511 0.1654 -2.4308 0.0851 -0.001472 1.7736 0.347 0.311 0.153
	PGA -0.0135 0.6889 -0.0488 -1.8987 0.2151 -1.9063 0.1740 -2.0131 0.0887 -0.002747 1.5473 0.436 0.409 0.153
	0.03 0.8282 0.5976 -0.0418 -2.1321 0.2159 -2.0530 0.1676 -1.5148 0.1163 -0.004463 1.1096 0.449 0.417 0.167
	0.04 0.4622 0.6273 -0.0391 -1.7242 0.1644 -1.6849 0.1270 -1.4513 0.0910 -0.004355 1.1344 0.445 0.417 0.155
	0.05 0.2734 0.6531 -0.0397 -1.5932 0.1501 -1.5698 0.1161 -1.5350 0.0766 -0.003939 1.1493 0.442 0.416 0.149
	0.06 0.0488 0.6945 -0.0420 -1.4913 0.1405 -1.4807 0.1084 -1.6563 0.0657 -0.003449 1.2154 0.438 0.414 0.143
	0.08 -0.2112 0.7517 -0.0460 -1.4151 0.1340 -1.4130 0.1027 -1.7821 0.0582 -0.002987 1.2858 0.433 0.410 0.140
	0.10 -0.5363 0.8319 -0.0521 -1.3558 0.1296 -1.3579 0.0985 -1.8953 0.0520 -0.002569 1.3574 0.428 0.405 0.138
	0.12 -0.9086 0.9300 -0.0597 -1.3090 0.1264 -1.3120 0.0948 -1.9863 0.0475 -0.002234 1.4260 0.422 0.399 0.138
	0.16 -1.3733 1.0572 -0.0698 -1.2677 0.1237 -1.2684 0.0910 -2.0621 0.0434 -0.001944 1.4925 0.416 0.392 0.139
	0.20 -1.9180 1.2094 -0.0819 -1.2315 0.1213 -1.2270 0.0872 -2.1196 0.0396 -0.001708 1.5582 0.409 0.384 0.141
	0.25 -2.5107 1.3755 -0.0949 -1.1992 0.1189 -1.1881 0.0833 -2.1598 0.0361 -0.001522 1.6049 0.402 0.376 0.144
	0.31 -3.1571 1.5549 -0.1087 -1.1677 0.1160 -1.1494 0.0791 -2.1879 0.0328 -0.001369 1.6232 0.395 0.366 0.148
	0.40 -3.8516 1.7429 -0.1228 -1.1354 0.1126 -1.1099 0.0746 -2.2064 0.0294 -0.001240 1.6320 0.387 0.356 0.152
	0.50 -4.5556 1.9258 -0.1360 -1.1015 0.1084 -1.0708 0.0700 -2.2171 0.0261 -0.001129 1.6109 0.378 0.345 0.156
	0.63 -5.2405 2.0926 -0.1471 -1.0659 0.1035 -1.0328 0.0655 -2.2220 0.0229 -0.001033 1.5735 0.369 0.333 0.160
	0.79 -5.8909 2.2357 -0.1557 -1.0279 0.0981 -0.9969 0.0612 -2.2229 0.0197 -0.000945 1.5262 0.360 0.320 0.164
	1.00 -6.4633 2.3419 -0.1605 -0.9895 0.0925 -0.9665 0.0577 -2.2211 0.0167 -0.000863 1.4809 0.350 0.307 0.168
	1.25 -6.9250 2.4037 -0.1612 -0.9545 0.0879 -0.9462 0.0558 -2.2178 0.0139 -0.000785 1.4710 0.341 0.294 0.172
	1.59 -7.2960 2.4189 -0.1573 -0.9247 0.0848 -0.9421 0.0567 -2.2137 0.0111 -0.000701 1.5183 0.331 0.280 0.177
	2.00 -7.5053 2.3805 -0.1492 -0.9128 0.0855 -0.9658 0.0619 -2.2110 0.0086 -0.000618 1.6365 0.323 0.267 0.181
	2.50 -7.5569 2.2933 -0.1376 -0.9285 0.0915 -1.0264 0.0729 -2.2108 0.0067 -0.000535 1.8421 0.315 0.254 0.186
	3.13 -7.4510 2.1598 -0.1228 -0.9872 0.1050 -1.1349 0.0914 -2.2141 0.0060 -0.000458 2.1028 0.308 0.242 0.190
	4.00 -7.1688 1.9738 -0.1048 -1.1274 0.1325 -1.3132 0.1207 -2.2224 0.0079 -0.000397 2.4336 0.299 0.227 0.195
	5.00 -6.8063 1.7848 -0.0879 -1.3324 0.1691 -1.5158 0.1533 -2.2374 0.0142 -0.000387 2.6686 0.291 0.214 0.198
	""")

