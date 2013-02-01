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
Module exports :class:`BindiEtAl2011`.
"""
from __future__ import division

import numpy as np

from nhlib.gsim.base import GMPE, CoeffsTable
from nhlib import const
from nhlib.imt import PGA, PGV, SA


class BindiEtAl2011(GMPE):
	"""
	Implements GMPE developed by Bindi, D., Pacor, F., Luzi, L., Puglia, R.,
	Massa, M., Ameri, G., Paolucci, R. and published as "Ground motion
	prediction equations derived from the Italian strong motion database" (2011,
	Bulletin of Earthquake Engineering, Volume 9, No. 6, pages 1899-1920). This
	class does not implement class E soils and unknown fault types.
	"""
	#: Supported tectonic region type is active shallow crust, give that the
	#: the equations have been derived for Italy.
	DEFINED_FOR_TECTONIC_REGION_TYPE = const.TRT.ACTIVE_SHALLOW_CRUST
	
	#: Supported intensity measure types are spectral acceleration, peak ground
	#: velocity and peak ground acceleration, p. 5.
	DEFINED_FOR_INTENSITY_MEASURE_TYPES = set([
		PGA,
		PGV,
		SA,
	])
	
	#: Supported intensity measure component is the geometric mean of two
	#: horizontal components :attr:`~nhlib.const.IMC.AVERAGE_HORIZONTAL`, p. 6.
	DEFINED_FOR_INTENSITY_MEASURE_COMPONENT = const.IMC.AVERAGE_HORIZONTAL
	
	#: Supported standard deviation types are inter-event, intra-event and
	#: total, p. 6.
	DEFINED_FOR_STANDARD_DEVIATION_TYPES = set([
		const.StdDev.TOTAL,
		const.StdDev.INTER_EVENT,
		const.StdDev.INTRA_EVENT,
	])
	
	#: Required site parameters is Vs30, p. 4.
	REQUIRES_SITES_PARAMETERS = set([
		'vs30',
	])
	
	#: Required rupture parameters are magnitude and rake, p. 6.
	REQUIRES_RUPTURE_PARAMETERS = set([
		'mag',
		'rake',
	])
	
	#: Required distance measure is rjb, p. 5.
	REQUIRES_DISTANCES = set([
		'rjb'
	])

	def get_mean_and_stddevs(self, sites, rup, dists, imt, stddev_types):
		"""
		See :meth:`superclass method
		<nhlib.gsim.base.GroundShakingIntensityModel.get_mean_and_stddevs>`
		for spec of input and result values.
		"""
		# extracting dictionary of coefficients specific to required
		# intensity measure type.
		C = self.COEFFS[imt]
		
		# mean value as given by equation (1), p. 5.
		mean = C['e1'] + self._get_distance_scaling_term(C, rup, dists) + self._get_magnitude_scaling_term(C, rup) + self._get_site_term(C, sites) + self._get_fault_term(C, rup)
		if isinstance(imt, PGA) or isinstance(imt, SA):
			# from log10(cm/s**2) to g
				mean = (10 ** mean) / 981
		if isinstance(imt, PGV):
			# from log10(cm/s) to m/s
			mean = (10 ** mean) / 100.

		# natural logarithm
		mean = np.log(mean)
		
		stddevs = self._get_stddevs(C, stddev_types, len(sites.vs30))
		
		return mean, stddevs
	
	def _get_distance_scaling_term(self, C, rup, dists):
		"""
		Get distance scaling term, equation (2), p. 5.
		"""
		#: p. 6
		Mref, Rref = 5., 1.
		
		return (C['c1'] + C['c2'] * (rup.mag - Mref)) * np.log10(np.sqrt(dists.rjb ** 2 + C['h'] ** 2) / Rref) - C['c3'] * (np.sqrt(dists.rjb ** 2 + C['h'] ** 2) - Rref)
	
	def _get_magnitude_scaling_term(self, C, rup):
		"""
		Get magnitude scaling term, equation (3), p. 5.
		"""
		#: p. 6
		Mh = 6.75
		
		if rup.mag <= Mh:
			return C['b1'] * (rup.mag - Mh) + C['b2'] * (rup.mag - Mh) ** 2
		else:
			#: Should be "C['b3'] * (rup.mag - Mh)", but b3 = 0, p. 6.
			return 0.
	
	def _get_site_term(self, C, sites):
		"""
		Get site term, p. 4 and 6.
		"""
		site_term = np.zeros_like(sites.vs30)
		
		site_term[sites.vs30 >= 800] = C['sA']
		site_term[(sites.vs30 >= 360) & (sites.vs30 < 800)] = C['sB']
		site_term[(sites.vs30 >= 180) & (sites.vs30 < 360)] = C['sC']
		site_term[sites.vs30 < 180] = C['sD']
		
		return site_term
	
	def _get_fault_term(self, C, rup):
		"""
		Get fault term, p. 6.
		"""
		if rup.rake > -135.0 and rup.rake <= -45.0:
			return C['f1']
		elif rup.rake > 45.0 and rup.rake <= 135.0:
			return C['f2']
		else:
			return C['f3']

	def _get_stddevs(self, C, stddev_types, num_sites):
		"""
		Get standard deviations.
		"""
		stddevs = []
		
		for stddev_type in stddev_types:
			assert stddev_type in self.DEFINED_FOR_STANDARD_DEVIATION_TYPES
			if stddev_type == const.StdDev.TOTAL:
				stddevs.append(C['s_total'] + np.zeros(num_sites))
			elif stddev_type == const.StdDev.INTRA_EVENT:
				stddevs.append(C['s_intra'] + np.zeros(num_sites))
			elif stddev_type == const.StdDev.INTER_EVENT:
				stddevs.append(C['s_inter'] + np.zeros(num_sites))
				
		return stddevs
	
	#: Coefficient table obtained by joining tables 1, p. 19 and 5, p. 22.
	COEFFS = CoeffsTable(sa_damping=5, table="""\
	IMT		e1		c1		c2		h		c3			b1		b2			sA	sB		sC		sD		sE		f1		f2		f3		f4	s_inter	s_intra	s_total
	pgv		2.305	-1.517	0.328	7.879	0.			0.236	-0.00686	0	0.205	0.289	0.321	0.428	-0.0308	0.0754	-0.0446	0	0.194	0.270	0.332
	pga		3.672	-1.940	0.413	10.322	0.000134	-0.262	-0.07070	0	0.162	0.240	0.105	0.570	-0.0503	0.105	-0.0544	0	0.172	0.290	0.337
	0.04	3.725	-1.976	0.422	9.445	0.000270	-0.315	-0.07870	0	0.161	0.240	0.060	0.614	-0.0442	0.106	-0.0615	0	0.154	0.307	0.343
	0.07	3.906	-2.050	0.446	9.810	0.000758	-0.375	-0.07730	0	0.154	0.235	0.057	0.536	-0.0454	0.103	-0.0576	0	0.152	0.324	0.358
	0.10	3.796	-1.794	0.415	9.500	0.002550	-0.290	-0.06510	0	0.178	0.247	0.037	0.599	-0.0656	0.111	-0.0451	0	0.154	0.328	0.363
	0.15	3.799	-1.521	0.320	9.163	0.003720	-0.0987	-0.05740	0	0.174	0.240	0.148	0.740	-0.0755	0.123	-0.0477	0	0.179	0.318	0.365
	0.20	3.750	-1.379	0.280	8.502	0.003840	0.00940	-0.05170	0	0.156	0.234	0.115	0.556	-0.0733	0.106	-0.0328	0	0.209	0.320	0.382
	0.25	3.699	-1.340	0.254	7.912	0.003260	0.0860	-0.04570	0	0.182	0.245	0.154	0.414	-0.0568	0.110	-0.0534	0	0.212	0.308	0.374
	0.30	3.753	-1.414	0.255	8.215	0.002190	0.124	-0.04350	0	0.201	0.244	0.213	0.301	-0.0564	0.0877	-0.0313	0	0.218	0.290	0.363
	0.35	3.600	-1.320	0.253	7.507	0.002320	0.154	-0.04370	0	0.220	0.257	0.243	0.235	-0.0523	0.0905	-0.0382	0	0.221	0.283	0.359
	0.40	3.549	-1.262	0.233	6.760	0.002190	0.225	-0.04060	0	0.229	0.255	0.226	0.202	-0.0565	0.0927	-0.0363	0	0.210	0.279	0.349
	0.45	3.550	-1.261	0.223	6.775	0.001760	0.292	-0.03060	0	0.226	0.271	0.237	0.181	-0.0597	0.0886	-0.0289	0	0.204	0.284	0.350
	0.50	3.526	-1.181	0.184	5.992	0.001860	0.384	-0.02500	0	0.218	0.280	0.263	0.168	-0.0599	0.0850	-0.0252	0	0.203	0.283	0.349
	0.60	3.561	-1.230	0.178	6.382	0.001140	0.436	-0.02270	0	0.219	0.296	0.355	0.142	-0.0559	0.0790	-0.0231	0	0.203	0.283	0.348
	0.70	3.485	-1.172	0.154	5.574	0.000942	0.529	-0.01850	0	0.210	0.303	0.496	0.134	-0.0461	0.0896	-0.0435	0	0.212	0.283	0.354
	0.80	3.325	-1.115	0.163	4.998	0.000909	0.545	-0.02150	0	0.210	0.304	0.621	0.150	-0.0457	0.0795	-0.0338	0	0.213	0.284	0.355
	0.90	3.318	-1.137	0.154	5.231	0.000483	0.563	-0.02630	0	0.212	0.315	0.680	0.154	-0.0351	0.0715	-0.0364	0	0.214	0.286	0.357
	1.00	3.264	-1.114	0.140	5.002	0.000254	0.599	-0.02700	0	0.221	0.332	0.707	0.152	-0.0298	0.0660	-0.0362	0	0.222	0.283	0.360
	1.25	2.895	-0.986	0.173	4.340	0.000783	0.579	-0.03360	0	0.244	0.365	0.717	0.183	-0.0207	0.0614	-0.0407	0	0.227	0.290	0.368
	1.50	2.675	-0.960	0.192	4.117	0.000802	0.575	-0.03530	0	0.251	0.375	0.667	0.203	-0.0140	0.0505	-0.0365	0	0.218	0.303	0.373
	1.75	2.584	-1.006	0.205	4.505	0.000427	0.574	-0.03710	0	0.252	0.357	0.593	0.220	0.00154	0.0370	-0.0385	0	0.219	0.305	0.376
	2.00	2.537	-1.009	0.193	4.373	0.000164	0.597	-0.03670	0	0.245	0.352	0.540	0.226	0.00512	0.0350	-0.0401	0	0.211	0.308	0.373
	""")
