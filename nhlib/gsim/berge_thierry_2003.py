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
Module exports :class:`BergeThierry2003`.
"""
from __future__ import division

import numpy as np

from nhlib.gsim.base import GMPE, CoeffsTable
from nhlib import const
from nhlib.imt import PGA, PGV, SA


class BergeThierry2003(GMPE):
    """
    Implements GMPE developed by Berge-Thierry et al.
    and published as "New empirical response spectral attenuation laws for
    moderate European earthquakes" (2003, Journal of Earthquake Engineering,
    Volume 7, No. 2, pages 193-222).
    """
    # TODO: adjust comments

    #: Supported tectonic region type is active shallow crust, see
    #: paragraph 'Introduction', page 99.
    DEFINED_FOR_TECTONIC_REGION_TYPE = const.TRT.STABLE_CONTINENTAL

    #: Supported intensity measure types are spectral acceleration,
    #: peak ground velocity and peak ground acceleration, see table 3
    #: pag. 110
    DEFINED_FOR_INTENSITY_MEASURE_TYPES = set([
        PGA,
        SA
    ])

    #: Supported intensity measure component is orientation-independent
    #: measure :attr:`~nhlib.const.IMC.GMRotI50`, see paragraph
    #: 'Response Variables', page 100 and table 8, pag 121.
    # TODO: this information is not in the paper, and needs to be confirmed
    DEFINED_FOR_INTENSITY_MEASURE_COMPONENT = const.IMC.GREATER_OF_TWO_HORIZONTAL

    #: Supported standard deviation types are inter-event, intra-event
    #: and total, see equation 2, pag 106.
    DEFINED_FOR_STANDARD_DEVIATION_TYPES = set([
        const.StdDev.TOTAL
    ])

    #: Required site parameters is Vs30.
    #: See paragraph 'Predictor Variables', pag 103
    REQUIRES_SITES_PARAMETERS = set(('vs30', ))

    #: Required rupture parameters are magnitude, and rake.
    #: See paragraph 'Predictor Variables', pag 103
    REQUIRES_RUPTURE_PARAMETERS = set(('mag', ))

    #: Required distance measure is Rjb.
    #: See paragraph 'Predictor Variables', pag 103
    REQUIRES_DISTANCES = set(('rhypo', ))

    def get_mean_and_stddevs(self, sites, rup, dists, imt, stddev_types):
        """
        See :meth:`superclass method
        <nhlib.gsim.base.GroundShakingIntensityModel.get_mean_and_stddevs>`
        for spec of input and result values.
        """
        # extracting dictionary of coefficients specific to required
        # intensity measure type.
        C = self.COEFFS[imt]

        # equation 1, pag 106, without sigma term, that is only the first 3
        # terms. The third term (site amplification) is computed as given in
        # equation (6), that is the sum of a linear term - equation (7) - and
        # a non-linear one - equations (8a) to (8c).
        # Mref, Rref values are given in the caption to table 6, pag 119.
        log10_mean = self._compute_magnitude_scaling(rup, C) + \
            self._compute_distance_scaling(rup, dists, C) + \
            self._get_site_amplification_linear(sites, C) + \
            self._get_site_amplification_non_linear(sites, rup, dists, C)
        # Convert ms-2 to g, and take the natural logarithm
        mean = (10**log10_mean) / 981.0
        mean = np.log(mean)

        stddevs = self._get_stddevs(C, stddev_types, num_sites=len(sites.vs30))
        stddevs = np.log(stddevs)

        return mean, stddevs

    def _get_stddevs(self, C, stddev_types, num_sites):
        """
        Return standard deviations as defined in table 8, pag 121.
        """
        stddevs = []
        for stddev_type in stddev_types:
            assert stddev_type in self.DEFINED_FOR_STANDARD_DEVIATION_TYPES
            if stddev_type == const.StdDev.TOTAL:
                stddevs.append(C['std'] + np.zeros(num_sites))
        stddevs = 10**np.array(stddevs)
        return stddevs

    def _compute_distance_scaling(self, rup, dists, C):
        """
        Compute distance-scaling term, equations (3) and (4), pag 107.
        """
        val = C['b'] * dists.rhypo - np.log10(dists.rhypo)
        return val

    def _compute_magnitude_scaling(self, rup, C):
        """
        Compute magnitude-scaling term, equations (5a) and (5b), pag 107.
        """
        val = C['a'] * rup.mag
        return val

    def _get_fault_type_dummy_variables(self, rup):
        """
        Get fault type dummy variables, see Table 2, pag 107.
        Fault type (Strike-slip, Normal, Thrust/reverse) is
        derived from rake angle.
        Rakes angles within 30 of horizontal are strike-slip,
        angles from 30 to 150 are reverse, and angles from
        -30 to -150 are normal. See paragraph 'Predictor Variables'
        pag 103.
        Note that the 'Unspecified' case is not considered,
        because rake is always given.
        """
        pass

    def _get_site_amplification_linear(self, sites, C):
        """
        Compute site amplification linear term,
        equation (7), pag 107.
        """
        # TODO: According to the paper, the threshold vs30 value for rock is 800 m/s
        choice_dict = {True: C['c1'], False: C['c2']}
        val = np.array([choice_dict[vs30 >= 760.0] for vs30 in sites.vs30])
        return val

    def _get_site_amplification_non_linear(self, sites, rup, dists, C):
        """
        Compute site amplification non-linear term,
        equations (8a) to (13d), pag 108-109.
        """
        return 0.

    def _compute_non_linear_slope(self, sites, C):
        """
        Compute non-linear slope factor,
        equations (13a) to (13d), pag 108-109.
        """
        pass

    def _compute_non_linear_term(self, pga4nl, bnl):
        """
        Compute non-linear term,
        equation (8a) to (8c), pag 108.
        """
        pass

    #: Coefficient table is constructed from values in tables 3, 6, 7 and 8
    #: (pages 110, 119, 120, 121). Spectral acceleration is defined for damping
    #: of 5%, see 'Response Variables' page 100.
    #: blin, b1, b2 are the period-dependent site-amplification coefficients.
    #: c1, c2, c3, h are the period-dependent distance scaling coefficients.
    #: e1, e2, e3, e4, e5, e6, e7, Mh are the period-dependent magnitude-
    # scaling coefficients.
    #: sigma, tau, std are the intra-event uncertainty, inter-event
    #: uncertainty, and total standard deviation, respectively.
    #: Note that only the inter-event and total standard deviation for
    #: 'specified' fault type are considered (because rake angle is always
    #: specified)

    # TODO: add other damping factors
    COEFFS = CoeffsTable(sa_damping=5, table="""\
    IMT	a	b	c1	c2	std
    pga	3.1180E-01	-9.3030E-04	1.5370E+00	1.5730E+00	2.9230E-01
    3.0000E-02	3.1140E-01	-9.3340E-04	1.5410E+00	1.5760E+00	2.9240E-01
    3.2258E-02	3.0970E-01	-9.4220E-04	1.5580E+00	1.5890E+00	2.9280E-01
    3.4000E-02	3.0830E-01	-9.5470E-04	1.5730E+00	1.6020E+00	2.9350E-01
    3.5714E-02	3.0680E-01	-9.8220E-04	1.5930E+00	1.6180E+00	2.9470E-01
    4.0000E-02	3.0330E-01	-1.1190E-03	1.6530E+00	1.6650E+00	2.9820E-01
    4.5455E-02	3.0160E-01	-1.2820E-03	1.7010E+00	1.7000E+00	3.0150E-01
    5.0000E-02	2.9920E-01	-1.3410E-03	1.7400E+00	1.7290E+00	3.0090E-01
    5.3000E-02	2.9810E-01	-1.4290E-03	1.7660E+00	1.7490E+00	3.0220E-01
    5.5556E-02	2.9690E-01	-1.4320E-03	1.7850E+00	1.7650E+00	3.0270E-01
    5.8824E-02	2.9600E-01	-1.4600E-03	1.8090E+00	1.7840E+00	3.0400E-01
    5.9999E-02	2.9600E-01	-1.4720E-03	1.8140E+00	1.7880E+00	3.0410E-01
    6.2500E-02	2.9650E-01	-1.5070E-03	1.8260E+00	1.7960E+00	3.0470E-01
    6.4998E-02	2.9440E-01	-1.5150E-03	1.8510E+00	1.8170E+00	3.0590E-01
    6.6667E-02	2.9330E-01	-1.5130E-03	1.8650E+00	1.8290E+00	3.0590E-01
    6.8966E-02	2.9240E-01	-1.5460E-03	1.8810E+00	1.8420E+00	3.0480E-01
    6.9999E-02	2.9200E-01	-1.5700E-03	1.8880E+00	1.8490E+00	3.0460E-01
    7.1429E-02	2.9090E-01	-1.5830E-03	1.9010E+00	1.8610E+00	3.0470E-01
    7.4074E-02	2.8790E-01	-1.5550E-03	1.9260E+00	1.8870E+00	3.0590E-01
    7.5002E-02	2.8710E-01	-1.5400E-03	1.9330E+00	1.8930E+00	3.0660E-01
    7.6923E-02	2.8570E-01	-1.5200E-03	1.9470E+00	1.9060E+00	3.0720E-01
    8.0000E-02	2.8660E-01	-1.5310E-03	1.9540E+00	1.9110E+00	3.0610E-01
    8.3333E-02	2.8580E-01	-1.5520E-03	1.9680E+00	1.9270E+00	3.0580E-01
    8.4998E-02	2.8560E-01	-1.5830E-03	1.9760E+00	1.9350E+00	3.0530E-01
    8.6957E-02	2.8480E-01	-1.5710E-03	1.9870E+00	1.9460E+00	3.0420E-01
    9.0001E-02	2.8190E-01	-1.5360E-03	2.0110E+00	1.9730E+00	3.0200E-01
    9.0909E-02	2.8090E-01	-1.5280E-03	2.0200E+00	1.9810E+00	3.0160E-01
    9.5238E-02	2.7810E-01	-1.5430E-03	2.0540E+00	2.0100E+00	3.0170E-01
    1.0000E-01	2.7860E-01	-1.4740E-03	2.0590E+00	2.0160E+00	3.0190E-01
    1.0526E-01	2.7760E-01	-1.4220E-03	2.0720E+00	2.0340E+00	3.0430E-01
    1.1000E-01	2.7830E-01	-1.4420E-03	2.0750E+00	2.0450E+00	3.0730E-01
    1.1111E-01	2.7830E-01	-1.4380E-03	2.0770E+00	2.0470E+00	3.0790E-01
    1.1765E-01	2.7930E-01	-1.4380E-03	2.0810E+00	2.0560E+00	3.1030E-01
    1.2000E-01	2.8060E-01	-1.4360E-03	2.0760E+00	2.0540E+00	3.1060E-01
    1.2500E-01	2.8310E-01	-1.4150E-03	2.0660E+00	2.0510E+00	3.1210E-01
    1.2903E-01	2.8630E-01	-1.4400E-03	2.0520E+00	2.0420E+00	3.1370E-01
    1.3001E-01	2.8670E-01	-1.4250E-03	2.0500E+00	2.0400E+00	3.1420E-01
    1.3333E-01	2.8870E-01	-1.3970E-03	2.0400E+00	2.0330E+00	3.1560E-01
    1.3793E-01	2.9030E-01	-1.3380E-03	2.0320E+00	2.0280E+00	3.1610E-01
    1.4000E-01	2.9150E-01	-1.3220E-03	2.0270E+00	2.0220E+00	3.1660E-01
    1.4286E-01	2.9330E-01	-1.3070E-03	2.0180E+00	2.0140E+00	3.1730E-01
    1.4815E-01	2.9500E-01	-1.2560E-03	2.0090E+00	2.0100E+00	3.1910E-01
    1.4999E-01	2.9550E-01	-1.2180E-03	2.0040E+00	2.0090E+00	3.1960E-01
    1.5385E-01	2.9550E-01	-1.0520E-03	1.9970E+00	2.0070E+00	3.2010E-01
    1.6000E-01	2.9390E-01	-8.0560E-04	1.9960E+00	2.0130E+00	3.2000E-01
    1.6667E-01	2.9520E-01	-7.0970E-04	1.9890E+00	2.0060E+00	3.2150E-01
    1.7001E-01	2.9740E-01	-6.9860E-04	1.9780E+00	1.9940E+00	3.2300E-01
    1.7391E-01	3.0160E-01	-7.3410E-04	1.9570E+00	1.9700E+00	3.2470E-01
    1.7999E-01	3.0890E-01	-7.7930E-04	1.9150E+00	1.9300E+00	3.2670E-01
    1.8182E-01	3.1090E-01	-7.8260E-04	1.9010E+00	1.9190E+00	3.2710E-01
    1.9001E-01	3.1470E-01	-7.3690E-04	1.8680E+00	1.8940E+00	3.2620E-01
    1.9048E-01	3.1490E-01	-7.3370E-04	1.8670E+00	1.8930E+00	3.2620E-01
    2.0000E-01	3.1670E-01	-6.8890E-04	1.8430E+00	1.8810E+00	3.2500E-01
    2.0833E-01	3.1960E-01	-6.7190E-04	1.8140E+00	1.8610E+00	3.2610E-01
    2.1739E-01	3.2540E-01	-6.7500E-04	1.7700E+00	1.8250E+00	3.2810E-01
    2.1978E-01	3.2710E-01	-6.9180E-04	1.7580E+00	1.8150E+00	3.2920E-01
    2.2727E-01	3.3030E-01	-6.6780E-04	1.7260E+00	1.7920E+00	3.3200E-01
    2.3810E-01	3.3400E-01	-6.1710E-04	1.6830E+00	1.7620E+00	3.3670E-01
    2.3998E-01	3.3440E-01	-5.9880E-04	1.6770E+00	1.7580E+00	3.3710E-01
    2.5000E-01	3.3650E-01	-5.7500E-04	1.6510E+00	1.7360E+00	3.3940E-01
    2.5974E-01	3.4300E-01	-7.0750E-04	1.6090E+00	1.6970E+00	3.4220E-01
    2.6316E-01	3.4420E-01	-7.2000E-04	1.5990E+00	1.6880E+00	3.4290E-01
    2.7778E-01	3.5010E-01	-7.5200E-04	1.5500E+00	1.6450E+00	3.4440E-01
    2.8003E-01	3.5110E-01	-7.5300E-04	1.5420E+00	1.6380E+00	3.4470E-01
    2.9002E-01	3.5550E-01	-7.8360E-04	1.5060E+00	1.6050E+00	3.4580E-01
    3.0003E-01	3.5900E-01	-8.5200E-04	1.4770E+00	1.5810E+00	3.4770E-01
    3.0303E-01	3.6020E-01	-8.7370E-04	1.4660E+00	1.5730E+00	3.4830E-01
    3.1696E-01	3.6710E-01	-9.2720E-04	1.4120E+00	1.5250E+00	3.4910E-01
    3.2000E-01	3.6900E-01	-9.4680E-04	1.3970E+00	1.5120E+00	3.4870E-01
    3.3333E-01	3.7420E-01	-1.0100E-03	1.3520E+00	1.4720E+00	3.4740E-01
    3.4002E-01	3.7520E-01	-1.0060E-03	1.3370E+00	1.4610E+00	3.4690E-01
    3.4483E-01	3.7600E-01	-9.6980E-04	1.3260E+00	1.4520E+00	3.4710E-01
    3.5714E-01	3.8070E-01	-9.1140E-04	1.2860E+00	1.4150E+00	3.4810E-01
    3.5997E-01	3.8220E-01	-9.0390E-04	1.2750E+00	1.4050E+00	3.4840E-01
    3.7037E-01	3.8670E-01	-8.6350E-04	1.2370E+00	1.3720E+00	3.4920E-01
    3.7994E-01	3.9090E-01	-8.0740E-04	1.1990E+00	1.3390E+00	3.5000E-01
    3.8462E-01	3.9310E-01	-7.9550E-04	1.1790E+00	1.3210E+00	3.5070E-01
    4.0000E-01	3.9970E-01	-7.0780E-04	1.1190E+00	1.2670E+00	3.5170E-01
    4.1667E-01	4.0280E-01	-6.6130E-04	1.0780E+00	1.2320E+00	3.5120E-01
    4.1999E-01	4.0340E-01	-6.5130E-04	1.0700E+00	1.2260E+00	3.5130E-01
    4.3478E-01	4.0700E-01	-6.1670E-04	1.0290E+00	1.1910E+00	3.5210E-01
    4.3995E-01	4.0890E-01	-6.1180E-04	1.0110E+00	1.1740E+00	3.5270E-01
    4.5455E-01	4.1480E-01	-5.9310E-04	9.6010E-01	1.1250E+00	3.5470E-01
    4.5998E-01	4.1650E-01	-5.8160E-04	9.4400E-01	1.1090E+00	3.5490E-01
    4.7619E-01	4.2220E-01	-5.4040E-04	8.9310E-01	1.0580E+00	3.5550E-01
    4.8008E-01	4.2390E-01	-5.4840E-04	8.7950E-01	1.0450E+00	3.5560E-01
    5.0000E-01	4.3230E-01	-5.6800E-04	8.1500E-01	9.7970E-01	3.5550E-01
    5.2002E-01	4.3720E-01	-5.3960E-04	7.6420E-01	9.3240E-01	3.5680E-01
    5.2632E-01	4.3790E-01	-5.0500E-04	7.5220E-01	9.2080E-01	3.5700E-01
    5.3996E-01	4.3940E-01	-4.3300E-04	7.2710E-01	8.9590E-01	3.5740E-01
    5.5556E-01	4.4180E-01	-3.6010E-04	6.9410E-01	8.6610E-01	3.5870E-01
    5.5991E-01	4.4250E-01	-3.3800E-04	6.8440E-01	8.5710E-01	3.5920E-01
    5.8005E-01	4.4720E-01	-2.7020E-04	6.3570E-01	8.1170E-01	3.6100E-01
    5.8824E-01	4.4920E-01	-2.5220E-04	6.1570E-01	7.9260E-01	3.6090E-01
    5.9988E-01	4.5160E-01	-2.1750E-04	5.8950E-01	7.6820E-01	3.6030E-01
    6.1996E-01	4.5590E-01	-1.9530E-04	5.4780E-01	7.2820E-01	3.6040E-01
    6.2500E-01	4.5690E-01	-1.9950E-04	5.3830E-01	7.1870E-01	3.6090E-01
    6.4020E-01	4.5960E-01	-1.6660E-04	5.1060E-01	6.9100E-01	3.6250E-01
    6.6007E-01	4.6370E-01	-1.5490E-04	4.7250E-01	6.5200E-01	3.6470E-01
    6.6667E-01	4.6550E-01	-1.5500E-04	4.5730E-01	6.3630E-01	3.6490E-01
    6.7981E-01	4.6880E-01	-1.6680E-04	4.2840E-01	6.0810E-01	3.6550E-01
    6.9979E-01	4.7320E-01	-1.7000E-04	3.8570E-01	5.6760E-01	3.6670E-01
    7.1429E-01	4.7710E-01	-2.0190E-04	3.5480E-01	5.3540E-01	3.6800E-01
    7.5019E-01	4.8470E-01	-3.0090E-04	2.8710E-01	4.6810E-01	3.7000E-01
    7.6923E-01	4.8750E-01	-3.1220E-04	2.5450E-01	4.3800E-01	3.7030E-01
    8.0000E-01	4.9400E-01	-2.5680E-04	1.9060E-01	3.7820E-01	3.7140E-01
    8.3333E-01	5.0100E-01	-1.9320E-04	1.2640E-01	3.1450E-01	3.7460E-01
    8.5034E-01	5.0400E-01	-1.4330E-04	9.6150E-02	2.8420E-01	3.7580E-01
    9.0009E-01	5.0980E-01	3.2820E-05	2.0060E-02	2.1300E-01	3.7470E-01
    9.0909E-01	5.1040E-01	8.3930E-05	8.1950E-03	2.0220E-01	3.7440E-01
    1.0000E+00	5.1990E-01	2.5160E-04	-1.1620E-01	8.2900E-02	3.7370E-01
    1.1001E+00	5.2730E-01	3.9080E-04	-2.1230E-01	-2.9000E-02	3.7940E-01
    1.1111E+00	5.2780E-01	4.0740E-04	-2.2070E-01	-3.8750E-02	3.8040E-01
    1.2005E+00	5.3610E-01	4.4790E-04	-3.1330E-01	-1.3380E-01	3.8380E-01
    1.2500E+00	5.4090E-01	4.8600E-04	-3.6790E-01	-1.8910E-01	3.8770E-01
    1.3004E+00	5.4440E-01	5.3290E-04	-4.1130E-01	-2.3770E-01	3.9200E-01
    1.4006E+00	5.4810E-01	7.6760E-04	-4.8700E-01	-3.1550E-01	3.9350E-01
    1.4286E+00	5.4940E-01	8.2720E-04	-5.0950E-01	-3.3850E-01	3.9410E-01
    1.4993E+00	5.5270E-01	9.1240E-04	-5.6040E-01	-3.9350E-01	3.9320E-01
    1.6000E+00	5.5570E-01	9.8440E-04	-6.1860E-01	-4.5830E-01	3.9190E-01
    1.6667E+00	5.5800E-01	1.0850E-03	-6.5640E-01	-5.0260E-01	3.9190E-01
    1.7986E+00	5.6200E-01	1.2450E-03	-7.2580E-01	-5.8340E-01	3.9420E-01
    2.0000E+00	5.6220E-01	1.3750E-03	-7.9630E-01	-6.6600E-01	4.0300E-01
    2.1978E+00	5.6170E-01	1.6520E-03	-8.6560E-01	-7.3950E-01	4.0550E-01
    2.3981E+00	5.6410E-01	1.8290E-03	-9.4060E-01	-8.1580E-01	4.0930E-01
    2.5000E+00	5.6540E-01	1.9210E-03	-9.7870E-01	-8.5420E-01	4.1100E-01
    2.5974E+00	5.6770E-01	2.0060E-03	-1.0190E+00	-8.9800E-01	4.1300E-01
    2.8011E+00	5.6660E-01	2.2770E-03	-1.0710E+00	-9.4950E-01	4.2020E-01
    3.0030E+00	5.6830E-01	2.4490E-03	-1.1300E+00	-1.0140E+00	4.2550E-01
    3.2051E+00	5.6860E-01	2.5360E-03	-1.1790E+00	-1.0690E+00	4.3010E-01
    3.3333E+00	5.7050E-01	2.5330E-03	-1.2200E+00	-1.1110E+00	4.3290E-01
    3.4014E+00	5.7150E-01	2.5410E-03	-1.2430E+00	-1.1350E+00	4.3400E-01
    3.5971E+00	5.7270E-01	2.5730E-03	-1.3000E+00	-1.1940E+00	4.3590E-01
    3.8023E+00	5.7120E-01	2.6620E-03	-1.3500E+00	-1.2420E+00	4.3650E-01
    4.0000E+00	5.7220E-01	2.7110E-03	-1.4170E+00	-1.3030E+00	4.3440E-01
    4.5045E+00	5.8560E-01	2.4490E-03	-1.6620E+00	-1.5200E+00	4.2780E-01
    5.0000E+00	5.9900E-01	2.1050E-03	-1.8860E+00	-1.7290E+00	4.2330E-01
    5.4945E+00	6.1060E-01	1.9410E-03	-2.0720E+00	-1.9040E+00	4.2600E-01
    5.9880E+00	6.1600E-01	1.8800E-03	-2.2010E+00	-2.0280E+00	4.2840E-01
    6.9930E+00	6.1750E-01	1.7660E-03	-2.3730E+00	-2.1920E+00	4.2990E-01
    8.0000E+00	6.1450E-01	1.7210E-03	-2.4910E+00	-2.3080E+00	4.2840E-01
    9.0090E+00	6.1220E-01	1.6370E-03	-2.5920E+00	-2.4080E+00	4.2360E-01
    1.0000E+01	6.0860E-01	1.5630E-03	-2.6680E+00	-2.4850E+00	4.1830E-01
    """)
