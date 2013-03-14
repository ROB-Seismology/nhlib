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
Module exports :class:`Campbell2003`, :class:`Campbell2003SHARE`
"""
from __future__ import division

import numpy as np
# standard acceleration of gravity in m/s**2
from scipy.constants import g

from openquake.hazardlib.gsim.base import CoeffsTable, GMPE
from openquake.hazardlib import const
from openquake.hazardlib.imt import PGA, SA


## Compute these logs only once
ln70, ln130 = np.log(70), np.log(130)
ln_g = np.log(g)

class Campbell2003adjusted(GMPE):
    """
    Implements GMPE developed by K.W Campbell and published as "Prediction of
    Strong Ground Motion Using the Hybrid Empirical Method and Its Use in the
    Development of Ground Motion (Attenuation) Relations in Eastern North
    America" (Bulletting of the Seismological Society of America, Volume 93,
    Number 3, pages 1012-1033, 2003). The class implements also the corrections
    given in the erratum (2004).
    """

    #: Supported tectonic region type is stable continental crust given that
    #: the equations have been derived for Eastern North America.
    DEFINED_FOR_TECTONIC_REGION_TYPE = const.TRT.STABLE_CONTINENTAL

    #: Supported intensity measure types are spectral acceleration,
    #: and peak ground acceleration, see table 6, page 1022 (PGA is assumed
    #: to be equal to SA at 0.01 s)
    DEFINED_FOR_INTENSITY_MEASURE_TYPES = set([
        PGA,
        SA
    ])

    #: Supported intensity measure component is the geometric mean of two
    #: horizontal components :attr:`~openquake.hazardlib.const.IMC.AVERAGE_HORIZONTAL`,
    DEFINED_FOR_INTENSITY_MEASURE_COMPONENT = const.IMC.AVERAGE_HORIZONTAL

    #: Supported standard deviation type is only total, see equation 35, page
    #: 1021
    DEFINED_FOR_STANDARD_DEVIATION_TYPES = set([
        const.StdDev.TOTAL
    ])

    #: No site parameters are needed
    REQUIRES_SITES_PARAMETERS = set()

    #: Required rupture parameter is only magnitude, see equation 30 page
    #: 1021.
    REQUIRES_RUPTURE_PARAMETERS = set(('mag', ))

    #: Required distance measure is closest distance to rupture, see equation
    #: 30 page 1021.
    REQUIRES_DISTANCES = set(('rrup', ))

    def get_mean_and_stddevs(self, sites, rup, dists, imt, stddev_types, kappa=0.03):
        """
        See :meth:`superclass method
        <.base.GroundShakingIntensityModel.get_mean_and_stddevs>`
        for spec of input and result values.
        """
        assert all(stddev_type in self.DEFINED_FOR_STANDARD_DEVIATION_TYPES
                   for stddev_type in stddev_types)

        C = self.COEFFS[(800, 0.03)][imt]
        mean = self._compute_mean(C, rup.mag, dists.rrup)
        stddevs = self._get_stddevs(C, stddev_types, rup.mag,
                                    dists.rrup.shape[0])

        # convert mean in m/s2 to mean in g
        mean = mean - ln_g

        return mean, stddevs

    def _compute_mean(self, C, mag, rrup):
        """
        Compute mean value according to equation 30, page 1021.
        """
        mean = (C['c1'] +
                self._compute_term1(C, mag) +
                self._compute_term2(C, mag, rrup) +
                self._compute_term3(C, rrup))
        return mean

    def _get_stddevs(self, C, stddev_types, mag, num_sites):
        """
        Return total standard deviation as for equation 35, page 1021.
        """
        stddevs = []
        for stddev_type in stddev_types:
            if mag < 7.16:
                sigma = C['c11'] + C['c12'] * mag
            elif mag >= 7.16:
                sigma = C['c13']
            stddevs.append(np.zeros(num_sites) + sigma)

        return stddevs

    def _compute_term1(self, C, mag):
        """
        This computes the term f1 in equation 31, page 1021
        """
        x = 8.5 - mag
        #return (C['c2'] * mag) + C['c3'] * (8.5 - mag) ** 2
        return (C['c2'] * mag) + C['c3'] * x * x

    def _compute_term2(self, C, mag, rrup):
        """
        This computes the term f2 in equation 32, page 1021
        """
        x = (C['c7'] * np.exp(C['c8'] * mag))
        c78_factor = x * x
        #c78_factor = (C['c7'] * np.exp(C['c8'] * mag)) ** 2
        #R = np.sqrt(rrup ** 2 + c78_factor)
        R = np.sqrt(rrup * rrup + c78_factor)

        return C['c4'] * np.log(R) + (C['c5'] + C['c6'] * mag) * rrup

    def _compute_term3(self, C, rrup):
        """
        This computes the term f3 in equation 34, page 1021 but corrected
        according to the erratum.
        """
        f3 = np.zeros_like(rrup)

        idx_between_70_130 = (rrup > 70) & (rrup <= 130)
        idx_greater_130 = rrup > 130

        f3[idx_between_70_130] = (
            #C['c9'] * (np.log(rrup[idx_between_70_130]) - np.log(70))
            C['c9'] * (np.log(rrup[idx_between_70_130]) - ln70)
        )

        f3[idx_greater_130] = (
            #C['c9'] * (np.log(rrup[idx_greater_130]) - np.log(70)) +
            #C['c10'] * (np.log(rrup[idx_greater_130]) - np.log(130))
            C['c9'] * (np.log(rrup[idx_greater_130]) - ln70) +
            C['c10'] * (np.log(rrup[idx_greater_130]) - ln130)
        )

        return f3

    #: Coefficient tables are constructed from the electronic suplements of
    #: the original paper.
    COEFFS = {}
    COEFFS[(800, 0.03)] = CoeffsTable(sa_damping=5, table="""\
    IMT       c1        c2        c3        c4        c5          c6          c7       c8       c9       c10       c11       c12       c13
    pga       1.8740    0.746     -0.0485   -1.760    -0.00225    0.000204    0.710    0.430    2.048    -1.614    1.030    -0.0860    0.414
    0.028     2.0672    0.752     -0.0427   -1.793    -0.00236    0.000206    0.715    0.433    2.036    -1.565    1.030    -0.0860    0.414
    0.040     2.0876    0.758     -0.0357   -1.769    -0.00237    0.000237    0.703    0.435    1.846    -1.536    1.036    -0.0849    0.429
    0.100     2.2006    0.737     -0.0302   -1.655    -0.00263    0.000255    0.636    0.446    1.812    -1.832    1.059    -0.0838    0.460
    0.200     2.2518    0.712     -0.0554   -1.580    -0.00282    0.000173    0.569    0.457    1.836    -1.485    1.077    -0.0838    0.478
    0.400     2.0873    0.638     -0.1081   -1.456    -0.00232    0.000130    0.495    0.458    1.630    -1.175    1.089    -0.0831    0.495
    1.000     1.9377    0.515     -0.1975   -1.323    -0.00166    0.000126    0.475    0.450    1.310    -0.879    1.110    -0.0793    0.543
    2.000     1.4741    0.450     -0.2545   -1.242    -0.00114    0.000113    0.463    0.453    1.108    -0.743    1.093    -0.0758    0.551
    """)

