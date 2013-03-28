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
Module exports :class:`ToroEtAl2002`, class:`ToroEtAl2002SHARE`.
"""
from __future__ import division

import numpy as np
# standard acceleration of gravity in m/s**2
from scipy.constants import g

from openquake.hazardlib.gsim.campbell_2003 import _compute_faulting_style_term
from openquake.hazardlib.gsim.base import CoeffsTable, GMPE
from openquake.hazardlib import const
from openquake.hazardlib.imt import PGA, SA


## Compute these logs only once
ln_g = np.log(g)


class ToroEtAl2002adjusted(GMPE):
    """
    Implements GMPE developed by G. R. Toro, N. A. Abrahamson, J. F. Sneider
    and published in "Model of Strong Ground Motions from Earthquakes in
    Central and Eastern North America: Best Estimates and Uncertainties"
    (Seismological Research Letters, Volume 68, Number 1, 1997) and
    "Modification of the Toro et al. 1997 Attenuation Equations for Large
    Magnitudes and Short Distances" (available at:
    http://www.riskeng.com/downloads/attenuation_equations)
    The class implements equations for Midcontinent, based on moment magnitude.
    SA at 3 and 4 s (not supported by the original equations) have been added
    in the context of the SHARE project and they are obtained from SA at 2 s
    scaled by specific factors for 3 and 4 s.
    """

    #: Supported tectonic region type is stable continental crust,
    #: given that the equations have been derived for central and eastern
    #: north America
    DEFINED_FOR_TECTONIC_REGION_TYPE = const.TRT.STABLE_CONTINENTAL

    #: Supported intensity measure types are spectral acceleration,
    #: and peak ground acceleration, see table 2 page 47.
    DEFINED_FOR_INTENSITY_MEASURE_TYPES = set([
        PGA,
        SA
    ])

    #: Supported intensity measure component is the geometric mean of two
    #: horizontal components :attr:`~openquake.hazardlib.const.IMC.AVERAGE_HORIZONTAL`,
    DEFINED_FOR_INTENSITY_MEASURE_COMPONENT = const.IMC.AVERAGE_HORIZONTAL

    #: Supported standard deviation type is only total.
    DEFINED_FOR_STANDARD_DEVIATION_TYPES = set([
        const.StdDev.TOTAL
    ])

    #: Required site parameters is Vs30.
    #: See paragraph 'Equations for soil sites', p. 2200
    REQUIRES_SITES_PARAMETERS = set(('vs30', ))

    #: Required rupture parameter is only magnitude.
    REQUIRES_RUPTURE_PARAMETERS = set(('mag', ))

    #: Required distance measure is rjb, see equation 4, page 46.
    REQUIRES_DISTANCES = set(('rjb', ))

    def get_mean_and_stddevs(self, sites, rup, dists, imt, stddev_types, kappa=0.03):
        """
        See :meth:`superclass method
        <.base.GroundShakingIntensityModel.get_mean_and_stddevs>`
        for spec of input and result values.
        """
        assert all(stddev_type in self.DEFINED_FOR_STANDARD_DEVIATION_TYPES
                   for stddev_type in stddev_types)

        C = self.COEFFS[(sites.vs30, kappa)][imt]
        mean = self._compute_mean(C, rup.mag, dists.rjb)
        stddevs = self._compute_stddevs(C, rup.mag, dists.rjb, imt,
                                        stddev_types)

        # apply decay factor for 3 and 4 seconds (not originally supported
        # by the equations)
        if isinstance(imt, SA):
            if imt.period == 3.0:
                mean /= 0.612
            if imt.period == 4.0:
                mean /= 0.559

        # convert mean in m/s2 to mean in g
        mean = mean - ln_g

        return mean, stddevs

    def _compute_term1(self, C, mag):
        """
        Compute magnitude dependent terms (2nd and 3rd) in equation 3
        page 46.
        """
        mag_diff = mag - 6

        #return C['c2'] * mag_diff + C['c3'] * mag_diff ** 2
        return C['c2'] * mag_diff + C['c3'] * mag_diff * mag_diff

    def _compute_term2(self, C, mag, rjb):
        """
        Compute distance dependent terms (4th, 5th and 6th) in equation 3
        page 46. The factor 'RM' is computed according to the 2002 model
        (equation 4-3).
        """
        x = np.exp(-1.25 + 0.227 * mag)
        RM = np.sqrt(rjb * rjb + (C['c7'] * C['c7']) * x * x)
        #RM = np.sqrt(rjb ** 2 + (C['c7'] ** 2) *
        #             np.exp(-1.25 + 0.227 * mag) ** 2)

        return (-C['c4'] * np.log(RM) -
                (C['c5'] - C['c4']) *
                np.maximum(np.log(RM / 100), 0) - C['c6'] * RM)

    def _compute_mean(self, C, mag, rjb):
        """
        Compute mean value according to equation 3, page 46.
        """
        mean = (C['c1'] +
                self._compute_term1(C, mag) +
                self._compute_term2(C, mag, rjb))
        return mean

    def _compute_stddevs(self, C, mag, rjb, imt, stddev_types):
        """
        Compute total standard deviation, equations 5 and 6, page 48.
        """
        # aleatory uncertainty
        sigma_ale_m = np.interp(mag, [5.0, 5.5, 8.0],
                                [C['m50'], C['m55'], C['m80']])
        sigma_ale_rjb = np.interp(rjb, [5.0, 20.0], [C['r5'], C['r20']])
        sigma_ale = np.sqrt(sigma_ale_m ** 2 + sigma_ale_rjb ** 2)

        # epistemic uncertainty
        if isinstance(imt, PGA) or (isinstance(imt, SA) and imt.period < 1):
            sigma_epi = 0.36 + 0.07 * (mag - 6)
        else:
            sigma_epi = 0.34 + 0.06 * (mag - 6)

        sigma_total = np.sqrt(sigma_ale ** 2 + sigma_epi ** 2)

        stddevs = []
        for _ in stddev_types:
            stddevs.append(sigma_total)

        return stddevs

    #: Coefficient table: factors c1 to c7 replaced with the values
    #: in Table 20 in the report of Drouet et al. (2010), p. 35
    COEFFS = {}
    COEFFS[(800, 0.03)] = CoeffsTable(sa_damping=5, table="""\
    IMT    c1    c2    c3    c4    c5    c6       c7   m50   m55   m80   r5    r20
    pga    3.56  0.83  0.00  1.02  0.58  0.0029   6.4  0.55  0.59  0.50  0.54  0.20
    0.03   4.02  0.85  0.00  1.06  0.84  0.0026   6.7  0.62  0.63  0.50  0.62  0.35
    0.04   4.39  0.83  0.00  1.13  0.87  0.0026   7.1  0.62  0.63  0.50  0.57  0.29
    0.10   4.20  0.81  0.01  0.95  0.47  0.0047   6.1  0.59  0.61  0.50  0.50  0.17
    0.20   3.89  0.83  0.01  0.87  0.19  0.0048   5.7  0.60  0.64  0.56  0.45  0.12
    0.40   3.36  1.05 -0.10  0.85  0.21  0.0035   5.7  0.63  0.68  0.64  0.45  0.12
    1.00   2.40  1.42 -0.20  0.84  0.24  0.0024   5.8  0.63  0.64  0.67  0.45  0.12
    2.00   1.56  1.84 -0.30  0.87  0.27  0.0017   6.1  0.61  0.62  0.66  0.45  0.12
    """)

    COEFFS[(2000, 0.005)] = CoeffsTable(sa_damping=5, table="""\
    IMT    c1    c2    c3    c4    c5    c6       c7   m50   m55   m80   r5    r20
    pga    5.22  0.75  0.02  1.34  0.89  0.0026   8.6  0.55  0.59  0.50  0.54  0.20
    0.03   6.49  0.76  0.02  1.43  1.69  0.0015   9.3  0.62  0.63  0.50  0.62  0.35
    0.04   6.11  0.76  0.01  1.35  1.46  0.0021   8.9  0.62  0.63  0.50  0.57  0.29
    0.10   4.69  0.78  0.01  1.02  0.41  0.0052   7.1  0.59  0.61  0.50  0.50  0.17
    0.20   4.04  0.82  0.00  0.93  0.12  0.0050   6.6  0.60  0.64  0.56  0.45  0.12
    0.40   3.35  1.05 -0.10  0.89  0.16  0.0036   6.4  0.63  0.68  0.64  0.45  0.12
    1.00   2.35  1.41 -0.19  0.87  0.20  0.0025   6.3  0.63  0.64  0.67  0.45  0.12
    2.00   1.58  1.83 -0.29  0.91  0.22  0.0018   6.7  0.61  0.62  0.66  0.45  0.12
    """)

