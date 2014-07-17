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
Module exports :class:`ToroEtAl2002HTT`
"""
from __future__ import division

import numpy as np

from openquake.hazardlib.gsim.toro_2002 import ToroEtAl2002
from openquake.hazardlib.gsim.base import CoeffsTable
from openquake.hazardlib.imt import PGA, SA


class ToroEtAl2002HTT(ToroEtAl2002):
    """
    Implements vs30 and kappa host to target adjusments for ToroEtAl2002 GMPE.
    """

    #: HTT requires vs30 and kappa
    REQUIRES_SITES_PARAMETERS = set(('vs30', 'kappa'))

    def get_mean_and_stddevs(self, sites, rup, dists, imt, stddev_types):
        """
        Apply HTT
        """
        mean, stddevs = super(ToroEtAl2002HTT, self).get_mean_and_stddevs(sites, rup, dists, imt, stddev_types)

        if isinstance(imt, SA):
            freq = 1. / imt.period
        else:
            freq = 100.
        mean += np.log(self.get_kappa_cf(sites.kappa, freq))
        mean += np.log(self.get_vs30_cf(sites.vs30, imt))
        
        return mean, stddevs

    def get_kappa_cf(self, target_kappas, freq):
        """
        """
        host_kappa = 0.006
        host_amps = np.exp(-np.pi * host_kappa * freq)
        target_amps = np.exp(-np.pi * target_kappas * freq)
        amps = target_amps / host_amps
        return amps

    def get_vs30_cf(self, target_vs30s, imt):
        """
        """
        cfs = []
        for target_vs30 in target_vs30s:
           cfs.append(self.HTT_COEFFS[imt]['%.f' % target_vs30])
        return np.array(cfs)

    HTT_COEFFS = CoeffsTable(sa_damping=5, table="""\
imt 600 700 800 900 1000 1100 1200 1300 1400 1500 1600 1700 1800 1900 2000 2100 2200 2300 2400 2500 2600 2700
pga  3.8714 3.3918 3.0264 2.7255 2.4149 2.1876 2.0106 1.8555 1.7305 1.6265 1.5337 1.4547 1.3862 1.3238 1.2688 1.2197 1.1739 1.1327 1.0952 1.0608 1.0292 1.0000
0.03 3.5041 2.9610 2.5895 2.3198 2.1015 1.9330 1.7977 1.6826 1.5867 1.5052 1.4335 1.3710 1.3160 1.2663 1.2218 1.1817 1.1446 1.1108 1.0798 1.0512 1.0247 1.0000
0.04 3.3059 2.8137 2.4747 2.2268 2.0276 1.8724 1.7471 1.6407 1.5517 1.4756 1.4087 1.3502 1.2985 1.2518 1.2099 1.1720 1.1371 1.1051 1.0758 1.0486 1.0234 1.0000
0.10 2.7496 2.3979 2.1499 1.9645 1.8168 1.6991 1.6025 1.5202 1.4499 1.3890 1.3353 1.2878 1.2456 1.2075 1.1731 1.1418 1.1131 1.0868 1.0626 1.0402 1.0194 1.0000
0.20 2.3981 2.1286 1.9319 1.7820 1.6624 1.5661 1.4868 1.4196 1.3625 1.3131 1.2697 1.2314 1.1974 1.1667 1.1391 1.1139 1.0909 1.0698 1.0503 1.0323 1.0156 1.0000
0.40 2.0334 1.8296 1.6832 1.5727 1.4854 1.4152 1.3574 1.3085 1.2669 1.2309 1.1994 1.1716 1.1467 1.1243 1.1040 1.0855 1.0684 1.0527 1.0381 1.0245 1.0119 1.0000
1.00 1.5760 1.4759 1.4022 1.3449 1.2985 1.2600 1.2274 1.1993 1.1746 1.1528 1.1333 1.1157 1.0997 1.0850 1.0716 1.0591 1.0476 1.0368 1.0267 1.0173 1.0084 1.0000
2.00 1.3826 1.3237 1.2783 1.2418 1.2114 1.1857 1.1636 1.1442 1.1270 1.1117 1.0978 1.0852 1.0736 1.0630 1.0531 1.0440 1.0355 1.0275 1.0200 1.0130 1.0063 1.0000
    """)

