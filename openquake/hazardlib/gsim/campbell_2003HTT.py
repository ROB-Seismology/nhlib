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
Module exports :class:`Campbell2003HTT`
"""
from __future__ import division

import numpy as np

from openquake.hazardlib.gsim.campbell_2003 import Campbell2003
from openquake.hazardlib.gsim.base import CoeffsTable
from openquake.hazardlib.imt import PGA, SA


class Campbell2003HTT(Campbell2003):
    """
    Implements vs30 and kappa host to target adjusments for Campbell2003 GMPE.
    """

    #: HTT requires vs30 and kappa
    REQUIRES_SITES_PARAMETERS = set(('vs30', 'kappa'))

    def get_mean_and_stddevs(self, sites, rup, dists, imt, stddev_types):
        """
        Apply HTT
        """
        mean, stddevs = super(Campbell2003HTT, self).get_mean_and_stddevs(sites, rup, dists, imt, stddev_types)

        htt_coeffs = self.HTT_COEFFS[imt]

        if isinstance(imt, SA):
            freq = 1. / imt.period
        else:
            freq = 100. ## freq of PGA
        mean += np.log(self.get_kappa_cf(sites.kappa, freq))
        mean += np.log(self.get_vs30_cf(sites.vs30, htt_coeffs))

        return mean, stddevs

    def get_kappa_cf(self, target_kappas, freq):
        """
        """
        host_kappa = 0.0069
        host_amps = np.exp(-np.pi * host_kappa * freq)
        target_amps = np.exp(-np.pi * target_kappas * freq)
        amps = target_amps / host_amps
        return amps

    def get_vs30_cf(self, target_vs30s, htt_coeffs):
        """
        """
        if (target_vs30s == target_vs30s[0]).all():
            return htt_coeffs['%.f' % target_vs30s[0]]
        else:
            cfs = []
            for target_vs30 in target_vs30s:
               cfs.append(htt_coeffs['%.f' % target_vs30])
            return np.array(cfs)

    HTT_COEFFS = CoeffsTable(sa_damping=5, table="""\
imt 600 700 800 900 1000 1100 1200 1300 1400 1500 1600 1700 1800 1900 2000 2100 2200 2300 2400 2500 2600 2700
pga  3.8714 3.3918 3.0264 2.7255 2.4149 2.1876 2.0106 1.8555 1.7305 1.6265 1.5337 1.4547 1.3862 1.3238 1.2688 1.2197 1.1739 1.1327 1.0952 1.0608 1.0292 1.0000
0.02 3.7997 3.1809 2.7606 2.4592 2.2110 2.0223 1.8722 1.7437 1.6378 1.5483 1.4694 1.4011 1.3412 1.2871 1.2389 1.1955 1.1553 1.1189 1.0855 1.0548 1.0264 1.0000
0.03 3.5041 2.9610 2.5895 2.3198 2.1015 1.9330 1.7977 1.6826 1.5867 1.5052 1.4335 1.3710 1.3160 1.2663 1.2218 1.1817 1.1446 1.1108 1.0798 1.0512 1.0247 1.0000
0.05 3.1592 2.7044 2.3894 2.1578 1.9724 1.8271 1.7092 1.6092 1.5252 1.4532 1.3899 1.3344 1.2852 1.2409 1.2009 1.1647 1.1314 1.1008 1.0727 1.0467 1.0225 1.0000
0.07 2.9116 2.5195 2.2450 2.0412 1.8788 1.7501 1.6450 1.5559 1.4805 1.4156 1.3584 1.3079 1.2629 1.2223 1.1855 1.1521 1.1213 1.0931 1.0672 1.0431 1.0208 1.0000
0.10 2.7496 2.3979 2.1499 1.9645 1.8168 1.6991 1.6025 1.5202 1.4499 1.3890 1.3353 1.2878 1.2456 1.2075 1.1731 1.1418 1.1131 1.0868 1.0626 1.0402 1.0194 1.0000
0.15 2.5378 2.2380 2.0242 1.8617 1.7308 1.6252 1.5379 1.4638 1.4007 1.3461 1.2982 1.2559 1.2182 1.1843 1.1537 1.1259 1.1005 1.0771 1.0556 1.0357 1.0172 1.0000
0.20 2.3981 2.1286 1.9319 1.7820 1.6624 1.5661 1.4868 1.4196 1.3625 1.3131 1.2697 1.2314 1.1974 1.1667 1.1391 1.1139 1.0909 1.0698 1.0503 1.0323 1.0156 1.0000
0.30 2.1906 1.9565 1.7877 1.6601 1.5590 1.4777 1.4108 1.3542 1.3061 1.2644 1.2279 1.1956 1.1669 1.1411 1.1177 1.0965 1.0770 1.0592 1.0427 1.0275 1.0133 1.0000
0.50 1.9108 1.7316 1.6030 1.5060 1.4295 1.3683 1.3181 1.2757 1.2394 1.2079 1.1801 1.1554 1.1332 1.1131 1.0948 1.0780 1.0625 1.0482 1.0349 1.0225 1.0109 1.0000
0.75 1.6951 1.5650 1.4721 1.4015 1.3451 1.2990 1.2604 1.2273 1.1986 1.1733 1.1508 1.1306 1.1123 1.0957 1.0804 1.0664 1.0533 1.0412 1.0299 1.0193 1.0094 1.0000
1.00 1.5760 1.4759 1.4022 1.3449 1.2985 1.2600 1.2274 1.1993 1.1746 1.1528 1.1333 1.1157 1.0997 1.0850 1.0716 1.0591 1.0476 1.0368 1.0267 1.0173 1.0084 1.0000
1.50 1.4519 1.3793 1.3242 1.2804 1.2444 1.2140 1.1881 1.1654 1.1455 1.1277 1.1116 1.0971 1.0838 1.0717 1.0604 1.0500 1.0403 1.0312 1.0227 1.0147 1.0071 1.0000
2.00 1.3826 1.3237 1.2783 1.2418 1.2114 1.1857 1.1636 1.1442 1.1270 1.1117 1.0978 1.0852 1.0736 1.0630 1.0531 1.0440 1.0355 1.0275 1.0200 1.0130 1.0063 1.0000
3.00 1.3009 1.2566 1.2219 1.1937 1.1700 1.1499 1.1324 1.1169 1.1032 1.0909 1.0797 1.0695 1.0601 1.0515 1.0435 1.0361 1.0291 1.0226 1.0164 1.0107 1.0052 1.0000
4.00 1.2505 1.2145 1.1861 1.1628 1.1432 1.1264 1.1118 1.0989 1.0874 1.0770 1.0676 1.0590 1.0511 1.0438 1.0370 1.0307 1.0248 1.0192 1.0140 1.0091 1.0044 1.0000
    """)

