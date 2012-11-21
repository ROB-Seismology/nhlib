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
Module :mod:`nhlib.mfd.truncated_gr` defines a Truncated Gutenberg-Richter MFD.
"""
import math

from nhlib.mfd.base import BaseMFD


class ModifiedGRMFD(BaseMFD):
    """
    Truncated Gutenberg-Richter MFD is defined in a functional form.

    The annual occurrence rate for a specific bin (magnitude band)
    is defined as ::

        rate = 10 ** (a_val - b_val * mag_lo) - 10 ** (a_val - b_val * mag_hi)

    where

    * ``a_val`` is the cumulative ``a`` value (``10 ** a`` is the number
      of earthquakes per year with magnitude greater than or equal to 0),
    * ``b_val`` is Gutenberg-Richter ``b`` value -- the decay rate
      of exponential distribution. It describes the relative size distribution
      of earthquakes: a higher ``b`` value indicates a relatively larger
      proportion of small events and vice versa.
    * ``mag_lo`` and ``mag_hi`` are lower and upper magnitudes of a specific
      bin respectively.

    :param min_mag:
        The lowest possible magnitude for this MFD. The first bin in the
        :meth:`result histogram <get_annual_occurrence_rates>` will be aligned
        to make its left border match this value.
    :param max_mag:
        The highest possible magnitude. The same as for ``min_mag``: the last
        bin in the histogram will correspond to the magnitude value equal to
        ``max_mag - bin_width / 2``.
    :param bin_width:
        A positive float value -- the width of a single histogram bin.

    Values for ``min_mag`` and ``max_mag`` don't have to be aligned with
    respect to ``bin_width``. They get rounded accordingly anyway so that
    both are divisible by ``bin_width`` just before converting a function
    to a histogram. See :meth:`_get_min_mag_and_num_bins`.
    """
    MODIFICATIONS = set(('increment_max_mag',
                         'set_max_mag',
                         'increment_b',
                         'set_ab'))

    def __init__(self, min_mag, max_mag, bin_width, a_val, b_val):
        self.min_mag = min_mag
        self.max_mag = max_mag
        self.bin_width = bin_width
        self.a_val = a_val
        self.b_val = b_val

        self.check_constraints()

    def check_constraints(self):
        """
        Checks the following constraints:

        * Bin width is greater than 0.
        * Minimum magnitude is positive.
        * Maximum magnitude is greater than minimum magnitude
          by at least one bin width (or equal to that value).
        * ``b`` value is positive.
        """
        if not self.bin_width > 0:
            raise ValueError('bin width must be positive')

        if not self.min_mag >= 0:
            raise ValueError('minimum magnitude must be non-negative')

        if not self.max_mag >= self.min_mag + self.bin_width:
            raise ValueError('maximum magnitude must be higher than minimum '
                             'magnitude by bin width at least')

        if not 0 < self.b_val:
            raise ValueError('b value must be non-negative')

    def _get_rate(self, mag):
        """
        Calculate and return an annual occurrence rate for a specific bin.

        :param mag:
            Magnitude value corresponding to the center of the bin of interest.
        :returns:
            Float number, the annual occurrence rate calculated using formula
            described in :class:`TruncatedGRMFD`.
        """
        mag_lo = mag - self.bin_width / 2.0
        mag_hi = mag + self.bin_width / 2.0
        rate = self._get_mag_rate(mag_lo) - self._get_mag_rate(mag_hi)
        return rate

    def _get_mag_rate(self, mag):
        """
        """
        return (10**(self.a_val-self.b_val*self.min_mag))*((10**(-1*self.b_val*mag)-10**(-1*self.b_val*self.max_mag))/(10**(-1*self.b_val*self.min_mag)-10**(-1*self.b_val*self.max_mag)))

    def _get_min_mag_and_num_bins(self):
        """
        Estimate the number of bins in the histogram and return it
        along with the first bin center abscissa (magnitude) value.

        Rounds ``min_mag`` and ``max_mag`` with respect to ``bin_width``
        to make the distance between them include integer number of bins.

        :returns:
            A tuple of two items: first bin center and total number of bins.
        """
        min_mag = round(self.min_mag / self.bin_width) * self.bin_width
        max_mag = round(self.max_mag / self.bin_width) * self.bin_width
        if min_mag != max_mag:
            min_mag += self.bin_width / 2.0
            max_mag -= self.bin_width / 2.0
        # here we use math round on the result of division and not just
        # cast it to integer because for some magnitude values that can't
        # be represented as an IEEE 754 double precisely the result can
        # look like 7.999999999999 which would become 7 instead of 8
        # being naively casted to int so we would lose the last bin.
        num_bins = int(round((max_mag - min_mag) / self.bin_width)) + 1
        return min_mag, num_bins

    def get_min_mag(self):
        """
        See :meth:`_get_min_mag_and_num_bins`.
        """
        min_mag, num_bins = self._get_min_mag_and_num_bins()
        return min_mag

    def get_annual_occurrence_rates(self):
        """
        Calculate and return the annual occurrence rates histogram.

        The result histogram has only one bin if minimum and maximum magnitude
        values appear equal after rounding.

        :returns:
            See :meth:`nhlib.mfd.base.BaseMFD.get_annual_occurrence_rates`.
        """
        mag, num_bins = self._get_min_mag_and_num_bins()
        rates = []
        for i in xrange(num_bins):
            rate = self._get_rate(mag)
            rates.append((mag, rate))
            mag += self.bin_width
        return rates