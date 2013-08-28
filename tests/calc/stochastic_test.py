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
import unittest

from openquake.hazardlib.tom import PoissonTOM
from openquake.hazardlib.calc.stochastic import stochastic_event_set_poissonian


class StochasticEventSetTestCase(unittest.TestCase):
    class FakeRupture(object):
        def __init__(self, occurrences):
            self.occurrences = occurrences

        def sample_number_of_occurrences(self):
            return self.occurrences

    class FakeSource(object):
        def __init__(self, source_id, ruptures, time_span):
            self.source_id = source_id
            self.time_span = time_span
            self.ruptures = ruptures

        def iter_ruptures(self, tom):
            assert tom.time_span is self.time_span
            assert isinstance(tom, PoissonTOM)
            return iter(self.ruptures)

    class FailSource(FakeSource):
        def iter_ruptures(self, tom):
            raise ValueError('Something bad happened')

    def setUp(self):
        self.time_span = 15
        self.r1_1 = self.FakeRupture(1)
        self.r1_0 = self.FakeRupture(0)
        self.r1_2 = self.FakeRupture(2)
        self.r2_1 = self.FakeRupture(1)
        self.source1 = self.FakeSource(
            1, [self.r1_1, self.r1_0, self.r1_2], self.time_span)
        self.source2 = self.FakeSource(
            2, [self.r2_1], self.time_span)

    def test_no_filter(self):
        ses = list(
            stochastic_event_set_poissonian(
                [self.source1, self.source2],
                self.time_span
            ))
        self.assertEqual(ses, [self.r1_1, self.r1_2, self.r1_2, self.r2_1])

    def test_filter(self):
        def extract_first_source(sources_sites):
            for source, _sites in sources_sites:
                yield source, None
                break
        fake_sites = [1, 2, 3]
        ses = list(
            stochastic_event_set_poissonian(
                [self.source1, self.source2],
                self.time_span, fake_sites, extract_first_source
            ))
        self.assertEqual(ses, [self.r1_1, self.r1_2, self.r1_2])

        def extract_first_rupture(ruptures_sites):
            for rupture, _sites in ruptures_sites:
                yield rupture, None
                break
        ses = list(
            stochastic_event_set_poissonian(
                [self.source1, self.source2],
                self.time_span, fake_sites,
                extract_first_source,
                extract_first_rupture
            ))
        self.assertEqual(ses, [self.r1_1])
        self.source1 = self.FakeSource(1, [self.r1_1, self.r1_0, self.r1_2],
                                       self.time_span)
        self.source2 = self.FakeSource(2, [self.r2_1], self.time_span)

    def test(self):
        ses = list(stochastic_event_set_poissonian(
            [self.source1, self.source2], self.time_span))
        self.assertEqual(ses, [self.r1_1, self.r1_2, self.r1_2, self.r2_1])

    def test_source_errors(self):
        # exercise the case where an error occurs while computing on a given
        # seismic source; in this case, we expect an error to be raised which
        # signals the id of the source in question
        fail_source = self.FailSource(2, [self.r2_1], self.time_span)
        with self.assertRaises(RuntimeError) as ae:
            list(stochastic_event_set_poissonian([self.source1, fail_source],
                                                 self.time_span))

        expected_error = (
            'An error occurred with source id=2. Error: Something bad happened'
        )
        self.assertEqual(expected_error, ae.exception.message)

    def test_source_errors_with_sites(self):
        # exercise the case where an error occurs while computing on a given
        # seismic source; in this case, we expect an error to be raised which
        # signals the id of the source in question
        fail_source = self.FailSource(2, [self.r2_1], self.time_span)
        fake_sites = [1, 2, 3]
        with self.assertRaises(RuntimeError) as ae:
            list(stochastic_event_set_poissonian([self.source1, fail_source],
                                                 self.time_span,
                                                 sites=fake_sites))

        expected_error = (
            'An error occurred with source id=2. Error: Something bad happened'
        )
        self.assertEqual(expected_error, ae.exception.message)
