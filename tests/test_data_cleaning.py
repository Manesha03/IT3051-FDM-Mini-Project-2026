"""Regression checks for destructive cleaning decisions and saved-data fidelity."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.clean_hotel_bookings import (
    correct_types_and_missing, load_cleaned, load_raw, parse_status_dates,
    remove_invalid_and_duplicates, validate_cleaned,
)


class DataCleaningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.template = load_raw().iloc[[0]].copy()

    def sample(self, count):
        data = pd.concat([self.template] * count, ignore_index=True)
        # Use synthetic identities; never depend on the real guest fields in tests.
        data['name'] = pd.Series([f'Guest {i}' for i in range(count)], dtype='string')
        data['email'] = pd.Series([f'test{i}@example.invalid' for i in range(count)], dtype='string')
        data['phone-number'] = pd.Series(['0012345'] * count, dtype='string')
        data['credit_card'] = pd.Series(['**** 0001'] * count, dtype='string')
        data['adults'] = 2
        data['children'] = 0.0
        data['babies'] = 0
        return data

    def test_exclusions_overlap_and_exact_duplicates_keep_source_lineage(self):
        raw = self.sample(9)
        raw.loc[1, 'children'] = np.nan
        raw.loc[2, ['adults', 'children', 'babies']] = 0
        raw.loc[3, 'adr'] = -6.38
        raw.loc[4, ['adults', 'children', 'babies']] = 0
        raw.loc[4, 'adr'] = -1
        raw.loc[5] = raw.loc[0]
        raw.loc[6, 'is_canceled'] = 2
        raw.loc[7, 'arrival_date_day_of_month'] = 32
        raw.loc[8, 'reservation_status_date'] = 'not-a-date'
        eligible, audit = remove_invalid_and_duplicates(raw)
        self.assertEqual(eligible.index.tolist(), [0])
        self.assertEqual(len(audit), 8)
        self.assertTrue(audit['source_row_index'].is_unique)
        reason = audit.set_index('source_row_index').loc[4, 'reason']
        self.assertIn('zero_guests', reason)
        self.assertIn('invalid_adr', reason)
        cleaned = correct_types_and_missing(eligible)
        self.assertTrue(validate_cleaned(raw, cleaned, audit)['passed'].all())

    def test_valid_extremes_zeros_and_matching_booking_profiles_are_retained(self):
        raw = self.sample(4)
        raw.loc[0, ['lead_time', 'adr', 'previous_bookings_not_canceled']] = [737, 5400, 72]
        raw.loc[1, ['adults', 'children', 'adr']] = [0, 2, 0]
        raw.loc[1, ['stays_in_weekend_nights', 'stays_in_week_nights']] = 0
        raw.loc[2:3, ['country', 'agent', 'company']] = np.nan
        untouched = raw.copy(deep=True)
        eligible, audit = remove_invalid_and_duplicates(raw)
        cleaned = correct_types_and_missing(eligible)
        self.assertEqual(len(cleaned), 4)
        self.assertTrue(audit.empty)
        self.assertEqual(cleaned.loc[0, 'adr'], 5400)
        self.assertEqual(cleaned.loc[1, 'children'], 2)
        self.assertEqual(cleaned.loc[2, 'country'], 'Unknown')
        self.assertEqual(cleaned.loc[2, 'agent'], 'Not_recorded')
        pd.testing.assert_frame_equal(raw, untouched)
        self.assertTrue(validate_cleaned(raw, cleaned, audit)['passed'].all())

    def test_fractional_negative_and_nonfinite_counts_are_rejected(self):
        raw = self.sample(3)
        raw['children'] = [0.5, -1, np.inf]
        eligible, audit = remove_invalid_and_duplicates(raw)
        self.assertTrue(eligible.empty)
        self.assertEqual(len(audit), 3)

    def test_target_corruption_is_detected(self):
        raw = self.sample(2)
        eligible, audit = remove_invalid_and_duplicates(raw)
        cleaned = correct_types_and_missing(eligible)
        cleaned.loc[0, 'is_canceled'] = 1 - cleaned.loc[0, 'is_canceled']
        with self.assertRaisesRegex(ValueError, 'Target labels unchanged'):
            validate_cleaned(raw, cleaned, audit)

    def test_raw_and_exported_date_formats_are_unambiguous(self):
        dates = parse_status_dates(pd.Series(['4/22/2015', '5/6/2015', '2015-05-06', '2/30/2015']))
        self.assertEqual(dates.iloc[0], pd.Timestamp('2015-04-22'))
        self.assertEqual(dates.iloc[1], pd.Timestamp('2015-05-06'))
        self.assertEqual(dates.iloc[1], dates.iloc[2])
        self.assertTrue(pd.isna(dates.iloc[3]))

    def test_csv_round_trip_restores_categories_dates_and_text_identifiers(self):
        raw = self.sample(2)
        raw['agent'] = [9.0, np.nan]
        cleaned = correct_types_and_missing(raw)
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            path = Path(directory) / 'cleaned.csv'
            cleaned.to_csv(path, index=False, date_format='%Y-%m-%d')
            restored = load_cleaned(path)
        pd.testing.assert_frame_equal(cleaned, restored)
        self.assertEqual(restored.loc[0, 'agent'], '9')
        self.assertEqual(restored.loc[0, 'phone-number'], '0012345')


if __name__ == '__main__':
    unittest.main()
