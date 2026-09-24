"""Check train-only fitting, unknown levels, grouping and predictor exclusions."""

import unittest

import numpy as np
import pandas as pd
from scipy import sparse

from scripts.clean_hotel_bookings import load_cleaned
from scripts.preprocess_hotel_bookings import (
    EXCLUSIONS, fit_preprocessor, prepare_predictors, split_positions,
)


class PreprocessingTests(unittest.TestCase):
    def test_test_values_do_not_affect_scaler_or_category_vocabulary(self):
        train = pd.DataFrame({
            'lead_time': [10, 20, 30, 40], 'previous_cancellations': [0, 0, 0, 1],
            'hotel': ['City Hotel'] * 4, 'country': ['PRT'] * 4,
            'agent': ['9'] * 4, 'company': ['Not_recorded'] * 4,
            'is_repeated_guest': [0, 1, 0, 1],
        })
        test = train.iloc[[0]].copy()
        test['lead_time'] = 100000
        test['hotel'] = 'TEST_ONLY_HOTEL'
        test['country'] = 'TEST_ONLY_COUNTRY'
        transformer, a, b = fit_preprocessor(train, test)
        self.assertEqual(transformer.named_transformers_['numeric'].center_[0], 25)
        self.assertNotIn('TEST_ONLY_HOTEL', transformer.named_transformers_['categorical'].categories_[0])
        self.assertNotIn('TEST_ONLY_COUNTRY', transformer.named_transformers_['high_cardinality'].categories_[0])
        self.assertEqual(a.shape[1], b.shape[1])
        self.assertTrue(sparse.issparse(b))
        self.assertTrue(np.isfinite(b.data).all())
        alternate = test.copy()
        alternate['lead_time'] = -100000
        other, other_a, _ = fit_preprocessor(train, alternate)
        np.testing.assert_array_equal(transformer.named_transformers_['numeric'].center_, other.named_transformers_['numeric'].center_)
        self.assertEqual((a != other_a).nnz, 0)

    def test_identical_profiles_stay_together_and_all_rows_are_partitioned(self):
        X = pd.DataFrame({'lead_time': np.repeat(np.arange(50), 2), 'hotel': ['City Hotel'] * 100})
        y = pd.Series(np.repeat(np.arange(50) % 2, 2))
        train, test, groups = split_positions(X, y)
        self.assertTrue(set(groups.iloc[train]).isdisjoint(groups.iloc[test]))
        self.assertEqual(set(train).union(test), set(range(100)))
        self.assertEqual(len(set(train).intersection(test)), 0)
        again_train, again_test, _ = split_positions(X, y)
        np.testing.assert_array_equal(train, again_train)
        np.testing.assert_array_equal(test, again_test)

    def test_predictors_exclude_target_and_risky_fields_without_derived_columns(self):
        cleaned = load_cleaned().head(20)
        original = cleaned.copy(deep=True)
        X, y, decisions = prepare_predictors(cleaned)
        self.assertFalse(set(X.columns).intersection([*EXCLUSIONS, 'is_canceled']))
        self.assertEqual(set(X.columns), set(cleaned.columns) - set(EXCLUSIONS) - {'is_canceled'})
        self.assertFalse(any(isinstance(dtype, pd.CategoricalDtype) for dtype in X.dtypes))
        np.testing.assert_array_equal(y, cleaned['is_canceled'])
        self.assertEqual(len(decisions), len(cleaned.columns))
        pd.testing.assert_frame_equal(cleaned, original)


if __name__ == '__main__':
    unittest.main()
