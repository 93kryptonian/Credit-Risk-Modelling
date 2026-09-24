"""Tests for src/data.py: safe division, dtype downcasting, and the
DAYS_EMPLOYED sentinel fix. These are the pure-logic pieces that don't
require the Kaggle CSVs on disk."""

import numpy as np
import pandas as pd

from src.data import DAYS_EMPLOYED_SENTINEL, downcast_frame, fix_days_employed, safe_divide


class TestSafeDivide:
    def test_normal_division(self):
        result = safe_divide(pd.Series([10.0, 20.0]), pd.Series([2.0, 4.0]))
        np.testing.assert_allclose(result, [5.0, 5.0])

    def test_division_by_zero_is_nan_not_inf_or_zero(self):
        result = safe_divide(pd.Series([1.0, -1.0, 0.0]), pd.Series([0.0, 0.0, 0.0]))
        assert result.isna().all(), "0/0 and x/0 must become NaN, never 0 or inf"

    def test_output_dtype_is_float32(self):
        result = safe_divide(pd.Series([1, 2, 3]), pd.Series([1, 1, 1]))
        assert result.dtype == np.float32

    def test_preserves_index(self):
        idx = pd.Index([10, 20, 30])
        num = pd.Series([1.0, 2.0, 3.0], index=idx)
        den = pd.Series([1.0, 2.0, 3.0], index=idx)
        result = safe_divide(num, den)
        assert list(result.index) == list(idx)


class TestDowncastFrame:
    def test_join_keys_excluded_from_downcasting(self):
        # Regression test for the documented bug: downcasting join keys
        # per-frame can pick different dtypes across frames and break
        # merges. SK_ID_CURR / SK_ID_BUREAU must stay untouched.
        frame = pd.DataFrame({
            "SK_ID_CURR": np.array([1, 2, 3], dtype=np.int64),
            "SK_ID_BUREAU": np.array([10, 20, 30], dtype=np.int64),
            "AMT_CREDIT": np.array([1000.0, 2000.0, 3000.0], dtype=np.float64),
        })
        out = downcast_frame(frame)
        assert out["SK_ID_CURR"].dtype == np.int64
        assert out["SK_ID_BUREAU"].dtype == np.int64
        assert out["AMT_CREDIT"].dtype == np.float32

    def test_integer_columns_downcast(self):
        frame = pd.DataFrame({"CNT_CHILDREN": np.array([0, 1, 2], dtype=np.int64)})
        out = downcast_frame(frame)
        assert out["CNT_CHILDREN"].dtype in (np.int8, np.int16, np.int32)

    def test_non_numeric_columns_untouched(self):
        frame = pd.DataFrame({"NAME_CONTRACT_TYPE": ["Cash loans", "Revolving loans"]})
        out = downcast_frame(frame)
        assert out["NAME_CONTRACT_TYPE"].dtype == object


class TestFixDaysEmployed:
    def test_sentinel_replaced_with_nan(self):
        frame = pd.DataFrame({"DAYS_EMPLOYED": [-500, DAYS_EMPLOYED_SENTINEL, -1200]})
        out = fix_days_employed(frame)
        assert pd.isna(out.loc[1, "DAYS_EMPLOYED"])
        assert out.loc[0, "DAYS_EMPLOYED"] == -500
        assert out.loc[2, "DAYS_EMPLOYED"] == -1200

    def test_anomaly_flag_marks_sentinel_rows_only(self):
        sentinel = DAYS_EMPLOYED_SENTINEL
        frame = pd.DataFrame({"DAYS_EMPLOYED": [-500, sentinel, sentinel]})
        out = fix_days_employed(frame)
        assert list(out["DAYS_EMPLOYED_ANOM"]) == [0, 1, 1]

    def test_original_frame_not_mutated(self):
        frame = pd.DataFrame({"DAYS_EMPLOYED": [DAYS_EMPLOYED_SENTINEL]})
        fix_days_employed(frame)
        # the sentinel must still be a real number, not silently NaN'd in place
        assert frame.loc[0, "DAYS_EMPLOYED"] == DAYS_EMPLOYED_SENTINEL
