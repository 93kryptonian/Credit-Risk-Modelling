"""Tests for src/features.py: business feature derivation and model-input
preparation."""

import pandas as pd
import pytest

from src.features import EXCLUDED_FEATURES, build_business_features, prepare_model_inputs


def minimal_application_frame():
    """The smallest frame build_business_features can run on -- every
    column it accesses unconditionally, nothing optional."""
    return pd.DataFrame({
        "AMT_ANNUITY": [10_000.0, 20_000.0],
        "AMT_INCOME_TOTAL": [100_000.0, 0.0],  # second row exercises safe_divide
        "AMT_CREDIT": [200_000.0, 400_000.0],
        "AMT_GOODS_PRICE": [180_000.0, 380_000.0],
        "CNT_FAM_MEMBERS": [2.0, 1.0],
        "CNT_CHILDREN": [1.0, 0.0],
        "DAYS_BIRTH": [-10_950.0, -14_600.0],  # ~30 and ~40 years old
        "DAYS_EMPLOYED": [-1825.0, -3650.0],
        "DAYS_REGISTRATION": [-2000.0, -3000.0],
        "DAYS_ID_PUBLISH": [-1000.0, -1500.0],
        "DAYS_LAST_PHONE_CHANGE": [-100.0, -200.0],
    })


class TestBuildBusinessFeatures:
    def test_dti_is_annuity_over_income(self):
        out = build_business_features(minimal_application_frame())
        assert out.loc[0, "DTI"] == pytest.approx(10_000.0 / 100_000.0, rel=1e-3)

    def test_dti_is_nan_not_zero_when_income_is_zero(self):
        out = build_business_features(minimal_application_frame())
        assert pd.isna(out.loc[1, "DTI"])

    def test_age_years_derived_from_negative_days_birth(self):
        out = build_business_features(minimal_application_frame())
        assert out.loc[0, "AGE_YEARS"] == pytest.approx(10_950 / 365.25, rel=1e-3)

    def test_goods_credit_ratio_computed(self):
        out = build_business_features(minimal_application_frame())
        assert out.loc[0, "GOODS_CREDIT_RATIO"] == pytest.approx(180_000 / 200_000, rel=1e-3)

    def test_income_per_adult_subtracts_children(self):
        out = build_business_features(minimal_application_frame())
        # row 0: 2 family members, 1 child -> 1 adult
        assert out.loc[0, "INCOME_PER_ADULT"] == pytest.approx(100_000.0, rel=1e-3)


class TestPrepareModelInputs:
    def _frame(self):
        return pd.DataFrame({
            "TARGET": [0, 1, 0, 1],
            "SK_ID_CURR": [100, 101, 102, 103],
            "CODE_GENDER": ["M", "F", "M", "F"],
            "DTI": [0.1, 0.3, 0.2, 0.4],
            "AGE_YEARS": [30.0, 45.0, 25.0, 60.0],
            "CONSTANT_COL": [1.0, 1.0, 1.0, 1.0],
        })

    def test_target_and_id_are_dropped_from_features(self):
        features, target, dropped = prepare_model_inputs(self._frame())
        assert "TARGET" not in features.columns
        assert "SK_ID_CURR" not in features.columns

    def test_gender_excluded_by_default(self):
        features, target, dropped = prepare_model_inputs(self._frame(), keep_gender=False)
        for col in EXCLUDED_FEATURES:
            assert col not in features.columns

    def test_gender_kept_when_explicitly_requested(self):
        # CODE_GENDER is a string column, so select_dtypes(numeric) would
        # drop it anyway unless it's encoded -- this test just confirms
        # the exclusion list itself is respected, not select_dtypes.
        features, target, dropped = prepare_model_inputs(self._frame(), keep_gender=True)
        assert "CODE_GENDER" not in dropped

    def test_constant_columns_are_dropped(self):
        features, target, dropped = prepare_model_inputs(self._frame())
        assert "CONSTANT_COL" not in features.columns
        assert "CONSTANT_COL" in dropped

    def test_target_series_matches_original_values(self):
        features, target, dropped = prepare_model_inputs(self._frame())
        assert list(target) == [0, 1, 0, 1]

    def test_only_numeric_columns_survive(self):
        features, target, dropped = prepare_model_inputs(self._frame())
        assert all(pd.api.types.is_numeric_dtype(features[c]) for c in features.columns)
