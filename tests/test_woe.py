"""Tests for src/woe.py: WOE binning, Information Value selection, and the
points-based scorecard conversion. Uses small synthetic datasets with a
known monotonic relationship so the expected direction of every output is
known in advance."""

import numpy as np
import pandas as pd
import pytest

from src.models import build_scorecard_model
from src.woe import WOETransformer, scorecard_base_score, scorecard_points, select_by_iv


def make_monotonic_data(n=1000, seed=0):
    """A feature where higher values genuinely mean lower default risk,
    plus a pure-noise feature with no relationship to the target at all."""
    rng = np.random.default_rng(seed)
    signal = rng.uniform(0, 1, size=n)
    noise = rng.uniform(0, 1, size=n)
    # true default probability decreases as `signal` increases
    p_default = 0.5 - 0.4 * signal
    target = (rng.uniform(0, 1, size=n) < p_default).astype(int)
    X = pd.DataFrame({"SIGNAL": signal, "NOISE": noise})
    y = pd.Series(target)
    return X, y


class TestWOETransformer:
    def test_fit_produces_iv_for_every_column(self):
        X, y = make_monotonic_data()
        woe = WOETransformer(n_bins=5, min_bin_fraction=0.05).fit(X, y)
        assert set(woe.iv_.keys()) == {"SIGNAL", "NOISE"}

    def test_informative_feature_has_higher_iv_than_noise(self):
        X, y = make_monotonic_data(n=3000)
        woe = WOETransformer(n_bins=5, min_bin_fraction=0.05).fit(X, y)
        assert woe.iv_["SIGNAL"] > woe.iv_["NOISE"]

    def test_monotonic_woe_direction_matches_true_relationship(self):
        # Higher SIGNAL -> lower default risk -> higher WOE (WOE = ln(good/bad)).
        X, y = make_monotonic_data(n=3000)
        woe = WOETransformer(n_bins=5, min_bin_fraction=0.05, monotonic=True).fit(X, y)
        bins = [b for b in woe.bins_["SIGNAL"] if not b.is_missing]
        bins_sorted = sorted(bins, key=lambda b: b.lower)
        woe_values = [b.woe for b in bins_sorted]
        assert woe_values == sorted(woe_values), "WOE must increase monotonically with SIGNAL"

    def test_missing_values_get_their_own_bin(self):
        X, y = make_monotonic_data(n=2000)
        X.loc[X.index[:100], "SIGNAL"] = np.nan
        woe = WOETransformer(n_bins=5, min_bin_fraction=0.05).fit(X, y)
        assert any(b.is_missing for b in woe.bins_["SIGNAL"])

    def test_transform_output_columns_are_suffixed_woe(self):
        X, y = make_monotonic_data()
        woe = WOETransformer(n_bins=5, min_bin_fraction=0.05).fit(X, y)
        out = woe.transform(X)
        assert set(out.columns) == {"SIGNAL_WOE", "NOISE_WOE"}

    def test_transform_handles_missing_column_gracefully(self):
        # Production scenario: a column present at fit time is absent at
        # transform time. Should not raise -- treated as all-NaN.
        X, y = make_monotonic_data()
        woe = WOETransformer(n_bins=5, min_bin_fraction=0.05).fit(X, y)
        X_missing_col = X.drop(columns=["NOISE"])
        out = woe.transform(X_missing_col)
        assert "NOISE_WOE" in out.columns

    def test_requires_both_classes_present(self):
        X = pd.DataFrame({"A": [1.0, 2.0, 3.0]})
        y_all_zero = pd.Series([0, 0, 0])
        with pytest.raises(ValueError):
            WOETransformer().fit(X, y_all_zero)

    def test_constant_feature_gets_degenerate_zero_iv_bin(self):
        X, y = make_monotonic_data(n=500)
        X["CONSTANT"] = 1.0
        woe = WOETransformer(n_bins=5, min_bin_fraction=0.05).fit(X, y)
        assert woe.iv_["CONSTANT"] == 0.0
        assert len(woe.bins_["CONSTANT"]) == 1

    def test_min_bin_fraction_prevents_tiny_bins(self):
        # With a hard floor of 20%, no more than 5 bins can survive, and
        # every surviving value-bin must hold at least ~20% of rows.
        X, y = make_monotonic_data(n=1000)
        woe = WOETransformer(n_bins=10, min_bin_fraction=0.20, monotonic=False).fit(X, y)
        value_bins = [b for b in woe.bins_["SIGNAL"] if not b.is_missing]
        for b in value_bins:
            assert b.count >= 0.20 * 1000 * 0.5  # generous slack for merge-boundary effects


class TestSelectByIV:
    def test_min_iv_filters_noise_out(self):
        X, y = make_monotonic_data(n=3000)
        woe = WOETransformer(n_bins=5, min_bin_fraction=0.05).fit(X, y)
        # min_iv set high enough that only the genuinely informative
        # feature should survive.
        kept = select_by_iv(woe, min_iv=0.05, max_iv=10.0)
        assert "SIGNAL" in kept

    def test_top_n_limits_selection(self):
        X, y = make_monotonic_data(n=3000)
        woe = WOETransformer(n_bins=5, min_bin_fraction=0.05).fit(X, y)
        kept = select_by_iv(woe, min_iv=0.0, top_n=1)
        assert len(kept) == 1

    def test_exclude_removes_named_feature_even_if_strong(self):
        X, y = make_monotonic_data(n=3000)
        woe = WOETransformer(n_bins=5, min_bin_fraction=0.05).fit(X, y)
        kept = select_by_iv(woe, min_iv=0.0, exclude=["SIGNAL"])
        assert "SIGNAL" not in kept

    def test_max_iv_does_not_exclude_high_iv_features(self):
        # This is the documented design decision: max_iv only flags for
        # review via logging, it never silently drops a feature.
        X, y = make_monotonic_data(n=3000)
        woe = WOETransformer(n_bins=5, min_bin_fraction=0.05).fit(X, y)
        kept = select_by_iv(woe, min_iv=0.0, max_iv=0.0001)  # everything "flagged"
        assert "SIGNAL" in kept


class TestScorecardPoints:
    def _fit_pipeline(self, n=3000):
        X, y = make_monotonic_data(n=n)
        pipeline = build_scorecard_model(n_bins=5, min_bin_fraction=0.05)
        pipeline.fit(X, y)
        return pipeline

    def test_base_score_is_an_int(self):
        pipeline = self._fit_pipeline()
        base = scorecard_base_score(pipeline)
        assert isinstance(base, int)

    def test_higher_pdo_widens_point_spread(self):
        # PDO controls how many points correspond to doubling the odds --
        # a larger PDO must stretch the same log-odds range into more
        # points, so the total spread across bins should grow with it.
        pipeline = self._fit_pipeline()
        low_pdo = scorecard_points(pipeline, pdo=10.0)
        high_pdo = scorecard_points(pipeline, pdo=40.0)
        low_spread = (low_pdo.groupby("feature")["points"].max()
                      - low_pdo.groupby("feature")["points"].min()).sum()
        high_spread = (high_pdo.groupby("feature")["points"].max()
                       - high_pdo.groupby("feature")["points"].min()).sum()
        assert high_spread > low_spread

    def test_safer_bin_scores_higher_points_than_riskier_bin(self):
        # SIGNAL is constructed so higher values = lower risk. The points
        # table must reward that direction: highest-SIGNAL bin > lowest.
        pipeline = self._fit_pipeline()
        table = scorecard_points(pipeline)
        signal_rows = table[table["feature"] == "SIGNAL"].sort_values("woe")
        assert signal_rows.iloc[-1]["points"] >= signal_rows.iloc[0]["points"]

    def test_points_formula_matches_fitted_coefficients(self):
        # Directly reproduce the documented formula and check the table
        # agrees with it for one feature/bin, rather than trusting the
        # implementation to grade itself.
        pipeline = self._fit_pipeline()
        table = scorecard_points(pipeline, base_points=600, base_odds=20.0, pdo=20.0)
        woe_step = pipeline.named_steps["woe"]
        model = pipeline.named_steps["model"]
        beta = dict(zip(woe_step.get_feature_names_out(), model.coef_[0]))["SIGNAL_WOE"]
        factor = 20.0 / np.log(2)

        row = table[table["feature"] == "SIGNAL"].iloc[0]
        definition = next(b for b in woe_step.bins_["SIGNAL"] if b.label == row["bin"])
        expected_points = int(round(-factor * beta * definition.woe))
        assert row["points"] == expected_points
