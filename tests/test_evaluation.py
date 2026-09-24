"""Tests for src/evaluation.py: stability metrics, champion/challenger
comparison, and the business impact simulation. The business_impact_
simulation tests in particular guard the central invariant the whole
business case rests on -- approval and review volume held constant."""

import numpy as np
import pandas as pd
import pytest

from src.evaluation import (
    BusinessAssumptions,
    bootstrap_auc_ci,
    business_impact_simulation,
    capacity_analysis,
    compare_model_agreement,
    population_stability_index,
)


def rng():
    return np.random.default_rng(42)


class TestPopulationStabilityIndex:
    def test_identical_distributions_near_zero(self):
        r = rng()
        dist = r.normal(0, 1, size=5000)
        psi = population_stability_index(dist, dist.copy())
        assert psi < 0.01

    def test_shifted_distribution_flags_as_significant(self):
        r = rng()
        expected = r.normal(0, 1, size=5000)
        actual = r.normal(3, 1, size=5000)  # big shift
        psi = population_stability_index(expected, actual)
        assert psi > 0.25  # conventional "significant shift" threshold

    def test_larger_shift_gives_larger_psi(self):
        r = rng()
        expected = r.normal(0, 1, size=3000)
        small_shift = population_stability_index(expected, r.normal(0.3, 1, size=3000))
        big_shift = population_stability_index(expected, r.normal(2.0, 1, size=3000))
        assert big_shift > small_shift


class TestBootstrapAucCI:
    def test_ci_bounds_are_ordered_and_in_unit_interval(self):
        r = rng()
        y_true = r.integers(0, 2, size=500)
        y_score = r.uniform(0, 1, size=500)
        low, high = bootstrap_auc_ci(y_true, y_score, n_boot=100, random_state=1)
        assert 0.0 <= low <= high <= 1.0

    def test_perfectly_separable_scores_give_high_ci(self):
        y_true = np.array([0] * 250 + [1] * 250)
        y_score = np.array([0.1] * 250 + [0.9] * 250)  # perfect separation
        low, high = bootstrap_auc_ci(y_true, y_score, n_boot=100, random_state=1)
        assert low > 0.95


class TestCapacityAnalysis:
    def _synthetic(self, n=4000, seed=7):
        r = np.random.default_rng(seed)
        y_score = r.uniform(0, 1, size=n)
        p_default = 0.3 * y_score  # higher score -> higher default risk
        y_true = (r.uniform(0, 1, size=n) < p_default).astype(int)
        return y_true, y_score

    def test_recall_is_non_decreasing_with_capacity(self):
        y_true, y_score = self._synthetic()
        table = capacity_analysis(y_true, y_score, capacities=(0.05, 0.10, 0.20, 0.30))
        recalls = table["recall"].tolist()
        assert recalls == sorted(recalls)

    def test_reviewed_count_scales_with_capacity(self):
        y_true, y_score = self._synthetic(n=10_000)
        table = capacity_analysis(y_true, y_score, capacities=(0.10, 0.20))
        row10 = table[table["review_capacity"] == "top 10%"].iloc[0]
        row20 = table[table["review_capacity"] == "top 20%"].iloc[0]
        assert row20["reviewed"] > row10["reviewed"]

    def test_lift_exceeds_one_when_score_is_informative(self):
        y_true, y_score = self._synthetic(n=10_000)
        table = capacity_analysis(y_true, y_score, capacities=(0.05,))
        assert table.iloc[0]["lift_vs_base_rate"] > 1.0


class TestCompareModelAgreement:
    def test_cell_counts_sum_to_total(self):
        r = rng()
        n = 2000
        sc = r.uniform(0, 1, size=n)
        bo = r.uniform(0, 1, size=n)
        y = r.integers(0, 2, size=n)
        result = compare_model_agreement(sc, bo, pd.Series(y))
        total = sum(c["n"] for c in result["cells"].values())
        assert total == n

    def test_identical_models_have_zero_disagreement(self):
        r = rng()
        n = 1000
        proba = r.uniform(0, 1, size=n)
        y = r.integers(0, 2, size=n)
        result = compare_model_agreement(proba, proba.copy(), pd.Series(y))
        assert result["disagreement_share"] == pytest.approx(0.0, abs=1e-9)
        assert result["agreement_rate"] == pytest.approx(1.0, abs=1e-9)


class TestBusinessImpactSimulation:
    def _synthetic(self, n=5000, seed=3):
        # y_score strongly (but not perfectly) predicts y_true, so ranking
        # by it should demonstrably improve the approved book.
        r = np.random.default_rng(seed)
        true_risk = r.uniform(0, 1, size=n)
        y_true = (r.uniform(0, 1, size=n) < 0.08 + 0.5 * true_risk).astype(int)
        y_score = np.clip(true_risk + r.normal(0, 0.05, size=n), 0, 1)
        return y_true, y_score

    def test_approved_volume_held_constant(self):
        y_true, y_score = self._synthetic()
        sim = business_impact_simulation(y_true, y_score, BusinessAssumptions())
        assert sim["before"]["approved"] == sim["after"]["approved"]

    def test_reviewed_volume_held_constant(self):
        y_true, y_score = self._synthetic()
        sim = business_impact_simulation(y_true, y_score, BusinessAssumptions())
        assert sim["before"]["reviewed"] == sim["after"]["reviewed"]

    def test_informative_score_reduces_defaults_in_approved_book(self):
        y_true, y_score = self._synthetic()
        sim = business_impact_simulation(y_true, y_score, BusinessAssumptions())
        assert sim["after"]["defaults_in_book"] <= sim["before"]["defaults_in_book"]

    def test_informative_score_improves_review_precision(self):
        y_true, y_score = self._synthetic()
        sim = business_impact_simulation(y_true, y_score, BusinessAssumptions())
        assert sim["after"]["review_precision"] >= sim["before"]["review_precision"]

    def test_custom_assumptions_are_reflected_in_output(self):
        y_true, y_score = self._synthetic(n=1000)
        custom = BusinessAssumptions(monthly_applications=12_345)
        sim = business_impact_simulation(y_true, y_score, custom)
        assert sim["annualised"]["monthly_applications"] == 12_345
