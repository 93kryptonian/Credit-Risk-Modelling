"""Stability metrics (bootstrap CI, PSI, per-segment AUC), threshold/capacity
economics, champion/challenger comparison, and the business impact
simulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.data import RANDOM_STATE



# Core: stability metrics and champion/challenger comparison



def bootstrap_auc_ci(
    y_true: Iterable[int],
    y_score: Iterable[float],
    n_boot: int = 400,
    alpha: float = 0.05,
    random_state: int = RANDOM_STATE,
) -> Tuple[float, float]:
    """Bootstrap a confidence interval for ROC-AUC via resampling with
    replacement."""
    rng = np.random.default_rng(random_state)
    y_true = np.asarray(list(y_true))
    y_score = np.asarray(list(y_score))
    n = len(y_true)
    scores: List[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        sample_y = y_true[idx]
        if sample_y.min() == sample_y.max():
            continue  # degenerate resample, no positives or no negatives
        scores.append(roc_auc_score(sample_y, y_score[idx]))
    if not scores:
        return (np.nan, np.nan)
    return (
        float(np.quantile(scores, alpha / 2)),
        float(np.quantile(scores, 1 - alpha / 2)),
    )


def population_stability_index(
    expected: Iterable[float], actual: Iterable[float], n_bins: int = 10
) -> float:
    """Population Stability Index between two score distributions::

        PSI = sum over bins of ( actual% - expected% ) * ln( actual% / expected% )

    Conventional bands: <0.10 stable, 0.10-0.25 moderate shift, >0.25
    significant shift.
    """
    expected = np.asarray(list(expected), dtype=np.float64)
    actual = np.asarray(list(actual), dtype=np.float64)

    edges = np.unique(np.nanquantile(expected, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    expected_pct = np.histogram(expected, bins=edges)[0] / max(len(expected), 1)
    actual_pct = np.histogram(actual, bins=edges)[0] / max(len(actual), 1)

    # Floor at epsilon so an empty bin gives a finite contribution.
    eps = 1e-6
    expected_pct = np.clip(expected_pct, eps, None)
    actual_pct = np.clip(actual_pct, eps, None)

    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def segment_performance(
    frame: pd.DataFrame,
    y_true: pd.Series,
    y_score: np.ndarray,
    segments: Dict[str, pd.Series],
    min_size: int = 200,
) -> pd.DataFrame:
    """AUC and default rate within each population segment. Groups below
    ``min_size`` are skipped (AUC on a tiny group is noisy)."""
    overall = roc_auc_score(y_true, y_score)
    rows: List[Dict[str, object]] = []

    for segment_name, labels in segments.items():
        labels = labels.reindex(frame.index)
        for group in pd.unique(labels.dropna()):
            mask = (labels == group).to_numpy()
            if mask.sum() < min_size:
                continue
            group_y = y_true[mask]
            if group_y.nunique() < 2:
                continue
            auc = roc_auc_score(group_y, y_score[mask])
            rows.append(
                {
                    "segment": segment_name,
                    "group": str(group),
                    "n": int(mask.sum()),
                    "default_rate": float(group_y.mean()),
                    "auc": float(auc),
                    "auc_vs_overall": float(auc - overall),
                }
            )

    return pd.DataFrame(rows).sort_values(
        ["segment", "auc"], ascending=[True, False], ignore_index=True
    )


def compare_model_agreement(
    scorecard_proba: np.ndarray,
    boosting_proba: np.ndarray,
    y_true: pd.Series,
    quantile: float = 0.80,
) -> Dict[str, object]:
    """Compare where the scorecard and boosting model agree/disagree on a
    high-risk flag at ``quantile``, and report AUC within the disagreement
    subset."""
    sc_cut = float(np.quantile(scorecard_proba, quantile))
    bo_cut = float(np.quantile(boosting_proba, quantile))
    sc_flag = scorecard_proba >= sc_cut
    bo_flag = boosting_proba >= bo_cut

    y = np.asarray(y_true)
    cells = {
        "both_flag_high_risk": sc_flag & bo_flag,
        "scorecard_only": sc_flag & ~bo_flag,
        "boosting_only": ~sc_flag & bo_flag,
        "both_clear": ~sc_flag & ~bo_flag,
    }

    summary = {}
    for label, mask in cells.items():
        summary[label] = {
            "n": int(mask.sum()),
            "share": float(mask.mean()),
            "default_rate": float(y[mask].mean()) if mask.any() else np.nan,
        }

    disagreement = cells["scorecard_only"] | cells["boosting_only"]
    result: Dict[str, object] = {
        "quantile": quantile,
        "scorecard_cutoff": sc_cut,
        "boosting_cutoff": bo_cut,
        "rank_correlation": float(
            pd.Series(scorecard_proba).corr(pd.Series(boosting_proba), method="spearman")
        ),
        "agreement_rate": float((sc_flag == bo_flag).mean()),
        "cells": summary,
        "disagreement_share": float(disagreement.mean()),
    }

    if disagreement.any() and pd.Series(y[disagreement]).nunique() == 2:
        result["auc_within_disagreement_boosting"] = float(
            roc_auc_score(y[disagreement], boosting_proba[disagreement])
        )
        result["auc_within_disagreement_scorecard"] = float(
            roc_auc_score(y[disagreement], scorecard_proba[disagreement])
        )
    return result



# Analysis / optional: threshold/capacity economics, business impact



def threshold_economics(
    y_true: Iterable[int],
    y_score: Iterable[float],
    cost_false_negative: float = 5_000.0,
    cost_false_positive: float = 500.0,
    thresholds: Optional[Sequence[float]] = None,
) -> pd.DataFrame:
    """Sweep decision thresholds and compute total cost at each, given the
    FN/FP cost parameters."""
    y_true = np.asarray(list(y_true))
    y_score = np.asarray(list(y_score))
    if thresholds is None:
        thresholds = np.round(np.arange(0.05, 0.96, 0.05), 2)

    rows: List[Dict[str, object]] = []
    positives = int(y_true.sum())
    for threshold in thresholds:
        predicted = (y_score >= threshold).astype(np.int8)
        tp = int(((y_true == 1) & (predicted == 1)).sum())
        fp = int(((y_true == 0) & (predicted == 1)).sum())
        fn = int(((y_true == 1) & (predicted == 0)).sum())
        flagged = tp + fp
        rows.append(
            {
                "threshold": float(threshold),
                "precision": tp / flagged if flagged else 0.0,
                "recall": tp / positives if positives else 0.0,
                "flagged": flagged,
                "flagged_pct": 100.0 * flagged / len(y_true),
                "false_negatives": fn,
                "false_positives": fp,
                "total_cost": fn * cost_false_negative + fp * cost_false_positive,
            }
        )
    return pd.DataFrame(rows)


def capacity_analysis(
    y_true: Iterable[int],
    y_score: Iterable[float],
    capacities: Sequence[float] = (0.05, 0.10, 0.15, 0.20, 0.30),
) -> pd.DataFrame:
    """Evaluate the model under a fixed review-capacity constraint: for each
    capacity level, how many defaults are caught among the highest-scored
    applicants, and the resulting lift over the base default rate."""
    y_true = np.asarray(list(y_true))
    y_score = np.asarray(list(y_score))
    base_rate = float(y_true.mean())
    positives = int(y_true.sum())

    rows: List[Dict[str, object]] = []
    for capacity in capacities:
        cutoff = float(np.quantile(y_score, 1 - capacity))
        selected = y_score >= cutoff
        caught = int(y_true[selected].sum())
        reviewed = int(selected.sum())
        precision = caught / reviewed if reviewed else 0.0
        rows.append(
            {
                "review_capacity": f"top {capacity * 100:.0f}%",
                "implied_threshold": cutoff,
                "reviewed": reviewed,
                "defaults_caught": caught,
                "recall": caught / positives if positives else 0.0,
                "precision": precision,
                "lift_vs_base_rate": precision / base_rate if base_rate else np.nan,
            }
        )
    return pd.DataFrame(rows)


@dataclass
class BusinessAssumptions:
    """Explicit, challengeable economic assumptions for the impact simulation.

    Every number is an input, not a finding, gathered here so a reviewer can
    change one and see how much of the conclusion depends on it. They must
    still be replaced with Home Credit's actual portfolio economics before
    any real decision is taken -- but two of them are now grounded in public
    data rather than picked freehand, and the two that can't be sourced are
    labelled as such below.

    Sourced:
      * ``loss_given_default`` (0.65) -- unsecured personal loans typically
        run 50-70% LGD industry-wide, vs. 10-40% for secured/mortgage/auto
        (https://en.wikipedia.org/wiki/Loss_given_default). 0.65 sits
        mid-range.
      * ``manual_review_cost`` (IDR 25,000/file) -- an Indonesian Credit
        Analyst earns ~IDR 6-9M/month (Jobstreet Indonesia salary guide,
        https://id.jobstreet.com/id/career-advice/role/credit-analyst/salary).
        At the IDR 7.5M midpoint, 21 working days/month and an assumed 15
        files reviewed/day: 7,500,000 / (21*15) = IDR 23,810, rounded to
        25,000. The salary is sourced; the throughput (15/day) is a stated
        assumption a reviewer can override.
      * ``avg_loan_principal`` (IDR 25,000,000) -- Home Credit Indonesia's
        own published product ranges run IDR 10,000-10,000,000 for goods/POS
        financing and up to IDR 100-150,000,000 for multiguna cash loans.
        25M sits between the two bands as a plausible blend, but the true
        weighted average depends on Home Credit's product mix, which isn't
        public -- treat this one as informed, not exact.

    Not sourceable -- internal operating parameters of a specific private
    company that no public filing discloses. Modelled scenario assumptions,
    not measured data: ``review_capacity_pct``, ``baseline_approval_rate``,
    ``baseline_review_precision``, ``monthly_applications``.
    """

    avg_loan_principal: float = 25_000_000.0
    loss_given_default: float = 0.65
    margin_per_good_loan: float = 3_500_000.0
    manual_review_cost: float = 25_000.0
    review_capacity_pct: float = 0.15
    baseline_approval_rate: float = 0.85
    baseline_review_precision: float = 0.12
    monthly_applications: int = 50_000

    @property
    def loss_per_default(self) -> float:
        """Expected monetary loss from a single defaulted loan."""
        return self.avg_loan_principal * self.loss_given_default

    def to_dict(self) -> Dict[str, float]:
        """Return the assumptions as a plain dict for reporting."""
        return {
            "avg_loan_principal": self.avg_loan_principal,
            "loss_given_default": self.loss_given_default,
            "loss_per_default": self.loss_per_default,
            "margin_per_good_loan": self.margin_per_good_loan,
            "manual_review_cost": self.manual_review_cost,
            "review_capacity_pct": self.review_capacity_pct,
            "baseline_approval_rate": self.baseline_approval_rate,
            "baseline_review_precision": self.baseline_review_precision,
            "monthly_applications": self.monthly_applications,
        }


def process_comparison() -> pd.DataFrame:
    """Return a stage-by-stage description of the underwriting process
    before and after the model, as a DataFrame."""
    stages = [
        {
            "stage": "1. Application intake",
            "before": "Application captured; no risk signal attached.",
            "after": "Same intake, plus a scorecard total and a risk score "
                     "(uncalibrated, used for ranking) computed within seconds.",
            "what_changes": "Risk becomes visible at the point of intake "
                            "instead of after review.",
        },
        {
            "stage": "2. Triage / queueing",
            "before": "First-in-first-out, or simple policy rules "
                      "(age, income floor, document completeness).",
            "after": "Queue ordered by predicted risk. Low-risk applications "
                     "route to fast-track, high-risk to senior review.",
            "what_changes": "Review capacity is aimed at risk rather than "
                            "spent in arrival order.",
        },
        {
            "stage": "3. Credit assessment",
            "before": "Analyst reads the file and forms a judgement. External "
                      "bureau data consulted manually, if at all.",
            "after": "Analyst starts from a scorecard breakdown showing which "
                     "factors moved the score and by how many points, "
                     "including aggregated bureau, instalment, card and POS "
                     "history the analyst could not read manually.",
            "what_changes": "Analyst time shifts from data gathering to "
                            "judgement on genuinely marginal cases.",
        },
        {
            "stage": "4. Decision",
            "before": "Approve / decline. Consistency varies between analysts "
                      "and across branches.",
            "after": "Approve / refer / decline against a documented score "
                     "cut-off, applied identically to every applicant. "
                     "Agreement between both models marks a low-touch / "
                     "straight-through candidate, never an automatic approval.",
            "what_changes": "Decisions become reproducible and auditable; "
                            "analyst-to-analyst variance drops.",
        },
        {
            "stage": "5. Applicant communication",
            "before": "Generic decline notice, or none. Applicant does not "
                      "learn what to improve.",
            "after": "Specific reason codes generated from the scorecard "
                     "attribution, in plain language.",
            "what_changes": "Declines become actionable for the applicant and "
                            "defensible to a regulator.",
        },
        {
            "stage": "6. Portfolio monitoring",
            "before": "Default rate observed after the fact, by vintage.",
            "after": "Score distribution and PSI tracked monthly; drift "
                     "triggers revalidation before losses appear.",
            "what_changes": "Risk management moves from lagging to leading "
                            "indicators.",
        },
        {
            "stage": "7. Thin-file applicants",
            "before": "Declined or heavily manually scrutinised by default, "
                      "because no bureau score exists.",
            "after": "Scored on alternative signals (instalment behaviour, "
                     "POS delinquency, address and document stability), with "
                     "the model's lower confidence on this segment explicitly "
                     "flagged.",
            "what_changes": "The mission population becomes assessable rather "
                            "than automatically excluded.",
        },
    ]
    return pd.DataFrame(stages)


def business_impact_simulation(
    y_true: Iterable[int],
    y_score: Iterable[float],
    assumptions: Optional[BusinessAssumptions] = None,
) -> Dict[str, object]:
    """Simulate portfolio outcomes before and after model deployment."""
    assumptions = assumptions or BusinessAssumptions()
    y_true = np.asarray(list(y_true))
    y_score = np.asarray(list(y_score))
    n = len(y_true)
    base_rate = float(y_true.mean())

    n_approve = int(round(n * assumptions.baseline_approval_rate))
    n_review = int(round(n * assumptions.review_capacity_pct))

    # =========== BEFORE: no risk ranking =========== #
    # Uninformed approval -> approved book's default rate == population rate.
    before_defaults = base_rate * n_approve
    before_good = n_approve - before_defaults
    before_credit_loss = before_defaults * assumptions.loss_per_default
    before_margin = before_good * assumptions.margin_per_good_loan
    before_review_cost = n_review * assumptions.manual_review_cost
    before_caught = n_review * assumptions.baseline_review_precision
    before_profit = before_margin - before_credit_loss - before_review_cost

    # =========== AFTER: model-ranked =========== #
    # Approve the n_approve LOWEST-risk applicants; review the n_review
    # HIGHEST-risk. Same volumes, different ordering.
    order = np.argsort(y_score)  # ascending risk
    approved_idx = order[:n_approve]
    reviewed_idx = order[::-1][:n_review]

    after_defaults = float(y_true[approved_idx].sum())
    after_good = n_approve - after_defaults
    after_credit_loss = after_defaults * assumptions.loss_per_default
    after_margin = after_good * assumptions.margin_per_good_loan
    after_review_cost = n_review * assumptions.manual_review_cost
    after_caught = float(y_true[reviewed_idx].sum())
    after_profit = after_margin - after_credit_loss - after_review_cost

    approved_default_rate_after = after_defaults / max(n_approve, 1)
    review_precision_after = after_caught / max(n_review, 1)

    before = {
        "process": "Uninformed approval + arrival-order review",
        "applications": n,
        "approved": n_approve,
        "approved_default_rate": base_rate,
        "defaults_in_book": before_defaults,
        "credit_loss": before_credit_loss,
        "margin_earned": before_margin,
        "reviewed": n_review,
        "review_precision": assumptions.baseline_review_precision,
        "defaults_caught_in_review": before_caught,
        "review_cost": before_review_cost,
        "net_profit": before_profit,
    }
    after = {
        "process": "Model-ranked approval + risk-ordered review",
        "applications": n,
        "approved": n_approve,
        "approved_default_rate": approved_default_rate_after,
        "defaults_in_book": after_defaults,
        "credit_loss": after_credit_loss,
        "margin_earned": after_margin,
        "reviewed": n_review,
        "review_precision": review_precision_after,
        "defaults_caught_in_review": after_caught,
        "review_cost": after_review_cost,
        "net_profit": after_profit,
    }

    defaults_avoided = before_defaults - after_defaults
    delta = {
        "defaults_avoided": defaults_avoided,
        "defaults_avoided_pct": (
            defaults_avoided / before_defaults if before_defaults else np.nan
        ),
        "approved_default_rate_change": approved_default_rate_after - base_rate,
        "credit_loss_saved": before_credit_loss - after_credit_loss,
        "extra_margin": after_margin - before_margin,
        "review_precision_lift": (
            review_precision_after / assumptions.baseline_review_precision
            if assumptions.baseline_review_precision
            else np.nan
        ),
        "extra_defaults_caught_in_review": after_caught - before_caught,
        "net_profit_change": after_profit - before_profit,
    }

    # Scale the held-out sample up to the stated monthly volume, then annualise.
    scale = assumptions.monthly_applications / max(n, 1)
    annualised = {
        "scale_factor": scale,
        "monthly_applications": assumptions.monthly_applications,
        "annual_defaults_avoided": defaults_avoided * scale * 12,
        "annual_credit_loss_saved": delta["credit_loss_saved"] * scale * 12,
        "annual_net_profit_change": delta["net_profit_change"] * scale * 12,
    }

    return {
        "assumptions": assumptions.to_dict(),
        "before": before,
        "after": after,
        "delta": delta,
        "annualised": annualised,
    }


def format_impact_simulation(simulation: Dict[str, object], currency: str = "IDR") -> str:
    """Render the impact simulation as an aligned text table."""
    before = simulation["before"]
    after = simulation["after"]
    delta = simulation["delta"]
    annual = simulation["annualised"]

    def money(value: float) -> str:
        """Format a large currency amount in billions."""
        return f"{currency} {value / 1e9:,.2f} B"

    lines = [
        "=" * 78,
        "BUSINESS IMPACT SIMULATION -- BEFORE vs AFTER THE MODEL",
        "=" * 78,
        f"{'metric':<38}{'BEFORE':>19}{'AFTER':>19}",
        "-" * 78,
        f"{'Applications assessed':<38}{before['applications']:>19,}{after['applications']:>19,}",
        f"{'Loans approved (held constant)':<38}{before['approved']:>19,}{after['approved']:>19,}",
        f"{'Default rate in approved book':<38}"
        f"{before['approved_default_rate']:>18.2%}{after['approved_default_rate']:>19.2%}",
        f"{'Defaults in approved book':<38}"
        f"{before['defaults_in_book']:>19,.0f}{after['defaults_in_book']:>19,.0f}",
        f"{'Credit loss':<38}{money(before['credit_loss']):>19}{money(after['credit_loss']):>19}",
        f"{'Margin earned':<38}{money(before['margin_earned']):>19}{money(after['margin_earned']):>19}",
        f"{'Review precision':<38}"
        f"{before['review_precision']:>18.1%}{after['review_precision']:>19.1%}",
        f"{'Net profit':<38}{money(before['net_profit']):>19}{money(after['net_profit']):>19}",
        "-" * 78,
        "IMPACT ON THE HELD-OUT SAMPLE",
        f"  defaults avoided in the approved book : {delta['defaults_avoided']:,.0f} "
        f"({delta['defaults_avoided_pct']:.1%} of the baseline)",
        f"  approved-book default rate            : "
        f"{delta['approved_default_rate_change']:+.2%}",
        f"  credit loss avoided                   : {money(delta['credit_loss_saved'])}",
        f"  review precision lift                 : "
        f"{delta['review_precision_lift']:.1f}x",
        f"  net profit change                     : {money(delta['net_profit_change'])}",
        "-" * 78,
        f"SCALED TO {annual['monthly_applications']:,} APPLICATIONS/MONTH, ANNUALISED",
        f"  defaults avoided per year   : {annual['annual_defaults_avoided']:,.0f}",
        f"  credit loss avoided per year: {money(annual['annual_credit_loss_saved'])}",
        f"  net profit change per year  : {money(annual['annual_net_profit_change'])}",
        "=" * 78,
        "All monetary figures follow from the stated assumptions and are",
        "illustrative. Replace them with Home Credit's actual portfolio",
        "economics before using this to support a decision.",
        "=" * 78,
    ]
    return "\n".join(lines)
