"""End-to-end pipeline for the Home Credit default-risk analysis.

The pipeline trains two complementary models:
- WOE + Logistic Regression for an interpretable scorecard.
- Gradient Boosting as a nonlinear ranking challenger.

Feature engineering, modeling and evaluation are implemented in ``src/``,
re-exported here so ``import credit_risk_model as crm`` keeps working for
notebook use. This module handles configuration, orchestration, CLI
execution and writing output artifacts.

Reason-code generation is deterministic by default, with an optional
OpenAI-based wording layer enabled via ``--genai-reasons``.

Usage
-----
    python credit_risk_model.py --sample 30000
    python credit_risk_model.py --output-dir artifacts
    python credit_risk_model.py --use-cache
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd

# Re-exported explicitly rather than by wildcard, so ``crm.<name>`` in the
# notebook keeps working while every name's origin stays traceable.
from src.data import (
    DAYS_EMPLOYED_SENTINEL,
    RANDOM_STATE,
    PipelineConfig,
    build_feature_matrix,
    fix_days_employed,
    read_table,
    safe_divide,
)
from src.features import (
    EXCLUDED_FEATURES,
    FEATURE_GLOSSARY,
    build_business_features,
    build_segments,
    prepare_model_inputs,
)
from src.woe import (
    SCORECARD_BASE_ODDS,
    SCORECARD_BASE_POINTS,
    SCORECARD_PDO,
    WOETransformer,
    scorecard_applicant_points,
    scorecard_base_score,
    scorecard_points,
    select_by_iv,
)
from src.models import (
    EvaluationResult,
    boosting_contributions,
    build_boosting_model,
    build_scorecard_model,
    evaluate_model,
    repeated_cv_auc,
    shap_explain,
    shap_explain_applicant,
)
from src.evaluation import (
    BusinessAssumptions,
    bootstrap_auc_ci,
    business_impact_simulation,
    capacity_analysis,
    compare_model_agreement,
    format_impact_simulation,
    population_stability_index,
    process_comparison,
    segment_performance,
    threshold_economics,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

LOGGER = logging.getLogger("credit_risk")

#: Gen AI configuration. Key is read from ``.env`` and never logged/persisted.
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")
OPENAI_KEY_NAME = "OPENAI_API_KEY"


def load_environment(dotenv_path: Optional[Path] = None) -> bool:
    """Load ``.env`` and report whether the OpenAI key is present. Never
    logs or persists the credential."""
    try:
        from dotenv import load_dotenv

        if dotenv_path is not None:
            load_dotenv(dotenv_path=str(dotenv_path), override=False)
        else:
            load_dotenv(override=False)
    except ImportError:
        LOGGER.warning(
            "python-dotenv not installed; relying on already-exported variables"
        )

    available = bool(os.environ.get(OPENAI_KEY_NAME))
    LOGGER.info(
        "Gen AI credential (%s): %s",
        OPENAI_KEY_NAME,
        "found" if available else "NOT found -- deterministic fallbacks will be used",
    )
    return available


def setup_logging(verbose: bool = True) -> None:
    """Configure root logging with a compact, timestamped format."""
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )



# Reason codes: deterministic by default, GenAI optional.



def deterministic_reason_codes(contributions: pd.DataFrame, top_n: int) -> List[str]:
    """Build reason-code strings directly from the ``top_n`` lowest-scoring
    point contributions, no LLM involved."""
    worst = contributions.nsmallest(top_n, "points_vs_median")
    reasons = []
    for _, row in worst.iterrows():
        reasons.append(
            f"{row['feature']}: value {row['value']!r} falls in band {row['bin']} "
            f"(observed default rate {row['bin_default_rate']:.1%}), "
            f"costing {abs(int(row['points_vs_median']))} points versus a typical applicant."
        )
    return reasons


def generate_reason_codes(
    contributions: pd.DataFrame,
    score: int,
    decision: str,
    top_n: int = 4,
    use_genai: bool = False,
    model: str = OPENAI_MODEL,
    feature_glossary: Optional[Dict[str, str]] = None,
) -> Dict[str, object]:
    """Turn scorecard point contributions into adverse-action reason strings.

    Defaults to ``deterministic_reason_codes``. If ``use_genai=True``, sends
    only the computed attributions (never the raw applicant record) to the
    OpenAI API, falling back to the deterministic path on any failure.
    """
    if contributions.empty:
        return {"reasons": [], "source": "deterministic", "narrative": ""}

    if not use_genai:
        return {
            "reasons": deterministic_reason_codes(contributions, top_n),
            "narrative": "",
            "source": "deterministic",
        }

    worst = contributions.nsmallest(top_n, "points_vs_median")
    glossary = feature_glossary or FEATURE_GLOSSARY
    evidence = [
        {
            "feature": row["feature"],
            "meaning": glossary.get(row["feature"], row["feature"]),
            "applicant_band": row["bin"],
            "band_default_rate": f"{row['bin_default_rate']:.1%}",
            "points_lost": abs(int(row["points_vs_median"])),
        }
        for _, row in worst.iterrows()
    ]

    if not os.environ.get(OPENAI_KEY_NAME):
        LOGGER.info("%s not set -- using deterministic reason codes", OPENAI_KEY_NAME)
        return {
            "reasons": deterministic_reason_codes(contributions, top_n),
            "narrative": "",
            "source": "deterministic",
        }

    try:  # pragma: no cover - requires network + credentials
        from openai import OpenAI

        client = OpenAI()  # reads OPENAI_API_KEY from the environment
        system = (
            "You write adverse-action explanations for consumer loan applicants "
            "at Home Credit Indonesia. You are a presentation layer over an "
            "already-final, auditable scorecard decision. You never invent a "
            "factor that was not supplied to you, never cite gender, age, "
            "ethnicity, marital status or any protected attribute as a reason, "
            "and never state or imply that a decision is final or that "
            "repayment is guaranteed. Reply with JSON only."
        )
        user = (
            f"Decision: {decision}\n"
            f"Applicant scorecard total: {score} points\n\n"
            "Factors that reduced this applicant's score the most, as computed "
            "by the points-based scorecard:\n"
            f"{json.dumps(evidence, indent=2)}\n\n"
            "Write exactly one short reason per factor, in plain language the "
            "borrower can act on: state the factor and why it lowered the "
            "assessment. Then add one closing sentence naming what would most "
            "improve a future application.\n\n"
            'Return JSON of the form {"reasons": [string], "narrative": string}'
        )
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        payload = json.loads(response.choices[0].message.content or "{}")
        return {
            "reasons": payload.get("reasons", []),
            "narrative": payload.get("narrative", ""),
            "source": "openai",
            "model": model,
        }
    except ImportError:
        LOGGER.info("openai SDK not installed -- using deterministic reason codes")
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("OpenAI reason-code generation failed (%s) -- using fallback", exc)

    return {
        "reasons": deterministic_reason_codes(contributions, top_n),
        "narrative": "",
        "source": "deterministic",
    }



# Orchestration



def run_pipeline(config: PipelineConfig) -> Dict[str, object]:
    """Run the credit-risk pipeline and write evaluation artifacts."""
    overall_start = time.time()
    frame = build_feature_matrix(config)
    features, target, dropped = prepare_model_inputs(frame, config.keep_gender)

    X_train, X_valid, y_train, y_valid = train_test_split(
        features,
        target,
        test_size=config.test_size,
        stratify=target,
        random_state=config.random_state,
    )
    LOGGER.info(
        "split: train %s / valid %s (positive rate %.4f / %.4f)",
        f"{len(X_train):,}",
        f"{len(X_valid):,}",
        y_train.mean(),
        y_valid.mean(),
    )

    # --- Feature selection by Information Value ------------------------- #
    LOGGER.info("fitting WOE on training split for IV-based feature selection")
    iv_scout = WOETransformer(
        n_bins=config.n_woe_bins, min_bin_fraction=config.min_bin_fraction
    ).fit(X_train, y_train)
    iv_table = iv_scout.iv_table()
    scorecard_features = select_by_iv(iv_scout, min_iv=0.02, max_iv=0.60, top_n=30)
    LOGGER.info(
        "IV selection kept %d of %d features for the scorecard",
        len(scorecard_features),
        X_train.shape[1],
    )
    # Flag unusually high IV for provenance review; do not remove automatically.
    high_iv_features = iv_table[iv_table["iv"] > 0.60]["feature"].tolist()

    results: Dict[str, object] = {
        "config": {
            "sample": config.sample,
            "test_size": config.test_size,
            "cv": f"{config.cv_splits}x{config.cv_repeats}",
            "keep_gender": config.keep_gender,
            "n_applicants": int(len(features)),
            "n_features_total": int(features.shape[1]),
            "n_features_scorecard": len(scorecard_features),
            "positive_rate": float(target.mean()),
        },
        "iv_top25": iv_table.head(25).to_dict("records"),
        "high_iv_features": high_iv_features,
        "dropped_columns": dropped[:50],
    }

    # --- Model 1: Scorecard Logistic Regression ------ #
    scorecard_pipeline = build_scorecard_model(
        n_bins=config.n_woe_bins, min_bin_fraction=config.min_bin_fraction
    )
    sc_result, sc_fitted, sc_proba = evaluate_model(
        "Scorecard Logistic Regression",
        scorecard_pipeline,
        X_train[scorecard_features],
        y_train,
        X_valid[scorecard_features],
        y_valid,
        config.cv_splits,
        config.cv_repeats,
    )
    results["scorecard"] = sc_result.to_dict()

    scorecard_table = scorecard_points(sc_fitted)
    scorecard_table.to_csv(config.output_dir / "scorecard_points.csv", index=False)
    results["scorecard_features"] = scorecard_features
    results["scorecard_top_spread"] = (
        scorecard_table.drop_duplicates("feature")
        .head(15)[["feature", "points_spread"]]
        .to_dict("records")
    )

    # --- Model 2: Gradient Boosting ------------ #
    scale_pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    boosting = build_boosting_model(scale_pos_weight)
    bo_result, bo_fitted, bo_proba = evaluate_model(
        "Gradient Boosting",
        boosting,
        X_train,
        y_train,
        X_valid,
        y_valid,
        config.cv_splits,
        config.cv_repeats,
    )
    results["boosting"] = bo_result.to_dict()

    importance = pd.DataFrame(
        {
            "feature": X_train.columns,
            "importance": getattr(bo_fitted, "feature_importances_", np.zeros(X_train.shape[1])),
        }
    ).sort_values("importance", ascending=False, ignore_index=True)
    importance.to_csv(config.output_dir / "boosting_importance.csv", index=False)
    results["boosting_top20"] = importance.head(20).to_dict("records")

    # --- Stability: per-segment performance ------------------ #
    segments = build_segments(frame.loc[X_valid.index])
    for label, proba in (("scorecard", sc_proba), ("boosting", bo_proba)):
        table = segment_performance(X_valid, y_valid, proba, segments)
        table.to_csv(config.output_dir / f"segment_performance_{label}.csv", index=False)
        results[f"segments_{label}"] = table.to_dict("records")

    # --- Operating point economics ------- #
    thresholds = threshold_economics(y_valid, bo_proba)
    thresholds.to_csv(config.output_dir / "threshold_economics.csv", index=False)
    best = thresholds.loc[thresholds["total_cost"].idxmin()]
    results["cost_optimal_threshold"] = best.to_dict()
    results["capacity_analysis"] = capacity_analysis(y_valid, bo_proba).to_dict("records")

    # --- Champion / challenger ------ #
    results["agreement"] = compare_model_agreement(sc_proba, bo_proba, y_valid)

    # --- SHAP interpretability ------ #
    # SHAP previously OOM-killed the process here; guarded so it stays optional.
    try:
        shap_output = shap_explain(bo_fitted, X_train)
        if shap_output.get("available"):
            shap_table = shap_output["importance"]
            shap_table.to_csv(config.output_dir / "shap_importance.csv", index=False)
            results["shap_top20"] = shap_table.head(20).to_dict("records")
    except BaseException as exc:  # noqa: BLE001 - includes MemoryError
        LOGGER.warning("SHAP stage failed (%s) -- continuing without it", exc)
        results["shap_error"] = str(exc)

    # --- Business impact: before vs after ------- #
    # bo_proba drives ranking here; sc_fitted/scorecard produce the reason codes below.
    assumptions = BusinessAssumptions()
    simulation = business_impact_simulation(y_valid, bo_proba, assumptions)
    results["impact_simulation"] = simulation
    results["process_comparison"] = process_comparison().to_dict("records")
    process_comparison().to_csv(
        config.output_dir / "process_before_after.csv", index=False
    )
    with open(config.output_dir / "impact_simulation.txt", "w") as handle:
        handle.write(format_impact_simulation(simulation))

    # --- Worked adverse-action explanation ------- #
    # Highest predicted-risk applicant in the validation set. Guarded: this
    # calls an external API when use_genai_reasons is set.
    try:
        worst_position = int(np.argmax(sc_proba))
        contributions = scorecard_applicant_points(
            sc_fitted, scorecard_table, X_valid[scorecard_features], worst_position
        )
        if not contributions.empty:
            base = int(scorecard_table.attrs.get("base_score", 0))
            feature_points = int(contributions["points"].sum())
            total_points = base + feature_points
            reasons = generate_reason_codes(
                contributions,
                score=total_points,
                decision="refer for manual review (high risk score)",
                use_genai=config.use_genai_reasons,
            )
            results["example_explanation"] = {
                "scorecard_base_score": base,
                "scorecard_feature_points": feature_points,
                "scorecard_total_points": total_points,
                "risk_score_uncalibrated": float(sc_proba[worst_position]),
                "actual_target": int(y_valid.iloc[worst_position]),
                "contributions": contributions.head(10).to_dict("records"),
                "reason_codes": reasons,
            }
    except BaseException as exc:  # noqa: BLE001
        LOGGER.warning("explanation stage failed (%s) -- continuing", exc)
        results["explanation_error"] = str(exc)

    results["runtime_seconds"] = round(time.time() - overall_start, 1)

    output_path = config.output_dir / "results.json"
    with open(output_path, "w") as handle:
        json.dump(results, handle, indent=2, default=str)
    LOGGER.info("wrote %s", output_path)

    print_summary(results)
    return results


def print_summary(results: Dict[str, object]) -> None:
    """Print a human-readable summary of a completed run."""
    sc = results.get("scorecard", {})
    bo = results.get("boosting", {})

    print("\n" + "=" * 74)
    print("HOME CREDIT DEFAULT RISK -- TWO-MODEL RESULT SUMMARY")
    print("=" * 74)
    config = results.get("config", {})
    print(
        f"applicants {config.get('n_applicants'):,} | "
        f"features {config.get('n_features_total')} | "
        f"positive rate {config.get('positive_rate', 0):.4f}"
    )
    print("-" * 74)
    header = f"{'model':<32}{'CV AUC':>16}{'valid AUC':>12}{'gap':>9}"
    print(header)
    for label, block in (("Scorecard LR", sc), ("Gradient Boosting", bo)):
        if not block:
            continue
        print(
            f"{label:<32}"
            f"{block.get('cv_auc_mean', float('nan')):.5f} "
            f"+/- {block.get('cv_auc_std', float('nan')):.5f}"
            f"{block.get('valid_auc', float('nan')):>12.5f}"
            f"{block.get('train_valid_gap', float('nan')):>+9.4f}"
        )
    print("-" * 74)
    agreement = results.get("agreement", {})
    if agreement:
        print(
            f"model agreement {agreement.get('agreement_rate', 0):.1%} | "
            f"rank corr {agreement.get('rank_correlation', float('nan')):.3f} | "
            f"disagreement {agreement.get('disagreement_share', 0):.1%}"
        )
    print(f"runtime {results.get('runtime_seconds')}s")
    print("=" * 74 + "\n")

    simulation = results.get("impact_simulation")
    if simulation:
        print(format_impact_simulation(simulation))


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Home Credit two-model credit risk pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("home-credit-default-risk"),
        help="Directory containing the competition CSV files",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Subsample N applicants for a fast run (recommended: 30000)",
    )
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--cv-repeats", type=int, default=2)
    parser.add_argument("--n-woe-bins", type=int, default=8)
    parser.add_argument(
        "--keep-gender",
        action="store_true",
        help="Retain CODE_GENDER (only to quantify the fairness trade-off)",
    )
    parser.add_argument("--use-cache", action="store_true")
    parser.add_argument(
        "--genai-reasons",
        action="store_true",
        help="Use the OpenAI API for adverse-action reason codes instead of "
             "the deterministic template",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command-line entry point. Returns 0 on success, 1 on a handled failure."""
    args = parse_args(argv)
    setup_logging(verbose=not args.quiet)
    load_environment()

    config = PipelineConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        sample=args.sample,
        test_size=args.test_size,
        cv_splits=args.cv_splits,
        cv_repeats=args.cv_repeats,
        n_woe_bins=args.n_woe_bins,
        keep_gender=args.keep_gender,
        use_cache=args.use_cache,
        use_genai_reasons=args.genai_reasons,
    )

    try:
        run_pipeline(config)
    except FileNotFoundError as exc:
        LOGGER.error("%s", exc)
        LOGGER.error("Pass --data-dir pointing at the competition CSV directory.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
