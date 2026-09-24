"""Model builders (WOE+logistic scorecard, gradient boosting), evaluation
with the stability suite, and SHAP/native tree-contribution explainability."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
import xgboost as xgb
from src.data import RANDOM_STATE
from src.evaluation import bootstrap_auc_ci, population_stability_index
from src.features import FEATURE_GLOSSARY
from src.woe import WOETransformer

LOGGER = logging.getLogger("credit_risk")



# Core: model builders and evaluation



def build_scorecard_model(
    n_bins: int = 8, min_bin_fraction: float = 0.05, C: float = 1.0
) -> Pipeline:
    """Build the WOE + logistic regression scorecard."""
    return Pipeline(
        [
            (
                "woe",
                WOETransformer(
                    n_bins=n_bins, min_bin_fraction=min_bin_fraction, monotonic=True
                ),
            ),
            (
                "model",
                LogisticRegression(
                    C=C,
                    penalty="l2",
                    solver="liblinear",
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def build_boosting_model(scale_pos_weight: float) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=4,
        min_child_weight=50,
        gamma=1.0,
        subsample=0.8,
        colsample_bytree=0.6,
        reg_alpha=0.5,
        reg_lambda=5.0,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
    )
@dataclass
class EvaluationResult:
    """Container for one model's evaluation output."""

    name: str
    cv_auc_mean: float = np.nan
    cv_auc_std: float = np.nan
    cv_fold_aucs: List[float] = field(default_factory=list)
    train_auc: float = np.nan
    valid_auc: float = np.nan
    valid_pr_auc: float = np.nan
    auc_ci: Tuple[float, float] = (np.nan, np.nan)
    n_features: int = 0
    extra: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON-serialisable summary of the result."""
        return {
            "name": self.name,
            "cv_auc_mean": self.cv_auc_mean,
            "cv_auc_std": self.cv_auc_std,
            "cv_fold_aucs": self.cv_fold_aucs,
            "train_auc": self.train_auc,
            "valid_auc": self.valid_auc,
            "valid_pr_auc": self.valid_pr_auc,
            "auc_ci_low": self.auc_ci[0],
            "auc_ci_high": self.auc_ci[1],
            "train_valid_gap": self.train_auc - self.valid_auc,
            "n_features": self.n_features,
            **self.extra,
        }


def repeated_cv_auc(
    estimator,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    n_repeats: int = 2,
    random_state: int = RANDOM_STATE,
) -> Tuple[float, float, List[float]]:
    """Repeated stratified CV AUC across multiple random partitions."""
    cv = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
    )
    scores: List[float] = []
    for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y), start=1):
        model = clone(estimator)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        proba = model.predict_proba(X.iloc[valid_idx])[:, 1]
        auc = roc_auc_score(y.iloc[valid_idx], proba)
        scores.append(float(auc))
        LOGGER.info("  fold %2d/%d  AUC %.5f", fold, n_splits * n_repeats, auc)
    return float(np.mean(scores)), float(np.std(scores)), scores


def evaluate_model(
    name: str,
    estimator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    cv_splits: int = 5,
    cv_repeats: int = 2,
    run_cv: bool = True,
) -> Tuple[EvaluationResult, object, np.ndarray]:
    """Fit an estimator and produce its full stability profile (CV AUC,
    train/valid gap, bootstrap CI, PSI). ``run_cv=False`` skips CV for fast
    iteration."""
    LOGGER.info("=" * 68)
    LOGGER.info("evaluating: %s", name)
    result = EvaluationResult(name=name, n_features=X_train.shape[1])

    if run_cv:
        LOGGER.info("repeated stratified CV (%d x %d)", cv_splits, cv_repeats)
        mean, std, folds = repeated_cv_auc(
            estimator, X_train, y_train, cv_splits, cv_repeats
        )
        result.cv_auc_mean, result.cv_auc_std, result.cv_fold_aucs = mean, std, folds
        LOGGER.info("CV AUC %.5f +/- %.5f", mean, std)

    started = time.time()
    fitted = clone(estimator)
    fitted.fit(X_train, y_train)
    LOGGER.info("final fit in %.0fs", time.time() - started)

    train_proba = fitted.predict_proba(X_train)[:, 1]
    valid_proba = fitted.predict_proba(X_valid)[:, 1]

    result.train_auc = float(roc_auc_score(y_train, train_proba))
    result.valid_auc = float(roc_auc_score(y_valid, valid_proba))
    result.valid_pr_auc = float(average_precision_score(y_valid, valid_proba))
    result.auc_ci = bootstrap_auc_ci(y_valid, valid_proba)
    result.extra["psi_train_vs_valid"] = population_stability_index(
        train_proba, valid_proba
    )

    LOGGER.info(
        "train %.5f | valid %.5f | gap %+.5f | PR-AUC %.5f | CI [%.5f, %.5f] | PSI %.4f",
        result.train_auc,
        result.valid_auc,
        result.train_auc - result.valid_auc,
        result.valid_pr_auc,
        result.auc_ci[0],
        result.auc_ci[1],
        result.extra["psi_train_vs_valid"],
    )
    return result, fitted, valid_proba



# Analysis / optional: SHAP and native tree-contribution explainability



def boosting_contributions(
    fitted_model, X: pd.DataFrame, row_index: int
) -> pd.Series:
    """Exact per-feature SHAP contributions for one applicant, via the
    booster's native ``pred_contribs`` path (no ``shap`` dependency needed --
    XGBoost/LightGBM compute exact TreeSHAP internally)."""
    row = X.iloc[[row_index]]

    if hasattr(fitted_model, "get_booster"):  # XGBoost
        import xgboost as xgb

        booster = fitted_model.get_booster()
        matrix = xgb.DMatrix(row, feature_names=list(X.columns))
        contributions = booster.predict(matrix, pred_contribs=True)[0]
        names = list(X.columns) + ["_BIAS"]
    elif hasattr(fitted_model, "booster_"):  # LightGBM
        contributions = fitted_model.predict(row, pred_contrib=True)[0]
        names = list(X.columns) + ["_BIAS"]
    else:
        raise TypeError(
            f"{type(fitted_model).__name__} exposes no SHAP contribution API"
        )

    series = pd.Series(contributions, index=names, dtype=np.float64)
    return series.reindex(series.abs().sort_values(ascending=False).index)


def shap_explain(
    fitted_model,
    X: pd.DataFrame,
    max_rows: int = 2_000,
    random_state: int = RANDOM_STATE,
) -> Dict[str, object]:
    """Compute SHAP values for the boosting model, on a stratified subsample
    of at most ``max_rows`` rows. Returns ``{"available": False}`` if
    ``shap`` isn't installed or the model isn't a supported tree ensemble."""
    try:
        import shap
    except ImportError:  # pragma: no cover
        LOGGER.warning("shap not installed -- falling back to native contributions")
        return {"available": False}

    if not hasattr(fitted_model, "get_booster") and not hasattr(fitted_model, "booster_"):
        LOGGER.warning("model is not a supported tree ensemble -- skipping SHAP")
        return {"available": False}

    sample = X
    if len(X) > max_rows:
        sample = X.sample(n=max_rows, random_state=random_state)

    started = time.time()
    # tree_path_dependent is required, not just preferred: the default
    # "interventional" mode's background-dataset allocation (218 features x
    # 400 trees) exhausted 8 GB and killed the process mid-run.
    explainer = shap.TreeExplainer(
        fitted_model, feature_perturbation="tree_path_dependent"
    )
    values = explainer.shap_values(sample, check_additivity=False)
    # Binary classifiers may return a list of two arrays (one per class) or a
    # single array for the positive class, depending on backend and version.
    if isinstance(values, list):
        values = values[1] if len(values) == 2 else values[0]
    values = np.asarray(values)
    if values.ndim == 3:  # (rows, features, classes)
        values = values[:, :, -1]

    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = float(np.asarray(base_value).ravel()[-1])

    importance = (
        pd.DataFrame(
            {
                "feature": sample.columns,
                "mean_abs_shap": np.abs(values).mean(axis=0),
                "mean_shap": values.mean(axis=0),
            }
        )
        .sort_values("mean_abs_shap", ascending=False, ignore_index=True)
    )
    importance["direction"] = np.where(
        importance["mean_shap"] > 0, "increases risk", "decreases risk"
    )
    importance["business_meaning"] = importance["feature"].map(
        lambda f: FEATURE_GLOSSARY.get(f, "")
    )

    LOGGER.info(
        "SHAP computed on %s rows in %.0fs", f"{len(sample):,}", time.time() - started
    )
    return {
        "available": True,
        "importance": importance,
        "values": values,
        "sample": sample,
        "base_value": float(base_value),
    }


def shap_explain_applicant(
    shap_output: Dict[str, object], row_position: int, top_n: int = 8
) -> pd.DataFrame:
    """Extract one applicant's SHAP attribution from a ``shap_explain`` result.
    Empty DataFrame if SHAP was unavailable."""
    if not shap_output.get("available"):
        return pd.DataFrame()

    values = shap_output["values"]
    sample: pd.DataFrame = shap_output["sample"]
    row = sample.iloc[row_position]

    table = pd.DataFrame(
        {
            "feature": sample.columns,
            "value": row.to_numpy(),
            "shap": values[row_position],
        }
    )
    table["direction"] = np.where(table["shap"] > 0, "increases risk", "decreases risk")
    table["business_meaning"] = table["feature"].map(
        lambda f: FEATURE_GLOSSARY.get(f, "")
    )
    return (
        table.reindex(table["shap"].abs().sort_values(ascending=False).index)
        .head(top_n)
        .reset_index(drop=True)
    )
