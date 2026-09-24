"""Weight of Evidence binning, Information Value selection, and the
points-based scorecard built on top of a fitted WOE + logistic pipeline.

    WOE(bin) = ln( P(bin | non-default) / P(bin | default) )
    IV = sum over bins of ( P(bin|non-default) - P(bin|default) ) * WOE

    IV: <0.02 useless | 0.02-0.10 weak | 0.10-0.30 medium | 0.30-0.50 strong
    | >0.50 unusually strong; investigate for leakage/provenance
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.data import safe_divide

LOGGER = logging.getLogger("credit_risk")

#: Points-based scorecard calibration. PDO = "points to double the odds".
SCORECARD_BASE_POINTS = 600
SCORECARD_BASE_ODDS = 20.0
SCORECARD_PDO = 20.0


@dataclass
class BinDefinition:
    """A single WOE bin for one feature."""

    label: str
    lower: float
    upper: float
    is_missing: bool
    woe: float
    event_rate: float
    count: int


class WOETransformer(BaseEstimator, TransformerMixin):
    """Bin numeric features and replace each value with its Weight of
    Evidence. Fit inside a scikit-learn ``Pipeline`` (training folds only)."""

    def __init__(
        self,
        n_bins: int = 8,
        min_bin_fraction: float = 0.05,
        monotonic: bool = True,
        smoothing: float = 0.5,
    ) -> None:
        self.n_bins = n_bins
        self.min_bin_fraction = min_bin_fraction
        self.monotonic = monotonic
        self.smoothing = smoothing

    # -- fitting helpers ================================= #

    def _initial_edges(self, values: pd.Series) -> np.ndarray:
        """Quantile bin edges for one feature, duplicates dropped."""
        quantiles = np.linspace(0, 1, self.n_bins + 1)[1:-1]
        edges = np.unique(np.nanquantile(values, quantiles))
        return edges

    def _bin_stats(
        self, codes: pd.Series, target: pd.Series, n_bins: int
    ) -> pd.DataFrame:
        """Tabulate event/non-event counts per bin code."""
        table = pd.DataFrame({"code": codes.to_numpy(), "y": target.to_numpy()})
        stats = table.groupby("code")["y"].agg(["sum", "count"])
        stats = stats.reindex(range(n_bins), fill_value=0)
        stats.columns = ["events", "count"]
        stats["non_events"] = stats["count"] - stats["events"]
        stats["event_rate"] = safe_divide(
            stats["events"].astype(np.float32), stats["count"].astype(np.float32)
        )
        return stats

    def _merge_small_and_monotonic(
        self, edges: np.ndarray, values: pd.Series, target: pd.Series
    ) -> np.ndarray:
        """Merge bins below ``min_bin_fraction``, then merge adjacent bins
        (weakest violation first) until the event rate is monotonic."""
        total = len(values)
        min_count = max(int(total * self.min_bin_fraction), 1)

        def assign(current_edges: np.ndarray) -> Tuple[pd.Series, int]:
            full = np.concatenate(([-np.inf], current_edges, [np.inf]))
            codes = pd.Series(
                np.digitize(values.to_numpy(), current_edges, right=False),
                index=values.index,
            )
            return codes, len(full) - 1

        # -- size pass --
        changed = True
        while changed and len(edges) > 0:
            changed = False
            codes, n_bins = assign(edges)
            stats = self._bin_stats(codes, target, n_bins)
            small = stats.index[stats["count"] < min_count].tolist()
            if small:
                # Drop the edge adjacent to the smallest bin.
                worst = int(stats["count"].idxmin())
                drop_index = min(max(worst - 1, 0), len(edges) - 1)
                edges = np.delete(edges, drop_index)
                changed = True

        if not self.monotonic:
            return edges

        # -- monotonic pass --
        while len(edges) > 1:
            codes, n_bins = assign(edges)
            stats = self._bin_stats(codes, target, n_bins)
            rates = stats["event_rate"].to_numpy()
            valid = ~np.isnan(rates)
            if valid.sum() < 3:
                break
            diffs = np.diff(rates[valid])
            increasing = np.all(diffs >= 0)
            decreasing = np.all(diffs <= 0)
            if increasing or decreasing:
                break
            # Drop the edge with the smallest gap between neighbouring event rates.
            gap = np.abs(diffs)
            drop_index = int(np.argmin(gap))
            drop_index = min(drop_index, len(edges) - 1)
            edges = np.delete(edges, drop_index)
        return edges

    # -- sklearn API ================================= #

    def fit(self, X: pd.DataFrame, y: Iterable[int]) -> "WOETransformer":
        """Learn bin edges and WOE values for every column of ``X``."""
        X = pd.DataFrame(X).copy()
        target = pd.Series(np.asarray(y), index=X.index).astype(np.int8)

        total_events = float(target.sum())
        total_non_events = float(len(target) - total_events)
        if total_events == 0 or total_non_events == 0:
            raise ValueError("WOETransformer requires both classes to be present")

        self.feature_names_in_ = list(X.columns)
        self.bins_: Dict[str, List[BinDefinition]] = {}
        self.iv_: Dict[str, float] = {}

        for column in X.columns:
            series = X[column]
            observed = series.dropna()
            definitions: List[BinDefinition] = []

            if observed.nunique() < 2:
                # Constant (or all-missing) feature: one degenerate bin, IV 0.
                self.bins_[column] = [
                    BinDefinition("all", -np.inf, np.inf, False, 0.0, np.nan, len(series))
                ]
                self.iv_[column] = 0.0
                continue

            edges = self._initial_edges(observed)
            if len(edges) > 0:
                edges = self._merge_small_and_monotonic(
                    edges, observed, target.loc[observed.index]
                )

            codes = pd.Series(
                np.digitize(observed.to_numpy(), edges, right=False),
                index=observed.index,
            )
            stats = self._bin_stats(codes, target.loc[observed.index], len(edges) + 1)

            iv = 0.0
            bounds = np.concatenate(([-np.inf], edges, [np.inf]))
            for code in range(len(edges) + 1):
                row = stats.loc[code]
                events = float(row["events"]) + self.smoothing
                non_events = float(row["non_events"]) + self.smoothing
                dist_events = events / (total_events + self.smoothing * len(stats))
                dist_non = non_events / (total_non_events + self.smoothing * len(stats))
                woe = float(np.log(dist_non / dist_events))
                iv += (dist_non - dist_events) * woe
                definitions.append(
                    BinDefinition(
                        label=f"[{bounds[code]:.4g}, {bounds[code + 1]:.4g})",
                        lower=float(bounds[code]),
                        upper=float(bounds[code + 1]),
                        is_missing=False,
                        woe=woe,
                        event_rate=float(row["event_rate"]),
                        count=int(row["count"]),
                    )
                )

            # Missing gets its own bin with an empirically estimated WOE.
            missing_mask = series.isna()
            if missing_mask.any():
                m_events = float(target[missing_mask].sum()) + self.smoothing
                m_non = float(missing_mask.sum() - target[missing_mask].sum()) + self.smoothing
                dist_events = m_events / (total_events + self.smoothing)
                dist_non = m_non / (total_non_events + self.smoothing)
                woe = float(np.log(dist_non / dist_events))
                iv += (dist_non - dist_events) * woe
                definitions.append(
                    BinDefinition(
                        label="MISSING",
                        lower=np.nan,
                        upper=np.nan,
                        is_missing=True,
                        woe=woe,
                        event_rate=float(target[missing_mask].mean()),
                        count=int(missing_mask.sum()),
                    )
                )

            self.bins_[column] = definitions
            self.iv_[column] = float(iv)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Replace each value with the WOE of the bin it falls into."""
        X = pd.DataFrame(X)
        output = pd.DataFrame(index=X.index)

        for column in self.feature_names_in_:
            definitions = self.bins_[column]
            value_bins = [b for b in definitions if not b.is_missing]
            missing_bin = next((b for b in definitions if b.is_missing), None)
            missing_woe = missing_bin.woe if missing_bin is not None else 0.0

            series = X[column] if column in X.columns else pd.Series(np.nan, index=X.index)
            result = np.full(len(series), missing_woe, dtype=np.float32)

            if value_bins:
                edges = np.array([b.upper for b in value_bins[:-1]], dtype=np.float64)
                woes = np.array([b.woe for b in value_bins], dtype=np.float32)
                observed = series.notna().to_numpy()
                if observed.any():
                    codes = np.digitize(
                        series.to_numpy(dtype=np.float64)[observed], edges, right=False
                    )
                    codes = np.clip(codes, 0, len(woes) - 1)
                    result[observed] = woes[codes]

            output[f"{column}_WOE"] = result

        return output

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Transformed feature names, for sklearn compatibility."""
        return np.array([f"{c}_WOE" for c in self.feature_names_in_])

    def iv_table(self) -> pd.DataFrame:
        """Information Value ranking of all fitted features, with a Siddiqi
        ``strength`` label, sorted descending."""
        table = pd.DataFrame(
            {"feature": list(self.iv_.keys()), "iv": list(self.iv_.values())}
        ).sort_values("iv", ascending=False, ignore_index=True)

        def label(value: float) -> str:
            if value < 0.02:
                return "useless"
            if value < 0.10:
                return "weak"
            if value < 0.30:
                return "medium"
            if value < 0.50:
                return "strong"
            return "unusually strong (review provenance)"

        table["strength"] = table["iv"].map(label)
        return table

    def bin_table(self, feature: str) -> pd.DataFrame:
        """Fitted bin detail for one feature, bin by bin (for notebook review)."""
        if feature not in self.bins_:
            raise KeyError(f"{feature!r} was not fitted")
        return pd.DataFrame(
            [
                {
                    "bin": b.label,
                    "count": b.count,
                    "event_rate": b.event_rate,
                    "woe": b.woe,
                }
                for b in self.bins_[feature]
            ]
        )


def select_by_iv(
    transformer: WOETransformer,
    min_iv: float = 0.02,
    max_iv: float = 0.60,
    top_n: Optional[int] = None,
    exclude: Optional[Sequence[str]] = None,
) -> List[str]:
    """Select features with iv >= min_iv. Features with iv > max_iv are
    logged, not removed, unless explicitly named in ``exclude``."""
    table = transformer.iv_table()
    kept = table[table["iv"] >= min_iv]
    if exclude:
        kept = kept[~kept["feature"].isin(set(exclude))]
    flagged = kept[kept["iv"] > max_iv]["feature"].tolist()
    if flagged:
        LOGGER.warning(
            "high-IV features retained pending provenance review: %s", flagged
        )
    names = kept["feature"].tolist()
    if top_n is not None:
        names = names[:top_n]
    return names


def scorecard_points(
    pipeline: Pipeline,
    base_points: int = SCORECARD_BASE_POINTS,
    base_odds: float = SCORECARD_BASE_ODDS,
    pdo: float = SCORECARD_PDO,
) -> pd.DataFrame:
    """Convert fitted WOE coefficients into an integer points scorecard::

        factor      = PDO / ln(2)
        offset      = base_points - factor * ln(base_odds)
        BASE SCORE  = offset - factor * intercept
        points(bin) = -factor * beta_i * WOE_bin

        applicant score = BASE SCORE + sum of one bin's points per feature

    Base score is attached as ``frame.attrs["base_score"]`` and also
    returned by :func:`scorecard_base_score`.
    """
    woe: WOETransformer = pipeline.named_steps["woe"]
    model: LogisticRegression = pipeline.named_steps["model"]
    if not hasattr(model, "coef_"):
        raise ValueError("pipeline must be fitted before extracting a scorecard")

    factor = pdo / np.log(2)
    offset = base_points - factor * np.log(base_odds)

    names = list(woe.get_feature_names_out())
    coefficients = dict(zip(names, model.coef_[0]))
    intercept = float(model.intercept_[0])

    base_score = offset - factor * intercept

    rows: List[Dict[str, object]] = []
    for feature in woe.feature_names_in_:
        beta = float(coefficients.get(f"{feature}_WOE", 0.0))
        for definition in woe.bins_[feature]:
            points = -factor * beta * definition.woe
            rows.append(
                {
                    "feature": feature,
                    "bin": definition.label,
                    "count": definition.count,
                    "event_rate": definition.event_rate,
                    "woe": definition.woe,
                    "coefficient": beta,
                    "points": int(round(points)),
                }
            )

    table = pd.DataFrame(rows)
    # points_spread = how much a variable can move an applicant's score.
    spread = (
        table.groupby("feature")["points"]
        .agg(lambda s: s.max() - s.min())
        .rename("points_spread")
    )
    table = table.merge(spread, on="feature").sort_values(
        ["points_spread", "feature", "points"], ascending=[False, True, True],
        ignore_index=True,
    )
    table.attrs["base_score"] = int(round(base_score))
    table.attrs["factor"] = float(factor)
    table.attrs["offset"] = float(offset)
    table.attrs["intercept"] = intercept
    return table


def scorecard_base_score(
    pipeline: Pipeline,
    base_points: int = SCORECARD_BASE_POINTS,
    base_odds: float = SCORECARD_BASE_ODDS,
    pdo: float = SCORECARD_PDO,
) -> int:
    """The scorecard's base score: what a hypothetical applicant at WOE = 0
    on every characteristic would score, before any feature points."""
    model: LogisticRegression = pipeline.named_steps["model"]
    if not hasattr(model, "coef_"):
        raise ValueError("pipeline must be fitted before extracting a scorecard")
    factor = pdo / np.log(2)
    offset = base_points - factor * np.log(base_odds)
    return int(round(offset - factor * float(model.intercept_[0])))


def scorecard_applicant_points(
    pipeline: Pipeline, scorecard: pd.DataFrame, X: pd.DataFrame, row_index: int
) -> pd.DataFrame:
    """Break one applicant's scorecard total into per-feature points, using
    the same bin/points table as :func:`scorecard_points`."""
    woe: WOETransformer = pipeline.named_steps["woe"]
    row = X.iloc[row_index]

    rows: List[Dict[str, object]] = []
    for feature in woe.feature_names_in_:
        value = row.get(feature, np.nan)
        definitions = woe.bins_[feature]

        if pd.isna(value):
            chosen = next((b for b in definitions if b.is_missing), None)
        else:
            chosen = None
            for definition in definitions:
                if definition.is_missing:
                    continue
                if definition.lower <= float(value) < definition.upper:
                    chosen = definition
                    break
        if chosen is None:
            continue

        match = scorecard[
            (scorecard["feature"] == feature) & (scorecard["bin"] == chosen.label)
        ]
        if match.empty:
            continue
        rows.append(
            {
                "feature": feature,
                "value": value,
                "bin": chosen.label,
                "bin_default_rate": chosen.event_rate,
                "points": int(match.iloc[0]["points"]),
            }
        )

    table = pd.DataFrame(rows)
    if table.empty:
        return table
    # Relative measure, unaffected by where the intercept is carried.
    median_points = table["points"].median()
    table["points_vs_median"] = table["points"] - median_points
    table = table.sort_values("points_vs_median", ignore_index=True)
    # total score = base score + sum(feature points)
    table.attrs["base_score"] = int(scorecard.attrs.get("base_score", 0))
    return table
