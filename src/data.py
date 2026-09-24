"""I/O, history-table aggregation and the pipeline configuration.

Loads the CSVs, aggregates every one-to-many history table down
to one row per SK_ID_CURR, joins them onto the application table, and adds
the DAYS_EMPLOYED sentinel fix and business features.
"""

from __future__ import annotations

import gc
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("credit_risk")

RANDOM_STATE = 42

#: DAYS_EMPLOYED sentinel value for "not employed" (365243 days).
DAYS_EMPLOYED_SENTINEL = 365243

#: Bureau-balance STATUS codes for an actively delinquent month.
#: "C" = closed, "X" = unknown, "0" = no DPD; "1".."5" are DPD buckets.
DELINQUENT_STATUSES = ("1", "2", "3", "4", "5")

_JOIN_KEY_COLUMNS = {"SK_ID_CURR", "SK_ID_BUREAU"}


@dataclass
class PipelineConfig:
    """Runtime configuration for a full pipeline run."""

    data_dir: Path
    output_dir: Path = Path("artifacts")
    sample: Optional[int] = None
    test_size: float = 0.20
    cv_splits: int = 5
    cv_repeats: int = 2
    n_woe_bins: int = 8
    min_bin_fraction: float = 0.05
    keep_gender: bool = False
    use_cache: bool = False
    use_genai_reasons: bool = False
    random_state: int = RANDOM_STATE

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def cache_path(self) -> Path:
        """Path of the cached engineered feature matrix (parquet)."""
        tag = f"sample{self.sample}" if self.sample else "full"
        return self.output_dir / f"features_{tag}.parquet"



# I/O helpers



def downcast_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Downcast numeric columns to float32/int32 to cut memory use.

    Join keys (SK_ID_CURR, SK_ID_BUREAU) are excluded: downcasting them
    per-frame can pick different dtypes across frames and break merges on
    newer pandas (worked silently on the old pinned pandas 1.4.2).
    """
    for column in frame.columns:
        if column in _JOIN_KEY_COLUMNS:
            continue
        dtype = frame[column].dtype
        if pd.api.types.is_float_dtype(dtype):
            frame[column] = frame[column].astype(np.float32)
        elif pd.api.types.is_integer_dtype(dtype):
            frame[column] = pd.to_numeric(frame[column], downcast="integer")
    return frame


def read_table(
    path: Path,
    usecols: Optional[Sequence[str]] = None,
    id_filter: Optional[np.ndarray] = None,
    id_column: str = "SK_ID_CURR",
    chunksize: int = 2_000_000,
) -> pd.DataFrame:
    """Read a competition CSV with a low memory ceiling.

    ``usecols`` narrows the read; ``id_filter`` streams in chunks and keeps
    only rows matching the given ids -- what makes ``--sample`` runs cheap
    (a 690 MB file becomes a few megabytes).
    """
    if not path.exists():
        raise FileNotFoundError(f"Expected competition file not found: {path}")

    started = time.time()
    if id_filter is None:
        frame = pd.read_csv(path, usecols=usecols)
    else:
        wanted = set(np.asarray(id_filter).tolist())
        pieces: List[pd.DataFrame] = []
        for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize):
            pieces.append(chunk[chunk[id_column].isin(wanted)])
        frame = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
        del pieces

    frame = downcast_frame(frame)
    LOGGER.info(
        "read %-28s %9s rows x %2d cols  (%.0fs, %.0f MB)",
        path.name,
        f"{len(frame):,}",
        frame.shape[1],
        time.time() - started,
        frame.memory_usage(deep=True).sum() / 1e6,
    )
    return frame


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide two Series, mapping division-by-zero/inf to NaN, not 0."""
    with np.errstate(divide="ignore", invalid="ignore"):
        result = numerator.astype(np.float64) / denominator.astype(np.float64)
    return result.replace([np.inf, -np.inf], np.nan).astype(np.float32)



# History table aggregation

# Every table except application_{train|test} is one-to-many against the
# applicant; each is grouped to one row per SK_ID_CURR before joining.
# TARGET is not referenced during aggregation.



def aggregate_bureau(
    data_dir: Path, ids: Optional[np.ndarray] = None
) -> pd.DataFrame:
    """Aggregate ``bureau.csv`` (other lenders' credit records) to one row
    per applicant, grouped by exposure, delinquency, recency and intensity."""
    cols = [
        "SK_ID_CURR",
        "SK_ID_BUREAU",
        "CREDIT_ACTIVE",
        "DAYS_CREDIT",
        "CREDIT_DAY_OVERDUE",
        "DAYS_CREDIT_ENDDATE",
        "AMT_CREDIT_MAX_OVERDUE",
        "CNT_CREDIT_PROLONG",
        "AMT_CREDIT_SUM",
        "AMT_CREDIT_SUM_DEBT",
        "AMT_CREDIT_SUM_LIMIT",
        "AMT_CREDIT_SUM_OVERDUE",
        "AMT_ANNUITY",
    ]
    bureau = read_table(data_dir / "bureau.csv", usecols=cols, id_filter=ids)
    if bureau.empty:
        return pd.DataFrame(columns=["SK_ID_CURR"])

    # AMT_CREDIT_SUM_DEBT contains negative values in the raw data; clip at 0.
    bureau["AMT_CREDIT_SUM_DEBT"] = bureau["AMT_CREDIT_SUM_DEBT"].clip(lower=0)

    bureau["_is_active"] = (bureau["CREDIT_ACTIVE"] == "Active").astype(np.int8)
    bureau["_is_closed"] = (bureau["CREDIT_ACTIVE"] == "Closed").astype(np.int8)
    bureau["_has_overdue"] = (bureau["AMT_CREDIT_SUM_OVERDUE"] > 0).astype(np.int8)

    grouped = bureau.groupby("SK_ID_CURR")
    agg = grouped.agg(
        BURO_COUNT=("SK_ID_BUREAU", "count"),
        BURO_ACTIVE_COUNT=("_is_active", "sum"),
        BURO_CLOSED_COUNT=("_is_closed", "sum"),
        BURO_OVERDUE_COUNT=("_has_overdue", "sum"),
        BURO_CREDIT_SUM_TOTAL=("AMT_CREDIT_SUM", "sum"),
        BURO_CREDIT_SUM_MAX=("AMT_CREDIT_SUM", "max"),
        BURO_DEBT_SUM_TOTAL=("AMT_CREDIT_SUM_DEBT", "sum"),
        BURO_LIMIT_SUM_TOTAL=("AMT_CREDIT_SUM_LIMIT", "sum"),
        BURO_OVERDUE_SUM_TOTAL=("AMT_CREDIT_SUM_OVERDUE", "sum"),
        BURO_MAX_OVERDUE_MAX=("AMT_CREDIT_MAX_OVERDUE", "max"),
        BURO_DAY_OVERDUE_MAX=("CREDIT_DAY_OVERDUE", "max"),
        BURO_PROLONG_SUM=("CNT_CREDIT_PROLONG", "sum"),
        BURO_DAYS_CREDIT_MAX=("DAYS_CREDIT", "max"),
        BURO_DAYS_CREDIT_MIN=("DAYS_CREDIT", "min"),
        BURO_DAYS_CREDIT_MEAN=("DAYS_CREDIT", "mean"),
        BURO_ANNUITY_SUM=("AMT_ANNUITY", "sum"),
    )

    # Debt as a share of total external credit.
    agg["BURO_UTILISATION"] = safe_divide(
        agg["BURO_DEBT_SUM_TOTAL"], agg["BURO_CREDIT_SUM_TOTAL"]
    )
    agg["BURO_OVERDUE_RATIO"] = safe_divide(
        agg["BURO_OVERDUE_SUM_TOTAL"], agg["BURO_CREDIT_SUM_TOTAL"]
    )
    agg["BURO_ACTIVE_RATIO"] = safe_divide(
        agg["BURO_ACTIVE_COUNT"].astype(np.float32),
        agg["BURO_COUNT"].astype(np.float32),
    )

    del bureau, grouped
    gc.collect()
    return downcast_frame(agg.reset_index())


def aggregate_bureau_balance(
    data_dir: Path, bureau_ids: Optional[np.ndarray] = None,
    bureau_to_curr: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Aggregate ``bureau_balance.csv`` (27.3M rows) to one row per applicant.

    Keyed on SK_ID_BUREAU, so needs ``bureau_to_curr`` to reach SK_ID_CURR
    grain; returns empty if that map isn't supplied.
    """
    if bureau_to_curr is None or bureau_to_curr.empty:
        LOGGER.warning("bureau_balance skipped: no SK_ID_BUREAU -> SK_ID_CURR map")
        return pd.DataFrame(columns=["SK_ID_CURR"])

    balance = read_table(
        data_dir / "bureau_balance.csv",
        usecols=["SK_ID_BUREAU", "MONTHS_BALANCE", "STATUS"],
        id_filter=bureau_ids,
        id_column="SK_ID_BUREAU",
    )
    if balance.empty:
        return pd.DataFrame(columns=["SK_ID_CURR"])

    balance["_delinquent"] = balance["STATUS"].isin(DELINQUENT_STATUSES).astype(np.int8)
    # STATUS "1".."5" -> severity 1..5, everything else -> 0.
    balance["_severity"] = (
        pd.to_numeric(balance["STATUS"], errors="coerce").fillna(0).astype(np.int8)
    )

    balance = balance.merge(bureau_to_curr, on="SK_ID_BUREAU", how="inner")
    grouped = balance.groupby("SK_ID_CURR")
    agg = grouped.agg(
        BB_MONTHS_COUNT=("MONTHS_BALANCE", "count"),
        BB_DELINQUENT_RATE=("_delinquent", "mean"),
        BB_DELINQUENT_MONTHS=("_delinquent", "sum"),
        BB_SEVERITY_MAX=("_severity", "max"),
        BB_SEVERITY_MEAN=("_severity", "mean"),
        BB_MONTHS_BALANCE_MIN=("MONTHS_BALANCE", "min"),
    )

    del balance, grouped
    gc.collect()
    return downcast_frame(agg.reset_index())


def aggregate_previous_application(
    data_dir: Path, ids: Optional[np.ndarray] = None
) -> pd.DataFrame:
    """Aggregate ``previous_application.csv`` (applicant's own prior Home
    Credit applications) to one row per applicant."""
    cols = [
        "SK_ID_CURR",
        "SK_ID_PREV",
        "NAME_CONTRACT_STATUS",
        "AMT_ANNUITY",
        "AMT_APPLICATION",
        "AMT_CREDIT",
        "AMT_DOWN_PAYMENT",
        "RATE_DOWN_PAYMENT",
        "DAYS_DECISION",
        "CNT_PAYMENT",
    ]
    prev = read_table(
        data_dir / "previous_application.csv", usecols=cols, id_filter=ids
    )
    if prev.empty:
        return pd.DataFrame(columns=["SK_ID_CURR"])

    status = prev["NAME_CONTRACT_STATUS"]
    prev["_approved"] = (status == "Approved").astype(np.int8)
    prev["_refused"] = (status == "Refused").astype(np.int8)
    prev["_cancelled"] = (status == "Canceled").astype(np.int8)

    # Amount actually credited versus amount applied for.
    prev["_grant_ratio"] = safe_divide(prev["AMT_CREDIT"], prev["AMT_APPLICATION"])

    grouped = prev.groupby("SK_ID_CURR")
    agg = grouped.agg(
        PREV_COUNT=("SK_ID_PREV", "count"),
        PREV_APPROVED_COUNT=("_approved", "sum"),
        PREV_REFUSED_COUNT=("_refused", "sum"),
        PREV_CANCELLED_COUNT=("_cancelled", "sum"),
        PREV_GRANT_RATIO_MEAN=("_grant_ratio", "mean"),
        PREV_GRANT_RATIO_MIN=("_grant_ratio", "min"),
        PREV_AMT_CREDIT_MEAN=("AMT_CREDIT", "mean"),
        PREV_AMT_CREDIT_MAX=("AMT_CREDIT", "max"),
        PREV_AMT_ANNUITY_MEAN=("AMT_ANNUITY", "mean"),
        PREV_DOWN_PAYMENT_RATE_MEAN=("RATE_DOWN_PAYMENT", "mean"),
        PREV_DAYS_DECISION_MAX=("DAYS_DECISION", "max"),
        PREV_DAYS_DECISION_MIN=("DAYS_DECISION", "min"),
        PREV_CNT_PAYMENT_MEAN=("CNT_PAYMENT", "mean"),
    )
    agg["PREV_REFUSED_RATE"] = safe_divide(
        agg["PREV_REFUSED_COUNT"].astype(np.float32),
        agg["PREV_COUNT"].astype(np.float32),
    )
    agg["PREV_APPROVED_RATE"] = safe_divide(
        agg["PREV_APPROVED_COUNT"].astype(np.float32),
        agg["PREV_COUNT"].astype(np.float32),
    )

    del prev, grouped
    gc.collect()
    return downcast_frame(agg.reset_index())


def aggregate_installments(
    data_dir: Path, ids: Optional[np.ndarray] = None
) -> pd.DataFrame:
    """Aggregate ``installments_payments.csv`` (13.6M rows, heaviest step)
    to applicant grain.

    ``_dpd``/``_dbd`` (days late/early) are clipped at 0 each so one doesn't
    net off the other. ``_underpaid_flag`` uses a one-cent tolerance so
    floating-point rounding on exact payments isn't misread as underpayment.
    """
    cols = [
        "SK_ID_CURR",
        "SK_ID_PREV",
        "DAYS_INSTALMENT",
        "DAYS_ENTRY_PAYMENT",
        "AMT_INSTALMENT",
        "AMT_PAYMENT",
    ]
    ins = read_table(
        data_dir / "installments_payments.csv", usecols=cols, id_filter=ids
    )
    if ins.empty:
        return pd.DataFrame(columns=["SK_ID_CURR"])

    ins["_payment_ratio"] = safe_divide(ins["AMT_PAYMENT"], ins["AMT_INSTALMENT"])
    ins["_payment_diff"] = (ins["AMT_INSTALMENT"] - ins["AMT_PAYMENT"]).astype(
        np.float32
    )
    ins["_dpd"] = (
        (ins["DAYS_ENTRY_PAYMENT"] - ins["DAYS_INSTALMENT"]).clip(lower=0)
    ).astype(np.float32)
    ins["_dbd"] = (
        (ins["DAYS_INSTALMENT"] - ins["DAYS_ENTRY_PAYMENT"]).clip(lower=0)
    ).astype(np.float32)
    ins["_late_flag"] = (ins["_dpd"] > 0).astype(np.int8)
    ins["_underpaid_flag"] = (
        (ins["AMT_PAYMENT"] + 0.01) < ins["AMT_INSTALMENT"]
    ).astype(np.int8)

    grouped = ins.groupby("SK_ID_CURR")
    agg = grouped.agg(
        INS_COUNT=("AMT_INSTALMENT", "count"),
        INS_LOAN_COUNT=("SK_ID_PREV", "nunique"),
        INS_LATE_RATE=("_late_flag", "mean"),
        INS_LATE_COUNT=("_late_flag", "sum"),
        INS_UNDERPAID_RATE=("_underpaid_flag", "mean"),
        INS_DPD_MEAN=("_dpd", "mean"),
        INS_DPD_MAX=("_dpd", "max"),
        INS_DPD_SUM=("_dpd", "sum"),
        INS_DBD_MEAN=("_dbd", "mean"),
        INS_PAYMENT_RATIO_MEAN=("_payment_ratio", "mean"),
        INS_PAYMENT_RATIO_MIN=("_payment_ratio", "min"),
        INS_PAYMENT_DIFF_MEAN=("_payment_diff", "mean"),
        INS_PAYMENT_DIFF_MAX=("_payment_diff", "max"),
        INS_AMT_PAYMENT_SUM=("AMT_PAYMENT", "sum"),
        INS_DAYS_INSTALMENT_MAX=("DAYS_INSTALMENT", "max"),
    )

    # Lateness rate restricted to the last 12 months.
    recent = ins[ins["DAYS_INSTALMENT"] >= -365]
    if not recent.empty:
        recent_agg = recent.groupby("SK_ID_CURR").agg(
            INS_LATE_RATE_12M=("_late_flag", "mean"),
            INS_DPD_MAX_12M=("_dpd", "max"),
            INS_COUNT_12M=("_late_flag", "count"),
        )
        agg = agg.join(recent_agg, how="left")
        # Recent lateness minus lifetime lateness; positive = deteriorating.
        agg["INS_LATE_TREND"] = (
            agg["INS_LATE_RATE_12M"] - agg["INS_LATE_RATE"]
        ).astype(np.float32)
        del recent, recent_agg

    del ins, grouped
    gc.collect()
    return downcast_frame(agg.reset_index())


def aggregate_credit_card(
    data_dir: Path, ids: Optional[np.ndarray] = None
) -> pd.DataFrame:
    """Aggregate ``credit_card_balance.csv`` to one row per applicant."""
    cols = [
        "SK_ID_CURR",
        "MONTHS_BALANCE",
        "AMT_BALANCE",
        "AMT_CREDIT_LIMIT_ACTUAL",
        "AMT_DRAWINGS_ATM_CURRENT",
        "AMT_DRAWINGS_CURRENT",
        "AMT_INST_MIN_REGULARITY",
        "AMT_PAYMENT_CURRENT",
        "AMT_TOTAL_RECEIVABLE",
        "CNT_DRAWINGS_CURRENT",
        "SK_DPD",
        "SK_DPD_DEF",
    ]
    card = read_table(
        data_dir / "credit_card_balance.csv", usecols=cols, id_filter=ids
    )
    if card.empty:
        return pd.DataFrame(columns=["SK_ID_CURR"])

    card["_utilisation"] = safe_divide(
        card["AMT_BALANCE"], card["AMT_CREDIT_LIMIT_ACTUAL"]
    )
    card["_min_payment_ratio"] = safe_divide(
        card["AMT_PAYMENT_CURRENT"], card["AMT_INST_MIN_REGULARITY"]
    )
    # 5% tolerance: <=1.05x minimum payment still counts as minimum-only.
    card["_min_only"] = (card["_min_payment_ratio"] <= 1.05).astype(np.int8)
    # Mask back to NaN where the ratio is undefined, instead of implying False.
    card.loc[card["_min_payment_ratio"].isna(), "_min_only"] = np.nan
    card["_atm_share"] = safe_divide(
        card["AMT_DRAWINGS_ATM_CURRENT"], card["AMT_DRAWINGS_CURRENT"]
    )
    card["_dpd_flag"] = (card["SK_DPD"] > 0).astype(np.int8)

    grouped = card.groupby("SK_ID_CURR")
    agg = grouped.agg(
        CC_MONTHS_COUNT=("MONTHS_BALANCE", "count"),
        CC_UTILISATION_MEAN=("_utilisation", "mean"),
        CC_UTILISATION_MAX=("_utilisation", "max"),
        CC_MIN_ONLY_RATE=("_min_only", "mean"),
        CC_ATM_SHARE_MEAN=("_atm_share", "mean"),
        CC_BALANCE_MEAN=("AMT_BALANCE", "mean"),
        CC_BALANCE_MAX=("AMT_BALANCE", "max"),
        CC_LIMIT_MEAN=("AMT_CREDIT_LIMIT_ACTUAL", "mean"),
        CC_DRAWINGS_MEAN=("AMT_DRAWINGS_CURRENT", "mean"),
        CC_DRAWINGS_CNT_MEAN=("CNT_DRAWINGS_CURRENT", "mean"),
        CC_RECEIVABLE_MEAN=("AMT_TOTAL_RECEIVABLE", "mean"),
        CC_DPD_RATE=("_dpd_flag", "mean"),
        CC_DPD_MAX=("SK_DPD", "max"),
        CC_DPD_DEF_MAX=("SK_DPD_DEF", "max"),
    )

    del card, grouped
    gc.collect()
    return downcast_frame(agg.reset_index())


def aggregate_pos_cash(
    data_dir: Path, ids: Optional[np.ndarray] = None
) -> pd.DataFrame:
    """Aggregate ``POS_CASH_balance.csv`` (10M rows) to one row per applicant.

    SK_DPD_DEF is the tolerance-adjusted days-past-due variant of SK_DPD.
    """
    cols = [
        "SK_ID_CURR",
        "MONTHS_BALANCE",
        "CNT_INSTALMENT",
        "CNT_INSTALMENT_FUTURE",
        "NAME_CONTRACT_STATUS",
        "SK_DPD",
        "SK_DPD_DEF",
    ]
    pos = read_table(data_dir / "POS_CASH_balance.csv", usecols=cols, id_filter=ids)
    if pos.empty:
        return pd.DataFrame(columns=["SK_ID_CURR"])

    pos["_dpd_flag"] = (pos["SK_DPD"] > 0).astype(np.int8)
    pos["_dpd_def_flag"] = (pos["SK_DPD_DEF"] > 0).astype(np.int8)
    pos["_completed"] = (pos["NAME_CONTRACT_STATUS"] == "Completed").astype(np.int8)

    grouped = pos.groupby("SK_ID_CURR")
    agg = grouped.agg(
        POS_MONTHS_COUNT=("MONTHS_BALANCE", "count"),
        POS_DPD_RATE=("_dpd_flag", "mean"),
        POS_DPD_DEF_RATE=("_dpd_def_flag", "mean"),
        POS_DPD_MAX=("SK_DPD", "max"),
        POS_DPD_DEF_MAX=("SK_DPD_DEF", "max"),
        POS_DPD_MEAN=("SK_DPD", "mean"),
        POS_COMPLETED_RATE=("_completed", "mean"),
        POS_INSTALMENT_MEAN=("CNT_INSTALMENT", "mean"),
        POS_INSTALMENT_FUTURE_MEAN=("CNT_INSTALMENT_FUTURE", "mean"),
    )

    del pos, grouped
    gc.collect()
    return downcast_frame(agg.reset_index())



# DAYS_EMPLOYED fix and full feature matrix



def fix_days_employed(frame: pd.DataFrame) -> pd.DataFrame:
    """Replace the DAYS_EMPLOYED sentinel (365243, ~18% of applicants) with
    NaN, and record it separately as DAYS_EMPLOYED_ANOM."""
    frame = frame.copy()
    anomaly = frame["DAYS_EMPLOYED"] == DAYS_EMPLOYED_SENTINEL
    frame["DAYS_EMPLOYED_ANOM"] = anomaly.astype(np.int8)
    frame.loc[anomaly, "DAYS_EMPLOYED"] = np.nan
    return frame


def build_feature_matrix(config: PipelineConfig) -> pd.DataFrame:
    """Load, aggregate, join and engineer the full applicant-level feature
    matrix.

    Each history table is aggregated and released before the next is read.
    Every join asserts SK_ID_CURR stayed unique and the row count unchanged.
    """
    # Imported here, not at module scope, to avoid a circular import with
    # src.features (which imports safe_divide from this module).
    from src.features import build_business_features

    if config.use_cache and config.cache_path.exists():
        LOGGER.info("loading cached feature matrix from %s", config.cache_path)
        return pd.read_parquet(config.cache_path)

    started = time.time()
    application = read_table(config.data_dir / "application_train.csv")

    if config.sample:
        application = application.sample(
            n=min(config.sample, len(application)), random_state=config.random_state
        ).reset_index(drop=True)
        LOGGER.info("subsampled to %s applicants", f"{len(application):,}")

    ids = application["SK_ID_CURR"].to_numpy()
    n_rows = len(application)
    application = fix_days_employed(application)

    # Read once more here for the SK_ID_BUREAU -> SK_ID_CURR map bureau_balance needs.
    bureau_map = read_table(
        config.data_dir / "bureau.csv",
        usecols=["SK_ID_CURR", "SK_ID_BUREAU"],
        id_filter=ids,
    )
    bureau_ids = bureau_map["SK_ID_BUREAU"].to_numpy() if not bureau_map.empty else None

    aggregates: List[pd.DataFrame] = [
        aggregate_bureau(config.data_dir, ids),
        aggregate_bureau_balance(config.data_dir, bureau_ids, bureau_map),
        aggregate_previous_application(config.data_dir, ids),
        aggregate_installments(config.data_dir, ids),
        aggregate_credit_card(config.data_dir, ids),
        aggregate_pos_cash(config.data_dir, ids),
    ]
    del bureau_map, bureau_ids
    gc.collect()

    frame = application
    for agg in aggregates:
        if agg.empty or "SK_ID_CURR" not in agg.columns:
            continue
        frame = frame.merge(agg, on="SK_ID_CURR", how="left")
        assert frame["SK_ID_CURR"].is_unique, "join fanned out -- grain broken"
        assert len(frame) == n_rows, "join changed the applicant row count"
    del aggregates, application
    gc.collect()

    frame = build_business_features(frame)
    frame = downcast_frame(frame)

    LOGGER.info(
        "feature matrix ready: %s applicants x %d columns (%.0fs, %.0f MB)",
        f"{len(frame):,}",
        frame.shape[1],
        time.time() - started,
        frame.memory_usage(deep=True).sum() / 1e6,
    )
    try:
        frame.to_parquet(config.cache_path, index=False)
        LOGGER.info("cached feature matrix -> %s", config.cache_path)
    except Exception as exc:  # pragma: no cover - parquet engine optional
        LOGGER.warning("could not cache feature matrix (%s)", exc)
    return frame
