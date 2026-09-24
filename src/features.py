"""Business feature engineering, population segmentation, and model-input
prep for the joined applicant-level frame produced by ``src.data``."""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.data import safe_divide

LOGGER = logging.getLogger("credit_risk")

#: Features excluded from every model run.
EXCLUDED_FEATURES = ("CODE_GENDER",)

#: Feature name -> plain-language description, used in reason codes and
#: notebook labels.
FEATURE_GLOSSARY: Dict[str, str] = {
    "DTI": "debt-to-income ratio (this loan's annual instalment vs annual income)",
    "LTI": "loan-to-income ratio (total credit vs annual income)",
    "CREDIT_TERM": "instalment as a share of total credit (implied loan tenor)",
    "TOTAL_DEBT_SERVICE_RATIO": "total debt service including existing external loans",
    "GOODS_CREDIT_RATIO": "value of financed goods relative to the credit advanced",
    "INCOME_PER_CAPITA": "income available per household member",
    "INCOME_PER_ADULT": "income available per working-age adult",
    "AGE_YEARS": "applicant age",
    "EMPLOYED_YEARS": "length of current employment",
    "EMPLOYED_RATIO": "share of life spent in current employment",
    "AGE_AT_FIRST_JOB": "age when current employment began",
    "PHONE_CHANGE_YEARS": "time since the applicant last changed phone number",
    "ID_PUBLISH_YEARS": "time since identity document was issued",
    "REGISTRATION_YEARS": "time since address registration",
    "EXT_SOURCE_MEAN": "average external credit bureau score",
    "EXT_SOURCE_MIN": "weakest external credit bureau score",
    "EXT_SOURCE_COUNT": "how many external credit scores exist for this applicant",
    "ADDRESS_MISMATCH_SCORE": "disagreement between registered, home and work addresses",
    "DOCUMENT_COUNT": "number of supporting documents provided",
    "BURO_UTILISATION": "share of externally granted credit already drawn down",
    "BURO_OVERDUE_RATIO": "overdue amount as a share of external credit",
    "BURO_ACTIVE_COUNT": "number of currently open external credits",
    "BURO_MAX_OVERDUE_MAX": "largest amount ever overdue at another lender",
    "BURO_DAYS_CREDIT_MAX": "how recently the applicant opened external credit",
    "PREV_REFUSED_RATE": "share of previous Home Credit applications that were refused",
    "PREV_GRANT_RATIO_MEAN": "how much of previous requested amounts were granted",
    "INS_LATE_RATE": "share of past instalments paid late",
    "INS_LATE_RATE_12M": "share of instalments paid late in the last 12 months",
    "INS_LATE_TREND": "recent lateness versus lifetime lateness (deterioration)",
    "INS_DPD_MAX": "worst single payment delay on record, in days",
    "INS_UNDERPAID_RATE": "share of instalments where less than the full amount was paid",
    "CC_UTILISATION_MEAN": "average credit card utilisation",
    "CC_UTILISATION_MAX": "peak credit card utilisation",
    "CC_MIN_ONLY_RATE": "share of months paying only the minimum on a credit card",
    "POS_DPD_DEF_RATE": "share of months in delinquency on point-of-sale loans",
    "BB_DELINQUENT_RATE": "share of months delinquent on external credits",
    "HISTORY_SOURCE_COUNT": "number of credit history sources available",
    "IS_THIN_FILE": "applicant has no credit history in any source",
}


def build_business_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive ratio/rate features from the joined application frame:
    affordability, capacity, stability, external assessment, and
    data-availability (HAS_*) flags."""
    out = frame

    # === Affordability ================================= #
    out["DTI"] = safe_divide(out["AMT_ANNUITY"], out["AMT_INCOME_TOTAL"])
    out["LTI"] = safe_divide(out["AMT_CREDIT"], out["AMT_INCOME_TOTAL"])
    out["CREDIT_TERM"] = safe_divide(out["AMT_ANNUITY"], out["AMT_CREDIT"])
    out["GOODS_CREDIT_RATIO"] = safe_divide(
        out["AMT_GOODS_PRICE"], out["AMT_CREDIT"]
    )

    # This loan's annuity plus existing external bureau annuities, against income.
    if "BURO_ANNUITY_SUM" in out.columns:
        total_annuity = out["AMT_ANNUITY"].fillna(0) + out["BURO_ANNUITY_SUM"].fillna(0)
        out["TOTAL_DEBT_SERVICE_RATIO"] = safe_divide(
            total_annuity, out["AMT_INCOME_TOTAL"]
        )

    # === Capacity ================================= #
    out["INCOME_PER_CAPITA"] = safe_divide(
        out["AMT_INCOME_TOTAL"], out["CNT_FAM_MEMBERS"].fillna(1).clip(lower=1)
    )
    adults = (out["CNT_FAM_MEMBERS"].fillna(1) - out["CNT_CHILDREN"].fillna(0)).clip(
        lower=1
    )
    out["INCOME_PER_ADULT"] = safe_divide(out["AMT_INCOME_TOTAL"], adults)
    out["CHILDREN_RATIO"] = safe_divide(
        out["CNT_CHILDREN"].astype(np.float32),
        out["CNT_FAM_MEMBERS"].fillna(1).clip(lower=1).astype(np.float32),
    )

    # === Stability ================================= #
    out["AGE_YEARS"] = (-out["DAYS_BIRTH"] / 365.25).astype(np.float32)
    out["EMPLOYED_YEARS"] = (-out["DAYS_EMPLOYED"] / 365.25).astype(np.float32)
    out["EMPLOYED_RATIO"] = safe_divide(out["DAYS_EMPLOYED"], out["DAYS_BIRTH"])
    out["REGISTRATION_YEARS"] = (-out["DAYS_REGISTRATION"] / 365.25).astype(np.float32)
    out["ID_PUBLISH_YEARS"] = (-out["DAYS_ID_PUBLISH"] / 365.25).astype(np.float32)
    out["PHONE_CHANGE_YEARS"] = (
        -out["DAYS_LAST_PHONE_CHANGE"] / 365.25
    ).astype(np.float32)
    out["AGE_AT_FIRST_JOB"] = (out["AGE_YEARS"] - out["EMPLOYED_YEARS"]).astype(
        np.float32
    )

    # === External assessment ====================== #
    ext_cols = [c for c in ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3") if c in out]
    if ext_cols:
        out["EXT_SOURCE_MEAN"] = out[ext_cols].mean(axis=1).astype(np.float32)
        out["EXT_SOURCE_MIN"] = out[ext_cols].min(axis=1).astype(np.float32)
        out["EXT_SOURCE_MAX"] = out[ext_cols].max(axis=1).astype(np.float32)
        out["EXT_SOURCE_STD"] = out[ext_cols].std(axis=1).astype(np.float32)
        # Count of non-null external scores.
        out["EXT_SOURCE_COUNT"] = out[ext_cols].notna().sum(axis=1).astype(np.int8)

    # === Document and contact completeness ====================== #
    doc_cols = [c for c in out.columns if c.startswith("FLAG_DOCUMENT_")]
    if doc_cols:
        out["DOCUMENT_COUNT"] = out[doc_cols].sum(axis=1).astype(np.int8)
    contact_cols = [
        c
        for c in ("FLAG_MOBIL", "FLAG_EMP_PHONE", "FLAG_WORK_PHONE", "FLAG_PHONE",
                  "FLAG_EMAIL", "FLAG_CONT_MOBILE")
        if c in out
    ]
    if contact_cols:
        out["CONTACT_COUNT"] = out[contact_cols].sum(axis=1).astype(np.int8)

    # Count of registered/living/working address mismatches.
    addr_cols = [
        c
        for c in ("REG_REGION_NOT_LIVE_REGION", "REG_REGION_NOT_WORK_REGION",
                  "LIVE_REGION_NOT_WORK_REGION", "REG_CITY_NOT_LIVE_CITY",
                  "REG_CITY_NOT_WORK_CITY", "LIVE_CITY_NOT_WORK_CITY")
        if c in out
    ]
    if addr_cols:
        out["ADDRESS_MISMATCH_SCORE"] = out[addr_cols].sum(axis=1).astype(np.int8)

    # === Data availability / thin-file segmentation =========== #
    availability = {
        "HAS_BUREAU": "BURO_COUNT",
        "HAS_BUREAU_BALANCE": "BB_MONTHS_COUNT",
        "HAS_PREV_APP": "PREV_COUNT",
        "HAS_INSTALMENTS": "INS_COUNT",
        "HAS_CREDIT_CARD": "CC_MONTHS_COUNT",
        "HAS_POS_CASH": "POS_MONTHS_COUNT",
    }
    present = []
    for flag, source in availability.items():
        if source in out.columns:
            out[flag] = out[source].notna().astype(np.int8)
            present.append(flag)
    if present:
        # Count of available history sources (0-6); IS_THIN_FILE flags zero sources.
        out["HISTORY_SOURCE_COUNT"] = out[present].sum(axis=1).astype(np.int8)
        out["IS_THIN_FILE"] = (out["HISTORY_SOURCE_COUNT"] == 0).astype(np.int8)

    return out


def build_segments(frame: pd.DataFrame) -> Dict[str, pd.Series]:
    """Build population segments for stability analysis: data-availability/
    thin-file, age band, income band, and contract type."""
    segments: Dict[str, pd.Series] = {}

    if "HISTORY_SOURCE_COUNT" in frame.columns:
        segments["data_availability"] = frame["HISTORY_SOURCE_COUNT"].map(
            lambda v: "0 sources (thin file)"
            if v == 0
            else ("1-2 sources" if v <= 2 else "3+ sources")
        )
    if "IS_THIN_FILE" in frame.columns:
        segments["thin_file"] = frame["IS_THIN_FILE"].map(
            {1: "thin file", 0: "has history"}
        )
    if "EXT_SOURCE_COUNT" in frame.columns:
        segments["ext_scores_available"] = frame["EXT_SOURCE_COUNT"].map(
            lambda v: f"{int(v)} external score(s)" if pd.notna(v) else None
        )
    if "AGE_YEARS" in frame.columns:
        segments["age_band"] = pd.cut(
            frame["AGE_YEARS"],
            bins=[0, 30, 40, 50, 60, 200],
            labels=["<30", "30-40", "40-50", "50-60", "60+"],
        ).astype(object)
    if "AMT_INCOME_TOTAL" in frame.columns:
        segments["income_band"] = pd.qcut(
            frame["AMT_INCOME_TOTAL"], q=4,
            labels=["Q1 lowest", "Q2", "Q3", "Q4 highest"], duplicates="drop",
        ).astype(object)
    if "NAME_CONTRACT_TYPE" in frame.columns:
        segments["contract_type"] = frame["NAME_CONTRACT_TYPE"].astype(object)

    return segments


def prepare_model_inputs(
    frame: pd.DataFrame, keep_gender: bool = False
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Split the engineered matrix into numeric features, target and dropped
    columns."""
    target = frame["TARGET"].astype(np.int8)
    dropped = ["TARGET", "SK_ID_CURR"]
    if not keep_gender:
        dropped.extend([c for c in EXCLUDED_FEATURES if c in frame.columns])

    features = frame.drop(columns=[c for c in dropped if c in frame.columns])
    features = features.select_dtypes(include=[np.number])

    # Zero-variance columns make the WOE binner emit degenerate single-bin features.
    constant = [c for c in features.columns if features[c].nunique(dropna=True) < 2]
    if constant:
        features = features.drop(columns=constant)
        dropped.extend(constant)

    LOGGER.info(
        "model inputs: %s rows x %d numeric features (dropped %d columns)",
        f"{len(features):,}",
        features.shape[1],
        len(dropped),
    )
    return features, target, dropped
