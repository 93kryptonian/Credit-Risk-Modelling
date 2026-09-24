# Home Credit Default Risk — Credit Risk Model

A two-model credit risk pipeline for the [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk)
Kaggle competition, built around a business framing rather than a leaderboard
score: **rank loan applicants for a capacity-constrained manual review
queue**, while keeping every decision explainable enough to defend to a
credit officer, an applicant, and a regulator.

- **Scorecard** — WOE binning + logistic regression, 30 features, carries the
  *explanation* (points a reviewer can add up by hand).
- **Challenger** — gradient boosting (XGBoost), 218 features, carries the
  *ranking* (validation AUC 0.7848 vs the scorecard's 0.7516, 95% CI
  non-overlapping).
- **Decision layer** — a deliberate separation from the modelling layer:
  both models are `Classifier` objects with a working `.predict()`, but the
  system never calls it. Every downstream decision (queue ordering, capacity
  triage, business-impact simulation) operates on `predict_proba()` output as
  a *rank*, never as a calibrated probability thresholded at 0.5. See
  [Why ranking, not classification](#why-ranking-not-classification) below.

At an unchanged approval volume and review capacity, ranking applicants this
way cuts the default rate in the approved book from **8.07% → 4.91%** and
raises review precision from **12% → 26%** (2.2×) in a held-out simulation.

---

## Contents

- [Results at a glance](#results-at-a-glance)
- [Architecture](#architecture)
- [Why ranking, not classification](#why-ranking-not-classification)
- [Repository layout](#repository-layout)
- [Getting started](#getting-started)
- [Running the pipeline](#running-the-pipeline)
- [Testing](#testing)
- [Key design decisions](#key-design-decisions)
- [Known limitations](#known-limitations)
- [License](#license)

---

## Results at a glance

| | Population |
|---|---:|
| Labelled applicants | 307,511 |
| Base default rate | 8.07% (11.4 : 1 imbalance) |
| Raw history data (7 tables, joined to applicant grain) | 58,538,856 rows |
| Engineered features | 218 numeric (38 with a written business rationale) |

| Model | Features | CV AUC | Validation AUC | 95% CI |
|---|---:|---|---|---|
| Scorecard (WOE + LogisticRegression) | 30 | 0.7500 ± 0.0035 | 0.7516 | 0.7444–0.7588 |
| XGBoost | 218 | 0.7815 ± 0.0036 | 0.7848 | 0.7785–0.7914 |

The two models disagree on ~9.4% of applicants. In the cell each model
flags *alone*, the observed default rate is 18.64% (XGBoost-only, 2.3× base
rate) versus 8.57% (scorecard-only, ≈ base rate) — the same-sized cells with
very different real risk are what justify giving each model a different job
rather than picking a single winner (`compare_model_agreement()` in
[`src/evaluation.py`](src/evaluation.py)).

| Business impact (held-out, volume held constant) | Before | After |
|---|---:|---:|
| Approved loans | 52,278 | 52,278 |
| Default rate in approved book | 8.07% | 4.91% |
| Review precision (top 15% capacity) | 12.0% | 26.0% |
| Recall at 15% review capacity | — | 48.3% |
| Lift vs. random selection at 15% capacity | 1.0× | 3.22× |

All rupiah-denominated figures in `results.json` / `impact_simulation.txt`
follow from explicit, swappable assumptions in `BusinessAssumptions`
(`src/evaluation.py`) — see [Known limitations](#known-limitations). The
capacity/lift numbers above do not depend on those assumptions at all.

---

## Architecture

```
7 competition CSVs (58.5M rows)
        │
        ├── aggregate_bureau() ───────────┐
        ├── aggregate_bureau_balance()    │  groupby → one row per
        ├── aggregate_previous_app()      │  SK_ID_CURR; TARGET is never
        ├── aggregate_installments()      │  referenced during aggregation
        ├── aggregate_credit_card()       │
        └── aggregate_pos_cash()         ─┘
        │
        ▼  merge + assert grain (is_unique, len == n_rows)
application_train (307,511 × 122)
        │
        ▼  fix_days_employed() + build_business_features()
feature matrix (307,511 × 218)
        │
        ▼  train_test_split(stratify, random_state=42)
train 246,008  /  valid 61,503
        │
   ┌────┴──────────────────────────────────┐
   │  MODELLING LAYER                       │
   │  both are sklearn classifiers with a   │
   │  working .predict() — never called     │
   ├─────────────────────────────────────────┤
   │  WOE + LogisticRegression   XGBClassifier│
   │  (30 features)              (218 features)│
   └────┬──────────────────────────┬─────────┘
        │  predict_proba()         │  predict_proba()
        ▼                          ▼
   ┌─────────────────────────────────────────┐
   │  DECISION LAYER (src/evaluation.py)      │
   │  operates on rank, not on a fixed         │
   │  threshold                                │
   │  capacity_analysis() · compare_model_     │
   │  agreement() · business_impact_simulation()│
   │  · scorecard_points() (readability only)  │
   └────────────────────────────────────────────┘
```

Five modules under `src/`, each with one responsibility:

| Module | Responsibility |
|---|---|
| [`src/data.py`](src/data.py) | I/O with a memory ceiling, one-to-many aggregation, grain-safety asserts |
| [`src/features.py`](src/features.py) | 38 business-rationale features, population segments, model-input prep |
| [`src/woe.py`](src/woe.py) | Weight-of-Evidence binning, Information Value selection, points scorecard |
| [`src/models.py`](src/models.py) | Model builders, repeated-CV evaluation, SHAP explainability |
| [`src/evaluation.py`](src/evaluation.py) | Stability metrics, champion/challenger comparison, business impact simulation — the decision layer |

[`credit_risk_model.py`](credit_risk_model.py) is the CLI entry point and
orchestrator; it re-exports the `src/` API explicitly (not via wildcard
import) so `import credit_risk_model as crm` keeps working for the notebook
while every re-exported name stays traceable to its source module. The
notebook imports this same module, so every number in
[`Home_Credit_Credit_Risk_Model.ipynb`](Home_Credit_Credit_Risk_Model.ipynb)
(65 cells) comes from the identical code path the CLI runs.

## Why ranking, not classification

`XGBClassifier` and `LogisticRegression` are, by name and by API, classifiers
— both expose a working `.predict()`. The decision layer deliberately never
calls it. Three reasons:

1. **The probabilities aren't calibrated.** Both models use
   `class_weight="balanced"` / `scale_pos_weight=11.39` to correct for the
   11.4:1 class imbalance during training. That shifts `predict_proba()`
   output upward — a threshold of 0.5 no longer means "50% chance of
   default." Rank is invariant to any monotonic transform; a fixed threshold
   is not.
2. **The threshold is a capacity decision, not a model property.** How many
   applicants get flagged should come from how many a review team can
   actually process, not from a constant baked into the model.
3. **The decision isn't binary.** The real workflow is
   approve / refer / decline plus a disagreement-routing policy from
   `compare_model_agreement()` — a single 0/1 label can't represent that.

The scorecard's points conversion (`scorecard_points()`) is a pure monotonic
transform of the same probability — it adds zero new ranking information,
only readability. Evidence it isn't for ranking: in the scorecard-only
disagreement cell, the observed default rate is 8.57%, essentially equal to
the 8.07% population base rate.

## Repository layout

```
.
├── credit_risk_model.py           # CLI entry point / orchestrator
├── src/
│   ├── data.py                    # I/O, aggregation, PipelineConfig
│   ├── features.py                # business features, segments
│   ├── woe.py                     # WOE, IV, scorecard
│   ├── models.py                  # model builders, evaluation, SHAP
│   └── evaluation.py               # stability + decision layer
├── tests/                          # pytest suite (58% coverage of src/)
├── Home_Credit_Credit_Risk_Model.ipynb   # 65-cell walkthrough with real output
├── Home_Credit_Credit_Risk_Presentation.pptx
├── requirements.txt / requirements-dev.txt
├── pytest.ini / pyproject.toml    # test + lint config
└── .github/workflows/tests.yml    # CI: ruff + pytest on every push
```

## Getting started

```bash
git clone <this-repo>
cd BusinessCase_Khumaeni
python -m venv .venv && source .venv/bin/activate   # or your preferred env manager
pip install -r requirements.txt
```

**Competition data** is not included in this repository (Kaggle's
competition rules don't permit redistribution). Download it from
[the competition page](https://www.kaggle.com/competitions/home-credit-default-risk/data),
accept the rules, and unzip the CSVs into `home-credit-default-risk/` at the
repo root — that path is already gitignored.

**Optional GenAI reason codes** (`--genai-reasons` flag) call the OpenAI
API to reword adverse-action explanations. Every other code path works
without this. To enable it, create a `.env` file at the repo root with:

```
OPENAI_API_KEY=your-key-here
OPENAI_MODEL=gpt-4o
```

The key is read once at startup, never logged, and never written to any
artifact (`load_environment()` in `credit_risk_model.py`).

## Running the pipeline

```bash
# Fast iteration on a 30k-applicant sample
python credit_risk_model.py --sample 30000

# Full run
python credit_risk_model.py --output-dir artifacts

# Reuse a cached feature matrix instead of re-aggregating from CSV
python credit_risk_model.py --use-cache

# All flags
python credit_risk_model.py --help
```

Each run writes reproducible artifacts to `--output-dir` (default
`artifacts/`): `results.json`, `scorecard_points.csv`,
`boosting_importance.csv`, `shap_importance.csv`,
`segment_performance_{scorecard,boosting}.csv`, `threshold_economics.csv`,
`process_before_after.csv`, and `impact_simulation.txt`.

Or open [`Home_Credit_Credit_Risk_Model.ipynb`](Home_Credit_Credit_Risk_Model.ipynb)
for the narrated walkthrough — it imports this same module
(`import credit_risk_model as crm`), so nothing in it can drift from what
the CLI actually does.

## Testing

```bash
pip install -r requirements-dev.txt
pytest                                    # 53 tests, no Kaggle data required
pytest --cov=src --cov-report=term-missing   # coverage report
ruff check src/ tests/ credit_risk_model.py  # lint
```

Tests exercise the pure-logic core with synthetic data — WOE binning
direction and monotonicity, the scorecard's points formula against a
hand-computed expectation, PSI on known-shifted distributions, and the
business-impact simulation's central invariant that approval and review
*volume* stay constant before vs. after. They don't require the Kaggle CSVs.
CI (`.github/workflows/tests.yml`) runs the same lint + test suite on every
push, on Python 3.10 and 3.11.

## Key design decisions

- **No imputation anywhere.** 41 of 122 raw columns are missing >50% of the
  time, and it's structured, not random (e.g. AUC degrades from 0.793 to
  0.763 as the number of available external bureau scores drops from 3 to
  1) — see `EXT_SOURCE_*` handling in `src/features.py`. WOE gives missing
  its own empirically-weighted bin; the boosting trees learn a native split
  direction for `NaN`.
- **A sentinel value is fixed, not dropped.** `DAYS_EMPLOYED == 365243`
  (≈1,001 years, held by 18% of applicants, 99.96% of them pensioners) is
  replaced with `NaN` plus a `DAYS_EMPLOYED_ANOM` flag — dropping it would
  discard real signal (sentinel rows default at 5.40% vs. 8.66% for
  everyone else).
- **Monotonic WOE binning is enforced by construction**, even though it
  costs a little AUC — "higher utilisation is never scored as lower risk"
  has to hold for a scorecard a human is going to defend.
- **High Information Value triggers a provenance review, not automatic
  exclusion.** An early version auto-dropped anything with IV > 0.60 as
  suspected leakage; that silently removed `EXT_SOURCE_MEAN` (IV 0.604), the
  strongest legitimate predictor. Leakage is a question of *when the data
  was known*, not of how strong it is.
- **Gradient boosting capacity was deliberately cut**, not grown. An
  unconstrained first pass overfit badly (train AUC 0.993 / valid 0.725,
  gap +0.27); cutting `max_depth`, raising `min_child_weight`, and adding
  `gamma`/`reg_lambda` closed the gap to +0.0295 while *raising* validation
  AUC to 0.7848 — evidence the problem was genuinely overfitting.
- **`class_weight="balanced"` over SMOTE/oversampling.** Oversampling
  duplicates rows, which can land on both sides of a CV fold boundary
  (leakage); SMOTE interpolates in a feature space where many columns are
  >50% missing, producing applicants that can't exist. Class weighting
  achieves the same rebalancing through the loss function without touching
  the data — at the cost of calibration, which is why the output is used
  as a rank (see [above](#why-ranking-not-classification)), not a
  probability.
- **Gender is excluded, and the cost is measured, not assumed.**
  `CODE_GENDER` correlates with the target here (10.17% vs. 6.99% default
  rate) — exactly why it's excluded rather than used. The performance cost
  of excluding it is under 0.005 AUC (`--keep-gender` exists solely to
  measure that cost).
- **Adverse-action explanations come from scorecard points, not SHAP.** A
  SHAP value is a second model on top of the first — a reviewer can't
  re-verify it without running code. Scorecard points are the exact
  integers published in `scorecard_points.csv`; a reviewer can add them up
  by hand and reach the same total.

## Known limitations

Stated here because a portfolio piece that hides its own caveats is less
useful than one that names them:

1. **Validation is a random split, not out-of-time.** It's a random 80/20
   split of one historical snapshot. Before trusting this on a new
   population, it needs validation against a later application vintage —
   this is the single limitation that most affects every other claim.
2. **Output is uncalibrated**, by design (see
   [Why ranking, not classification](#why-ranking-not-classification)). If
   it ever needs to be read as a probability of default — for pricing,
   provisioning, or IFRS 9 — it needs isotonic or Platt calibration first.
3. **No reject inference.** `TARGET` only reflects *approved* applicants;
   declined applicants are structurally invisible to this dataset.
4. **This is the public competition dataset**, not a live production
   population.
5. **No formal fairness audit** beyond excluding gender and measuring its
   cost — proxy discrimination through correlated features (e.g. income
   type, number of dependents) hasn't been tested.
6. **Categorical features aren't used.** `prepare_model_inputs()` filters to
   `select_dtypes(include=[np.number])`, so columns like
   `NAME_EDUCATION_TYPE` never reach either model. That's a scope decision
   (the `WOETransformer` here only handles numeric features), not a finding
   that they're uninformative — the cheapest next improvement.
7. **The business-impact rupiah figures are a scenario**, not measured
   results — every economic assumption lives in one `BusinessAssumptions`
   dataclass, with sourced assumptions distinguished from unsourced ones in
   its docstring.

## License

The code in this repository is [MIT licensed](LICENSE). The Home Credit
Default Risk dataset is **not** included and is governed by its own Kaggle
competition rules — download and use it under those terms, not this
license.
