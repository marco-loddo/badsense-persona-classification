# bAdsense Persona Classification

ML classification pipeline to predict smoker/drinker status from medical data, combined with an expected-value decision framework to optimise advertiser portfolio assignments. Built as coursework at Stockholm School of Economics (Data Science, Q4 2025).

## Problem

The dataset comes from a simulated production system: a MySQL database with mixed training and classification records, messy label encoding, and three separate data sources (demographics, blood tests, physiotherapy). The classification set has at least one target variable obscured — the goal is to predict the missing values and then make profit-maximising decisions for three advertisers under that uncertainty.

Two non-trivial aspects distinguish this from a standard ML homework:

1. **Label noise**: smoker/drinker fields contain freeform strings (`"cigarettes are my only friend"`, `"uuhh... i only drink on weekends"`) requiring canonical token mapping before any modelling can happen.
2. **Decision layer under uncertainty**: once predictions are made, each user must be included or excluded from three advertiser portfolios. The optimal decision is not based on the hard prediction but on the expected value computed from the model's output probabilities — a user with P(smoker)=0.6 and P(drinker)=0.4 has a calculable expected payoff for each advertiser.

## Solution

1. **Data ingestion**: monthly-chunked SQL downloads from three tables (`Main`, `BloodTests`, `Physiotherapy`) with retry logic; access each table at most once
2. **Feature engineering**: birth year normalisation (2-digit → 4-digit), blood test pivot (latest value per device per test), waistline/BMI contextual imputation via gender × age_bin group medians
3. **Training split**: deterministic separation based on label completeness; known values in the classification set are preserved and never re-predicted
4. **Modelling**: `HistGradientBoostingClassifier` with `RandomizedSearchCV` (3-fold); separate pipelines for smoker and drinker tasks with task-specific feature sets
5. **Threshold optimisation**: OOF probabilities used to find the classification threshold that maximises accuracy beyond the default 0.5
6. **EV-based portfolio decisions**: for each advertiser, include a user if and only if `E[value] = Σ P(persona) × price(persona) > 0`, where persona probabilities are derived under a smoker/drinker independence assumption

## Results

Evaluated against the complete hidden dataset (10,000 classifications) by course staff at SSE.

| Task | Test Accuracy | Course Rank |
|---|---|---|
| Drinker prediction | 84% | **3 / 45 students** |
| Smoker prediction | 88% | 11 / 45 students |

| Advertiser | Actual Profit | Estimated Profit | Error |
|---|---|---|---|
| Johnny Talker | $51,654 | $47,913 | −$3,741 |
| GigGolo | $46,118 | $43,533 | −$2,585 |
| VapeShape | $30,828 | $31,309 | +$481 |
| **Total** | **$128,600** | **$122,755** | **−$5,845** |

Total profit ranked **2nd out of 45 students** ($1,221 behind the top submission). Profit estimates were within 4.5% of actual outcomes, reflecting well-calibrated model probabilities.

## Tech Stack

`Python` `scikit-learn` `pandas` `NumPy` `SQLAlchemy` `PyMySQL`

## Project Structure

```
badsense-persona-classification/
├── src/
│   └── badsense_classification.ipynb
├── output_sample.json
├── README.md
├── requirements.txt
├── .env.example
└── .gitignore
```

## Setup

```bash
git clone https://github.com/marco-loddo/badsense-persona-classification
cd badsense-persona-classification
pip install -r requirements.txt
cp .env.example .env   # then fill in your DB credentials
```

## Usage

Open and run `src/badsense_classification.ipynb` in sequence:

1. **SETUP — Data preparation**: connects to DB, downloads and merges all tables
2. **SETUP — Split & Skeleton**: separates training from classification set, creates JSON skeleton
3. **SETUP — Cleaning & Imputation**: feature engineering and contextual imputation
4. **Q1 — Modelling**: trains smoker/drinker classifiers, optimises thresholds, generates predictions
5. **Q2 — Personas & Advertiser Decisions**: assigns personas, computes EV, outputs portfolio flags

The final JSON output matches the structure below:

```json
[
  {
    "device_id": "0002bc40...",
    "user_drinks": 1,
    "user_smokes": 1,
    "johnny_talker": 1,
    "giggolo": 0,
    "vapeshape": 1,
    "persona": "sinner"
  }
]
```

## Notes

- This is academic coursework from SSE's Data Science course (Q4 2025). The database was hosted on SSE's internal MySQL server and is no longer publicly accessible.
- The `output_sample.json` file contains 5 anonymised records for illustration only; the full classification set (10,000 records) is not included.
- Device IDs are anonymised hashes; no personal data is present in this repository.
