# Kaggle Competition Template

A practical, reusable starter for **tabular classification and regression** competitions. It includes a cross-validated scikit-learn baseline, preprocessing that is fitted separately inside each fold, Kaggle submission generation, and a notebook-driven workflow. Adapt the validation strategy, metric, features, and model to the rules of each competition before trusting a leaderboard score.

## Quick start

1. Create a virtual environment and install the project:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   python -m pip install --upgrade pip
   pip install -e ".[notebooks,dev]"
   ```

   For a lightweight command-line-only setup, use `pip install -e .`. Optional gradient-boosting libraries are available with `pip install -e ".[boosting]"`.

2. Download the competition data from Kaggle and place `train.csv`, `test.csv`, and (if provided) `sample_submission.csv` in `data/raw/`. Competition data is intentionally excluded from Git.

3. Inspect the problem rules, choose a validation strategy and metric that match the competition, and run a baseline:

   ```bash
   kaggle-train \
     --train data/raw/train.csv \
     --test data/raw/test.csv \
     --sample-submission data/raw/sample_submission.csv \
     --target TARGET \
     --task classification \
     --id-column id
   ```

   Replace `TARGET` and `id` with the actual column names. For regression, use `--task regression`. The command saves `submissions/submission.csv` and `models/baseline.joblib` by default. Omit `--sample-submission` if the competition does not provide one. Run `kaggle-train --help` to see all options.

   The starter reports accuracy for classification and RMSE for regression by default. Set `--scoring` to a compatible scikit-learn scorer when the competition uses a different metric (for example, `roc_auc`, `f1_macro`, or `neg_mean_absolute_error`). Confirm that the scorer's input and prediction requirements match the competition; probability-based metrics may require adapting the inference code.

4. Open the numbered notebooks for an EDA and experimentation workflow:

   ```bash
   jupyter lab
   ```

## What the baseline does

- Keeps the target and optional ID out of the features.
- Uses median imputation for numeric columns and most-frequent imputation plus one-hot encoding for categorical columns. Unseen test categories are handled safely.
- Fits preprocessing as part of a scikit-learn pipeline, independently within each cross-validation fold.
- Uses shuffled stratified folds for classification and shuffled K-folds for regression.
- Refits the baseline on all labeled rows, preserves the sample submission's columns and order, and writes the predictions into its target column.
- Saves the fitted pipeline so it can be reused with `joblib`.

This is a starting point, not a universal competition solution. It assumes CSV data and a single prediction column. Image, text, time-series, grouped, ranking, multilabel, and probability-submission competitions need task-specific data loaders, validation, models, and/or output logic.

## Validation and competition checklist

Before comparing experiments or submitting:

- Read the competition description, evaluation metric, submission format, data-use rules, and deadlines.
- Match local validation to the data-generating process. Use group-aware folds for repeated entities, time-ordered splits for forecasting, and the prescribed holdout where appropriate; do not randomly split when that would leak information.
- Check for duplicates, target leakage, train/test distribution shifts, class imbalance, missing values, and ID semantics. Never use public leaderboard feedback as a substitute for local validation.
- Keep all learned preprocessing, feature selection, and target encoding inside the CV pipeline/fold. Do not fit them on validation rows.
- Set and record seeds; compare experiments on the same folds and metric. Tune only after establishing a reliable baseline.
- Verify the submission has exactly the required row count, column names/order, ID alignment, finite predictions, and valid value range before upload.
- Save experiment notes (data version, fold strategy, metric, features, model, parameters, CV result, public score, and observations) alongside the notebook or in your experiment tracker.

## Project layout

```text
data/
  raw/                 # Downloaded, immutable competition files (not tracked)
  processed/           # Optional generated datasets (not tracked)
models/                # Saved model artifacts (not tracked)
notebooks/             # Ordered EDA, baseline, feature, training, and submission notebooks
src/kaggle_template/   # Reusable training and preprocessing code
submissions/           # Generated Kaggle CSVs (not tracked)
tests/                 # Focused tests for the starter workflow
pyproject.toml          # Installable package, dependencies, and tool configuration
requirements.txt        # Convenience install: editable project with notebook and dev extras
```

## Development

```bash
pytest
ruff check .
```

Customize the `kaggle_template.train` pipeline or add competition-specific modules under `src/`. Keep exploratory work in notebooks, then move reusable, testable logic into `src/`. Check the competition rules before sharing data, notebooks, models, or predictions.

