"""Train a cross-validated tabular baseline and write a Kaggle submission."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def build_pipeline(features: pd.DataFrame, task: str, seed: int) -> Pipeline:
    """Build a preprocessing and tree-model pipeline fitted within each CV fold."""
    numeric_columns = features.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = [
        column for column in features.columns if column not in numeric_columns
    ]

    numeric = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessing = ColumnTransformer(
        [
            ("numeric", numeric, numeric_columns),
            ("categorical", categorical, categorical_columns),
        ],
        remainder="drop",
    )

    if task == "classification":
        model = ExtraTreesClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
    else:
        model = ExtraTreesRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=-1,
        )

    return Pipeline([("preprocessing", preprocessing), ("model", model)])


def train_and_submit(
    train_path: Path,
    test_path: Path,
    target: str,
    task: str,
    sample_submission_path: Path | None,
    output_path: Path,
    model_path: Path,
    id_column: str | None = None,
    n_splits: int = 5,
    seed: int = 42,
    scoring: str | None = None,
) -> float:
    """Cross-validate, fit on all labeled rows, and write a submission CSV."""
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    if target not in train.columns:
        raise ValueError(f"Target column {target!r} is missing from {train_path}.")
    if train.empty or test.empty:
        raise ValueError("Training and test CSV files must both contain at least one row.")
    if train[target].isna().any():
        raise ValueError(f"Target column {target!r} contains missing values.")
    if n_splits < 2:
        raise ValueError("--n-splits must be at least 2.")

    excluded = {target}
    if id_column:
        if id_column not in train.columns or id_column not in test.columns:
            raise ValueError(f"ID column {id_column!r} must exist in both CSV files.")
        excluded.add(id_column)

    feature_columns = [column for column in train.columns if column not in excluded]
    if not feature_columns:
        raise ValueError("No feature columns remain after excluding the target and ID.")
    missing_columns = [column for column in feature_columns if column not in test.columns]
    if missing_columns:
        raise ValueError(f"Test data is missing training feature columns: {missing_columns}")

    features = train[feature_columns]
    labels = train[target]
    test_features = test[feature_columns]
    pipeline = build_pipeline(features, task, seed)

    if scoring is None:
        scoring = "accuracy" if task == "classification" else "neg_root_mean_squared_error"
    if task == "classification":
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    scores = cross_val_score(
        pipeline,
        features,
        labels,
        cv=splitter,
        scoring=scoring,
        n_jobs=1,
        error_score="raise",
    )
    pipeline.fit(features, labels)
    predictions = pipeline.predict(test_features)

    if sample_submission_path is not None:
        submission = pd.read_csv(sample_submission_path)
        if len(submission) != len(test):
            raise ValueError("Sample submission and test data must have the same number of rows.")
        if target not in submission.columns:
            raise ValueError(
                f"Target column {target!r} is missing from {sample_submission_path}."
            )
        if id_column and id_column in submission.columns and not np.array_equal(
            submission[id_column].to_numpy(), test[id_column].to_numpy()
        ):
            raise ValueError(
                f"Sample submission IDs in {id_column!r} do not match test data order."
            )
        submission[target] = predictions
    else:
        submission = pd.DataFrame({target: predictions})
        if id_column:
            submission.insert(0, id_column, test[id_column].to_numpy())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    joblib.dump(pipeline, model_path)

    mean_score = float(np.mean(scores))
    reported_score = -mean_score if scoring.startswith("neg_") else mean_score
    metric_name = scoring.removeprefix("neg_")
    print(f"CV {metric_name}: {reported_score:.6f} (mean of {n_splits} folds)")
    print(f"Submission: {output_path}")
    print(f"Model: {model_path}")
    return mean_score


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a tabular baseline and create a Kaggle submission."
    )
    parser.add_argument("--train", type=Path, default=Path("data/raw/train.csv"))
    parser.add_argument("--test", type=Path, default=Path("data/raw/test.csv"))
    parser.add_argument("--target", required=True, help="Name of the label column.")
    parser.add_argument(
        "--task",
        choices=("classification", "regression"),
        required=True,
        help="Choose the task type; competition metrics may require a different scorer.",
    )
    parser.add_argument("--sample-submission", type=Path)
    parser.add_argument("--output", type=Path, default=Path("submissions/submission.csv"))
    parser.add_argument("--model", type=Path, default=Path("models/baseline.joblib"))
    parser.add_argument("--id-column", help="Optional ID column to exclude from model features.")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--scoring",
        help="Optional scikit-learn scorer name (default: accuracy or negative RMSE).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    train_and_submit(
        train_path=args.train,
        test_path=args.test,
        target=args.target,
        task=args.task,
        sample_submission_path=args.sample_submission,
        output_path=args.output,
        model_path=args.model,
        id_column=args.id_column,
        n_splits=args.n_splits,
        seed=args.seed,
        scoring=args.scoring,
    )


if __name__ == "__main__":
    main()
