import pandas as pd
import pytest

from kaggle_template.train import train_and_submit


def test_training_writes_aligned_submission_and_model(tmp_path):
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    sample_path = tmp_path / "sample_submission.csv"
    output_path = tmp_path / "submissions" / "submission.csv"
    model_path = tmp_path / "models" / "model.joblib"

    pd.DataFrame(
        {
            "id": range(12),
            "number": range(12),
            "category": ["a", "b"] * 6,
            "target": [0, 1] * 6,
        }
    ).to_csv(train_path, index=False)
    pd.DataFrame(
        {"id": [20, 21], "number": [13, 14], "category": ["new", "a"]}
    ).to_csv(test_path, index=False)
    pd.DataFrame({"id": [20, 21], "target": [0, 0]}).to_csv(sample_path, index=False)

    train_and_submit(
        train_path=train_path,
        test_path=test_path,
        target="target",
        task="classification",
        sample_submission_path=sample_path,
        output_path=output_path,
        model_path=model_path,
        id_column="id",
        n_splits=2,
        seed=7,
    )

    submission = pd.read_csv(output_path)
    assert submission.columns.tolist() == ["id", "target"]
    assert submission["id"].tolist() == [20, 21]
    assert submission["target"].isin([0, 1]).all()
    assert model_path.is_file()


def test_training_reports_missing_test_features(tmp_path):
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    pd.DataFrame({"feature": [1, 2, 3, 4], "target": [1, 2, 3, 4]}).to_csv(
        train_path, index=False
    )
    pd.DataFrame({"other": [5]}).to_csv(test_path, index=False)

    with pytest.raises(ValueError, match="missing training feature columns"):
        train_and_submit(
            train_path=train_path,
            test_path=test_path,
            target="target",
            task="regression",
            sample_submission_path=None,
            output_path=tmp_path / "submission.csv",
            model_path=tmp_path / "model.joblib",
        )


def test_regression_without_sample_submission_includes_test_ids(tmp_path):
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    output_path = tmp_path / "submission.csv"
    pd.DataFrame(
        {"id": range(10), "feature": range(10), "target": [value * 2 for value in range(10)]}
    ).to_csv(train_path, index=False)
    pd.DataFrame({"id": [10, 11], "feature": [10, 11]}).to_csv(test_path, index=False)

    train_and_submit(
        train_path=train_path,
        test_path=test_path,
        target="target",
        task="regression",
        sample_submission_path=None,
        output_path=output_path,
        model_path=tmp_path / "model.joblib",
        id_column="id",
        n_splits=2,
    )

    submission = pd.read_csv(output_path)
    assert submission.columns.tolist() == ["id", "target"]
    assert submission["id"].tolist() == [10, 11]
    assert submission["target"].notna().all()


def test_training_rejects_misaligned_sample_submission_ids(tmp_path):
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    sample_path = tmp_path / "sample_submission.csv"
    pd.DataFrame({"id": range(8), "feature": range(8), "target": [0, 1] * 4}).to_csv(
        train_path, index=False
    )
    pd.DataFrame({"id": [8, 9], "feature": [8, 9]}).to_csv(test_path, index=False)
    pd.DataFrame({"id": [9, 8], "target": [0, 0]}).to_csv(sample_path, index=False)

    with pytest.raises(ValueError, match="IDs.*do not match test data order"):
        train_and_submit(
            train_path=train_path,
            test_path=test_path,
            target="target",
            task="classification",
            sample_submission_path=sample_path,
            output_path=tmp_path / "submission.csv",
            model_path=tmp_path / "model.joblib",
            id_column="id",
            n_splits=2,
        )
