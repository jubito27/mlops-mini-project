"""Tests for the model produced by this repository's DVC pipeline.

These tests intentionally use the local model and transformed holdout data
created by the current pipeline. They do not depend on an unrelated
DagsHub repository or a vectorizer artifact that this project does not build.
"""

import pickle
import unittest
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


ROOT = Path(__file__).resolve().parents[1]


class TestModelLoading(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        model_path = ROOT / "model" / "model.pkl"
        x_test_path = ROOT / "data" / "transformed" / "x_test.csv"
        y_test_path = ROOT / "data" / "transformed" / "y_test.csv"

        for path in (model_path, x_test_path, y_test_path):
            if not path.is_file():
                raise FileNotFoundError(
                    f"Required DVC pipeline artifact is missing: {path}"
                )

        with model_path.open("rb") as model_file:
            cls.model = pickle.load(model_file)

        cls.x_test = pd.read_csv(x_test_path)
        y_test_frame = pd.read_csv(y_test_path)

        if y_test_frame.shape[1] != 1:
            raise ValueError(
                "Expected y_test.csv to contain exactly one target column; "
                f"found {y_test_frame.shape[1]} columns."
            )

        cls.y_test = y_test_frame.iloc[:, 0]

        if cls.x_test.empty or cls.y_test.empty:
            raise ValueError("The transformed holdout test data is empty.")

        if len(cls.x_test) != len(cls.y_test):
            raise ValueError(
                "Feature/target row counts differ: "
                f"{len(cls.x_test)} != {len(cls.y_test)}"
            )

    def test_model_loaded_properly(self):
        self.assertIsNotNone(self.model)
        self.assertTrue(callable(getattr(self.model, "predict", None)))

    def test_model_input_signature(self):
        expected_features = getattr(self.model, "n_features_in_", None)
        self.assertIsNotNone(
            expected_features,
            "The trained model does not expose n_features_in_.",
        )
        self.assertEqual(
            self.x_test.shape[1],
            expected_features,
            "Holdout feature count does not match the trained model.",
        )

        predictions = self.model.predict(self.x_test)
        self.assertEqual(len(predictions), len(self.x_test))

    def test_model_performance(self):
        predictions = self.model.predict(self.x_test)

        accuracy = accuracy_score(self.y_test, predictions)
        precision = precision_score(
            self.y_test, predictions, average="macro", zero_division=0
        )
        recall = recall_score(
            self.y_test, predictions, average="macro", zero_division=0
        )
        f1 = f1_score(
            self.y_test, predictions, average="macro", zero_division=0
        )

        minimum_score = 0.40
        self.assertGreaterEqual(accuracy, minimum_score)
        self.assertGreaterEqual(precision, minimum_score)
        self.assertGreaterEqual(recall, minimum_score)
        self.assertGreaterEqual(f1, minimum_score)


if __name__ == "__main__":
    unittest.main()
