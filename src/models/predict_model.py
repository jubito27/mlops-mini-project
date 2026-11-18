import json
import logging
import os
import pickle
import sys
from pathlib import Path
from typing import Tuple, Union

import dagshub
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from dvclive import Live
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.multiclass import OneVsRestClassifier




# Set up DagsHub credentials for MLflow tracking
dagshub_token = os.getenv("DAGSHUB_PAT")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "jubito27"
repo_name = "mlops-mini-project"

# Set up MLflow tracking URI
mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')



# ---------------- Logging setup ----------------

def setup_logging(level: int = logging.DEBUG) -> logging.Logger:
    """Configure console logging at the requested level."""
    logger = logging.getLogger("model_evaluation")
    logger.setLevel(level)

    # Prevent duplicate handlers when the module is imported or re-run
    if logger.handlers:
        return logger

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


logger = setup_logging(logging.DEBUG)


# ---------------- Model and data loading ----------------

def load_model(model_path: str) -> object:
    """Load a pickled trained model."""
    logger.debug("Starting model loading")
    logger.debug("Model path: %s", model_path)

    if not os.path.isfile(model_path):
        logger.error("Model file does not exist: %s", model_path)
        raise FileNotFoundError(f"Model file does not exist: {model_path}")

    try:
        with open(model_path, "rb") as model_file:
            model = pickle.load(model_file)

        logger.debug("Loaded model: %s", model)
        logger.info("Model loaded successfully")

        return model

    except Exception:
        logger.exception("Failed to load model from: %s", model_path)
        raise


def load_data(
    x_test_path: str,
    y_test_path: str,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Load transformed test features and labels."""
    logger.debug("Starting test data loading")
    logger.debug("Test features path: %s", x_test_path)
    logger.debug("Test labels path: %s", y_test_path)

    for path in [x_test_path, y_test_path]:
        if not os.path.isfile(path):
            logger.error("Input CSV file does not exist: %s", path)
            raise FileNotFoundError(f"Input CSV file does not exist: {path}")

    try:
        x_test = pd.read_csv(x_test_path)
        y_test = pd.read_csv(y_test_path)

        logger.debug("Test features shape: %s", x_test.shape)
        logger.debug("Test labels shape: %s", y_test.shape)

        if x_test.empty:
            logger.error("Test feature dataframe is empty: %s", x_test_path)
            raise ValueError("Test feature dataframe is empty")

        if y_test.empty:
            logger.error("Test label dataframe is empty: %s", y_test_path)
            raise ValueError("Test label dataframe is empty")

        if x_test.shape[0] != y_test.shape[0]:
            logger.error(
                "Feature and label row counts do not match. Features: %s, Labels: %s",
                x_test.shape[0],
                y_test.shape[0],
            )
            raise ValueError(
                f"Feature and label row counts do not match: "
                f"{x_test.shape[0]} features vs {y_test.shape[0]} labels"
            )

        if y_test.shape[1] != 1:
            logger.error(
                "Label file must contain exactly one column. Found: %s",
                y_test.shape[1],
            )
            raise ValueError("Label file must contain exactly one column")

        y_test = y_test.iloc[:, 0]
        y_test.name = "sentiment"

        logger.info("Test data loaded successfully")
        return x_test, y_test

    except Exception:
        logger.exception("Failed to load test data")
        raise


# ---------------- Evaluation ----------------

def calculate_scores(
    model: object,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[pd.DataFrame, str]:
    """Calculate classification metrics and generate a classification report."""
    logger.debug("Starting model evaluation")

    if model is None:
        logger.error("Model object is None; cannot evaluate the model")
        raise ValueError("Model object is None; cannot evaluate the model")

    try:
        logger.debug("Generating predictions")
        y_pred = model.predict(x_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0,
        )
        recall = recall_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0,
        )
        f1 = f1_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0,
        )

        logger.debug(
            "Metrics - accuracy: %s, precision: %s, recall: %s, f1: %s",
            accuracy,
            precision,
            recall,
            f1,
        )

        logger.debug("Starting DVCLive metric tracking")

        with Live(save_dvc_exp=True) as live:
            live.log_metric("accuracy", accuracy)
            live.log_metric("precision", precision)
            live.log_metric("recall", recall)
            live.log_metric("f1_score", f1)

        logger.debug("DVCLive metric tracking completed")

        classification = classification_report(
            y_test,
            y_pred,
            zero_division=0,
        )

        y_proba = model.predict_proba(x_test)
        model_classes = np.asarray(model.classes_)
        test_classes = pd.unique(y_test)

        logger.debug("Model classes: %s", list(model_classes))
        logger.debug("Test classes: %s", list(test_classes))

        if len(test_classes) < 2:
            logger.error("ROC AUC requires at least two classes in y_test")
            raise ValueError("ROC AUC requires at least two classes in y_test")

        if not np.isin(test_classes, model_classes).all():
            logger.error(
                "y_test contains a class that the model was not trained to predict"
            )
            raise ValueError(
                "y_test contains a class that the model was not trained to predict"
            )

        class_aucs = []

        for class_label in test_classes:
            class_index = np.flatnonzero(model_classes == class_label)[0]
            y_binary = (y_test == class_label).astype(int)

            class_auc = roc_auc_score(
                y_binary,
                y_proba[:, class_index],
            )

            class_aucs.append(class_auc)

            logger.debug(
                "ROC AUC for class %s: %s",
                class_label,
                class_auc,
            )

        auc = float(np.mean(class_aucs))
        logger.debug("Mean one-vs-rest ROC AUC: %s", auc)

        logger.debug("Logging ROC AUC metric to DVCLive")

        with Live(save_dvc_exp=True) as live:
            live.log_metric("roc_auc", auc)

        scores = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": auc,
        }

        scores_df = pd.DataFrame(
            [scores],
            index=["logistic_regression"],
        ).T

        logger.info("Model evaluation completed successfully")
        return scores_df, classification

    except Exception:
        logger.exception("Failed during model evaluation")
        raise


# ---------------- Report saving ----------------

def save_reports(
    scores_df: pd.DataFrame,
    classification_report_text: str,
    scores_path: str,
    classification_path: str,
    output_dir: str,
) -> None:
    """Save evaluation metrics and classification report."""
    logger.debug("Starting report saving process")
    logger.debug("Report output directory: %s", output_dir)
    logger.debug("Scores output path: %s", scores_path)
    logger.debug("Classification report output path: %s", classification_path)

    try:
        os.makedirs(output_dir, exist_ok=True)
        logger.debug("Ensured report output directory exists: %s", output_dir)

        scores_df.to_csv(scores_path, index=True)
        logger.info("Evaluation scores saved successfully: %s", scores_path)

        Path(classification_path).write_text(
            classification_report_text,
            encoding="utf-8",
        )
        logger.info(
            "Classification report saved successfully: %s",
            classification_path,
        )

    except Exception:
        logger.exception("Failed to save evaluation reports")
        raise


# ---------------- MLflow helpers ----------------

def log_model_to_mlflow(
    model: object,
    x_test: pd.DataFrame,
    model_name: str,
) -> str:
    """Log the trained model to MLflow and return its model URI."""
    logger.debug("Starting MLflow model logging")
    logger.debug("Model name: %s", model_name)

    try:
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name=model_name,
            input_example=x_test[:5],
        )

        model_uri = model_info.model_uri

        logger.debug("Logged model URI: %s", model_uri)
        logger.info("Model logged successfully to MLflow: %s", model_name)

        return model_uri

    except Exception:
        logger.exception("Failed to log model to MLflow")
        raise


def save_model_info(
    run_id: str,
    model_name: str,
    model_uri: str,
    file_path: str,
) -> None:
    """Save MLflow run ID, model name, and model URI to a JSON file."""
    logger.debug("Starting model-info saving process")
    logger.debug("MLflow run ID: %s", run_id)
    logger.debug("Model name: %s", model_name)
    logger.debug("Model URI: %s", model_uri)
    logger.debug("Model-info output path: %s", file_path)

    try:
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)

        model_info = {
            "run_id": run_id,
            "model_name": model_name,
            "model_uri": model_uri,
        }

        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(model_info, file, indent=4)

        logger.info("Model info saved successfully: %s", file_path)

    except Exception:
        logger.exception("Failed to save model info to: %s", file_path)
        raise

# ---------------- Pipeline ----------------

def main(
    model_path: str,
    x_test_path: str,
    y_test_path: str,
    scores_path: str,
    classification_path: str,
    output_dir: str,
    experiment_info_path: str,
) -> None:
    """Run the complete model-evaluation pipeline."""
    logger.info("Starting model-evaluation pipeline")

    try:
        logger.debug("Initializing DagsHub and MLflow")

        # dagshub.init(
        #     repo_owner="jubito27",
        #     repo_name="mlops-mini-project",
        #     mlflow=True,
        # )

        # mlflow.set_tracking_uri(
        #     "https://dagshub.com/jubito27/mlops-mini-project.mlflow"
        # )

        logger.debug("DagsHub initialized successfully")

        mlflow.set_experiment("dvc-pipeline")

        with mlflow.start_run() as run:
            model = load_model(model_path=model_path)

            x_test, y_test = load_data(
                x_test_path=x_test_path,
                y_test_path=y_test_path,
            )

            scores_df, classification_report_text = calculate_scores(
                model=model,
                x_test=x_test,
                y_test=y_test,
            )

            save_reports(
                scores_df=scores_df,
                classification_report_text=classification_report_text,
                scores_path=scores_path,
                classification_path=classification_path,
                output_dir=output_dir,
            )

            logger.debug("Logging evaluation metrics to MLflow")

            for metric_name in scores_df.index:
                metric_value = float(scores_df.loc[metric_name].iloc[0])
                rounded_value = round(metric_value, 5)

                mlflow.log_metric(metric_name, rounded_value)

                logger.debug(
                    "Logged MLflow metric: %s = %s",
                    metric_name,
                    rounded_value,
                )

            logger.info("Evaluation metrics logged successfully to MLflow")

            logger.debug("Logging model parameters to MLflow")

            if hasattr(model, "get_params"):
                params = model.get_params()

                for param_name, param_value in params.items():
                    mlflow.log_param(param_name, param_value)

                    logger.debug(
                        "Logged MLflow parameter: %s = %s",
                        param_name,
                        param_value,
                    )

            logger.info("Model parameters logged successfully to MLflow")

            model_name = "logistic_regression_model"

            model_uri = log_model_to_mlflow(
                model=model,
                x_test=x_test,
                model_name=model_name,
            )

            save_model_info(
                run_id=run.info.run_id,
                model_name=model_name,
                model_uri=model_uri,
                file_path=experiment_info_path,
            )
            logger.debug("Logging evaluation reports to MLflow")

            mlflow.log_artifact(scores_path, artifact_path="reports")
            mlflow.log_artifact(
                classification_path,
                artifact_path="reports",
            )

            logger.info("Evaluation reports logged successfully to MLflow")
            logger.info("Model-evaluation pipeline completed successfully")

    except Exception:
        logger.exception("Model-evaluation pipeline failed")
        raise


if __name__ == "__main__":
    curr_dir = Path(__file__)
    home_dir = curr_dir.parent.parent.parent

    transformed_dir = home_dir / "data" / "transformed"
    reports_dir = home_dir / "reports" / "figures"
    model_dir = home_dir / "model"

    model_path = model_dir / "model.pkl"

    x_test_path = transformed_dir / "x_test.csv"
    y_test_path = transformed_dir / "y_test.csv"

    scores_path = reports_dir / "scores.csv"
    classification_path = reports_dir / "classification.txt"
    experiment_info_path = reports_dir / "experiment_info.json"

    logger.debug("Project home directory: %s", home_dir)
    logger.debug("Model path: %s", model_path)
    logger.debug("Test features path: %s", x_test_path)
    logger.debug("Test labels path: %s", y_test_path)
    logger.debug("Report output directory: %s", reports_dir)
    logger.debug("Experiment info path: %s", experiment_info_path)

    try:
        main(
            model_path=str(model_path),
            x_test_path=str(x_test_path),
            y_test_path=str(y_test_path),
            scores_path=str(scores_path),
            classification_path=str(classification_path),
            output_dir=str(reports_dir),
            experiment_info_path=str(experiment_info_path),
        )

        print("Model evaluation completed successfully")

    except FileNotFoundError:
        logger.exception("Required file or directory was not found")
        sys.exit(1)

    except ValueError as e:
        logger.exception("Validation error during model evaluation: %s", e)
        sys.exit(1)

    except Exception:
        logger.exception("Unexpected error occurred during model evaluation")
        sys.exit(1)