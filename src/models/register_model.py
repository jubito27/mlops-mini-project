import json
import logging
import os

import dagshub
import mlflow


# ---------------- Logging setup ----------------

logger = logging.getLogger("model_registration")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler("model_registration_errors.log")
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


# ---------------- DagsHub setup ----------------

def setup_dagshub() -> None:
    """Initialize DagsHub and configure the MLflow tracking server."""
    try:
        logger.debug("Initializing DagsHub")

        dagshub.init(
            repo_owner="jubito27",
            repo_name="mlops-mini-project",
            mlflow=True,
        )

        mlflow.set_tracking_uri(
            "https://dagshub.com/jubito27/mlops-mini-project.mlflow"
        )

        logger.debug("DagsHub initialized successfully")

    except Exception:
        logger.exception("Failed to set up connection with DagsHub")
        raise


# ---------------- Model registration ----------------

def load_model_info(file_path: str) -> dict:
    """Load model information from a JSON file."""
    logger.debug("Loading model info from: %s", file_path)

    if not os.path.isfile(file_path):
        logger.error("Model info file does not exist: %s", file_path)
        raise FileNotFoundError(f"Model info file does not exist: {file_path}")

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            model_info = json.load(file)

        logger.debug("Model info loaded: %s", model_info)
        return model_info

    except json.JSONDecodeError as e:
        logger.exception("Invalid JSON in model info file: %s", file_path)
        raise ValueError(f"Invalid JSON in model info file: {file_path}") from e

    except Exception:
        logger.exception("Unexpected error while loading model info")
        raise


def register_model(model_name: str, model_info: dict) -> None:
    """Register an MLflow model and assign the Staging alias."""
    setup_dagshub()

    mlflow.set_experiment("dvc-pipeline")

    try:
        logger.debug("Model name: %s", model_name)
        logger.debug("Model info: %s", model_info)

        if "model_uri" not in model_info:
            logger.error(
                "model_uri is missing from experiment_info.json. "
                "Update the training script to save the URI returned by log_model()."
            )
            raise KeyError("model_uri is missing from experiment_info.json")

        model_uri = model_info["model_uri"]
        logger.debug("Model URI: %s", model_uri)

        model_version = mlflow.register_model(
            model_uri=model_uri,
            name=model_name,
        )

        logger.debug(
            "Registered model version: %s",
            model_version.version,
        )

        client = mlflow.tracking.MlflowClient()

        client.set_registered_model_alias(
            name=model_name,
            alias="Staging",
            version=model_version.version,
        )

        logger.info(
            "Model %s version %s registered and assigned the Staging alias",
            model_name,
            model_version.version,
        )

    except Exception:
        logger.exception("Error during model registration")
        raise


def main() -> None:
    try:
        model_info_path = "reports/figures/experiment_info.json"
        model_info = load_model_info(model_info_path)

        model_name = "my_model"

        register_model(
            model_name=model_name,
            model_info=model_info,
        )

        print("Model registration completed successfully")

    except FileNotFoundError:
        logger.exception("Required model-info file was not found")
        raise

    except KeyError as e:
        logger.exception("Missing required key in experiment_info.json: %s", e)
        raise

    except Exception:
        logger.exception("Failed to complete the model-registration process")
        raise


if __name__ == "__main__":
    main()