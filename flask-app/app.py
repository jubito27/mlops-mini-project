import os
import pickle
from pathlib import Path

import automac
import mlflow
import pandas as pd
from flask import Flask, redirect, render_template, request, url_for
from mlflow import MlflowClient

ROOT = Path(__file__).resolve().parents[1]
app = Flask(__name__)


def configure_mlflow():
    """Configure authenticated access to this project's DagsHub MLflow server."""
    token = os.getenv("DAGSHUB_PAT")
    if token:
        os.environ["MLFLOW_TRACKING_USERNAME"] = token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = token

    owner = os.getenv("DAGSHUB_REPO_OWNER", "jubito27")
    repo = os.getenv("DAGSHUB_REPO_NAME", "mlops-mini-project")
    mlflow.set_tracking_uri(f"https://dagshub.com/{owner}/{repo}.mlflow")


def load_production_model():
    """Load the exact model version assigned the Production alias."""
    configure_mlflow()
    client = MlflowClient()
    try:
        production_version = client.get_model_version_by_alias(
            name="my_model",
            alias="Production",
        )
    except Exception as exc:
        raise RuntimeError(
            "No model version is assigned the Production alias in DagsHub. "
            "Run the CI model-promotion step before starting the Flask app."
        ) from exc

    model_uri = f"models:/my_model/{production_version.version}"
    app.logger.info("Loading Production model version %s", production_version.version)
    return mlflow.pyfunc.load_model(model_uri)


# Load artifacts once when the application starts.
model = load_production_model()
vectorizer_path = ROOT / "data" / "transformed" / "vectorizer.pkl"
if not vectorizer_path.is_file():
    raise FileNotFoundError(
        f"Vectorizer artifact not found at {vectorizer_path}. "
        "Run the DVC feature-engineering stage first; it stores this file under data/transformed/."
    )

with vectorizer_path.open("rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)


@app.route("/")
def home():
    return render_template("index.html", result=None)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return redirect(url_for("home"))

    text = request.form.get("text", "").strip()
    if not text:
        return "Please enter some text before predicting.", 400

    try:
        cleaned_text = automac.text.transform(text)
        if not isinstance(cleaned_text, str) or not cleaned_text.strip():
            return "No usable text remains after preprocessing.", 400

        transformed_text = vectorizer.transform([cleaned_text])
        features = pd.DataFrame(
            transformed_text.toarray(),
            columns=vectorizer.get_feature_names_out(),
        )
        prediction = model.predict(features)[0]
        return render_template("index.html", result=prediction)

    except Exception:
        app.logger.exception("Prediction failed")
        return "Prediction failed. Check the server logs for details.", 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
