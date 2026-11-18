from flask import Flask, render_template, request ,  redirect , url_for
import automac
import mlflow
import dagshub 
import pickle
import pandas as pd
from mlflow import MlflowClient
app = Flask(__name__)


def _get_latest_model_version(model_name):
    client = MlflowClient()
    latest_version = client.get_latest_versions(model_name , stages=['Production'])
    if not latest_version:
        latest_version = client.get_latest_versions(model_name , stages=['None'])
    return latest_version[0].version if latest_version else None

try:
    dagshub.init(repo_owner='jubito27', repo_name='mlops-mini-project', mlflow=True)
    mlflow.set_tracking_uri('https://dagshub.com/jubito27/mlops-mini-project.mlflow')
except Exception as e:
    print("Error occured as :" , e)

model_name= 'my_model'
model_version = _get_latest_model_version(model_name=model_name)
model_uri = f'models:/{model_name}/{model_version}'

model = mlflow.pyfunc.load_model(model_uri)
vectorizer = pickle.load(open('model/vectorizer.pkl' , 'rb'))

@app.route('/') 
def home():
    return render_template('index.html' , result = None)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    # Opening /predict directly should return to the input page.
    if request.method == 'GET':
        return redirect(url_for('home'))

    text = request.form.get('text', '').strip()

    if not text:
        return "Please enter some text before predicting.", 400

    try:
        # Use the same text-cleaning process used during training.
        cleaned_text = automac.text.transform(text)

        if not isinstance(cleaned_text, str) or not cleaned_text.strip():
            return "No usable text remains after preprocessing.", 400

        # Use the saved fitted vectorizer; do not fit it again here.
        transformed_text = vectorizer.transform([cleaned_text])

        # Preserve the vocabulary column names used during training.
        feature = pd.DataFrame(
            transformed_text.toarray(),
            columns=vectorizer.get_feature_names_out(),
        )

        predictions = model.predict(feature)

        # Display the prediction rather than the entire result array.
        return render_template('index.html' , result=predictions[0])

    except Exception:
        app.logger.exception("Prediction failed")
        return "Prediction failed. Check the server logs for details.", 500

if __name__ == '__main__':
    app.run(debug=True)
