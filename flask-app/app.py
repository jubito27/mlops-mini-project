from flask import Flask, render_template, request
import automac
import mlflow
import dagshub 
import pickle
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
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # getting the text
        text = request.form.get('text', '')

        # cleaning the text
        cleaned_text = automac.text.transform(text)

        # tranforming the text
        tranformed_text = vectorizer.transform([cleaned_text])
        result = model.predict(tranformed_text.toarray())
        return f"Received text: {result}"
    cleaned_text = automac.text.transform(text)

    
    
    return cleaned_text

if __name__ == '__main__':
    app.run(debug=True)
