"""Integration tests for the Flask sentiment-prediction application."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FLASK_APP_DIR = ROOT / "flask-app"

# The directory name contains a hyphen, so import it by adding its path.
sys.path.insert(0, str(FLASK_APP_DIR))

from app import app  # noqa: E402


class FlaskAppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()

    def test_home_page(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"<title>Sentioment Predictor</title>", response.data)

    def test_predict_page(self):
        response = self.client.post("/predict", data={"text": "I love this!"})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(
            b"Happy" in response.data or b"Sad" in response.data,
            "Response should contain either 'Happy' or 'Sad'",
        )


if __name__ == "__main__":
    unittest.main()
