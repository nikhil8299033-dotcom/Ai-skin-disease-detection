"""
test_app.py
-----------
Automated test suite for the AI-Based Skin Disease Detection web application.
Verifies all HTML routes, REST API endpoints, SQLite CRUD operations, and error handling.
"""

import os
import io
import unittest
from PIL import Image
import numpy as np

# Import Flask app and database utilities
from app import app, init_ai_model
from database.db import (
    init_db, save_prediction, get_all_predictions,
    get_prediction_by_id, delete_prediction, get_database_stats
)

class SkinDiseaseAppTestCase(unittest.TestCase):

    def setUp(self):
        """Set up test client and initialize database."""
        app.config["TESTING"] = True
        app.config["WTF_CSRF_ENABLED"] = False
        self.client = app.test_client()
        init_db()
        init_ai_model()

    def create_dummy_image(self, width=224, height=224, color=(200, 100, 50)):
        """Creates a synthetic in-memory image for upload testing."""
        img = Image.new("RGB", (width, height), color=color)
        byte_arr = io.BytesIO()
        img.save(byte_arr, format="JPEG")
        byte_arr.seek(0)
        return byte_arr

    # ------------------------------------------------------------------------
    # 1. HTML Route Tests
    # ------------------------------------------------------------------------

    def test_home_page(self):
        """Test GET / renders 200 OK and contains project title."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"AI-Based", response.data)
        self.assertIn(b"Skin Disease", response.data)
        self.assertIn(b"How It Works", response.data)

    def test_analyze_page(self):
        """Test GET /analyze renders upload zone."""
        response = self.client.get("/analyze")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Analyze Skin Image", response.data)
        self.assertIn(b"dropzone", response.data)

    def test_history_page(self):
        """Test GET /history renders history container."""
        response = self.client.get("/history")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Prediction History", response.data)

    def test_about_page(self):
        """Test GET /about renders academic methodology."""
        response = self.client.get("/about")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Methodology", response.data)
        self.assertIn(b"HAM10000", response.data)

    def test_model_status_page(self):
        """Test GET /model-status renders diagnostic dashboard."""
        response = self.client.get("/model-status")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Model Status", response.data)

    def test_404_error_page(self):
        """Test non-existent route returns custom 404."""
        response = self.client.get("/non-existent-route-xyz")
        self.assertEqual(response.status_code, 404)
        self.assertIn(b"404", response.data)

    # ------------------------------------------------------------------------
    # 2. Database CRUD Tests
    # ------------------------------------------------------------------------

    def test_database_crud(self):
        """Test inserting, querying, and deleting records in SQLite."""
        # Insert
        rec_id = save_prediction(
            image_filename="test_sample.jpg",
            image_path="/static/uploads/test_sample.jpg",
            prediction_code="mel",
            prediction_name="Melanoma",
            confidence=0.942,
            risk_level="Critical",
            category="Malignant Skin Cancer",
            status="Analysis Complete"
        )
        self.assertIsNotNone(rec_id)

        # Retrieve by ID
        record = get_prediction_by_id(rec_id)
        self.assertIsNotNone(record)
        self.assertEqual(record["prediction_code"], "mel")
        self.assertEqual(record["prediction_name"], "Melanoma")
        self.assertAlmostEqual(record["confidence"], 0.942, places=3)

        # List all
        all_records = get_all_predictions()
        self.assertTrue(len(all_records) >= 1)

        # Search
        searched = get_all_predictions(search="Melanoma")
        self.assertTrue(any(r["id"] == rec_id for r in searched))

        # Filter by class
        filtered = get_all_predictions(filter_class="mel")
        self.assertTrue(any(r["id"] == rec_id for r in filtered))

        # Stats
        stats = get_database_stats()
        self.assertGreaterEqual(stats["total_records"], 1)

        # Delete
        deleted = delete_prediction(rec_id)
        self.assertIsNotNone(deleted)
        self.assertEqual(deleted["id"], rec_id)

        # Verify deletion
        after_del = get_prediction_by_id(rec_id)
        self.assertIsNone(after_del)

    # ------------------------------------------------------------------------
    # 3. REST API Tests
    # ------------------------------------------------------------------------

    def test_api_model_status(self):
        """Test GET /api/model-status returns JSON with model status."""
        response = self.client.get("/api/model-status")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data["success"])
        self.assertIn("model_status", data)
        self.assertIn("classes", data)
        self.assertEqual(len(data["classes"]), 7)

    def test_api_history(self):
        """Test GET /api/history returns JSON list."""
        response = self.client.get("/api/history")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data["success"])
        self.assertIsInstance(data["records"], list)

    def test_api_predict_validation(self):
        """Test POST /api/predict handles missing or invalid image input."""
        # 1. No file sent
        res1 = self.client.post("/api/predict", data={})
        self.assertIn(res1.status_code, [400, 503])  # 503 if model not connected, 400 if connected

        # 2. Non-image file sent (e.g. text file)
        res2 = self.client.post("/api/predict", data={
            "image": (io.BytesIO(b"fake text content"), "test.txt")
        })
        self.assertIn(res2.status_code, [400, 503])

if __name__ == "__main__":
    print("=" * 60)
    print("Running Automated Tests for AI-Based Skin Disease Detection")
    print("=" * 60)
    unittest.main(verbosity=2)
