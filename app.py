"""
app.py
------
Flask Backend Application for AI-Based Skin Disease Detection.
Provides web routing, REST API endpoints, image validation & preprocessing with OpenCV,
and deep learning model inference with TensorFlow/Keras.
"""

import os
import json
import uuid
import time
import logging
from datetime import datetime
from PIL import Image

from flask import (
    Flask, render_template, request, jsonify,
    redirect, url_for, send_from_directory, abort
)
from werkzeug.utils import secure_filename

# Database utilities
from database.db import (
    init_db, save_prediction, get_all_predictions,
    get_prediction_by_id, delete_prediction, get_database_stats
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("SkinDiseaseApp")

# Application Initialization
app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "ai-skin-disease-detection-secret-key-2026")
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # Max 10MB upload limit

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
MODEL_DIR = os.path.join(BASE_DIR, "model")
CLASSES_FILE = os.path.join(MODEL_DIR, "classes.json")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}

# Global model state
AI_MODEL = None
MODEL_STATUS = {
    "connected": False,
    "model_file": None,
    "framework": "TensorFlow / Keras",
    "classes_count": 0,
    "input_shape": [224, 224, 3],
    "message": "AI model not connected. Please place a trained model in model/model.keras or run create_sample_model.py.",
    "loaded_at": None
}
DISEASE_CLASSES = []

def load_disease_classes():
    """Loads disease classes metadata from classes.json."""
    global DISEASE_CLASSES
    if os.path.exists(CLASSES_FILE):
        try:
            with open(CLASSES_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                DISEASE_CLASSES = data.get("classes", [])
                logger.info(f"Loaded {len(DISEASE_CLASSES)} disease classes from {CLASSES_FILE}")
        except Exception as e:
            logger.error(f"Failed to load classes.json: {e}")
            DISEASE_CLASSES = []
    else:
        logger.warning(f"classes.json not found at {CLASSES_FILE}")
        DISEASE_CLASSES = []

def init_ai_model():
    """
    Attempts to load a trained TensorFlow/Keras model from model/ directory.
    Supports .keras and .h5 formats.
    If not available, leaves MODEL_STATUS['connected'] = False.
    """
    global AI_MODEL, MODEL_STATUS

    load_disease_classes()

    keras_path = os.path.join(MODEL_DIR, "model.keras")
    h5_path = os.path.join(MODEL_DIR, "model.h5")

    target_path = None
    if os.path.exists(keras_path):
        target_path = keras_path
    elif os.path.exists(h5_path):
        target_path = h5_path

    if not target_path:
        logger.warning("No model file (model.keras or model.h5) found in model/ directory.")
        MODEL_STATUS["connected"] = False
        MODEL_STATUS["model_file"] = None
        MODEL_STATUS["message"] = "AI model not connected. Please place a trained model in model/model.keras or run python create_sample_model.py."
        MODEL_STATUS["classes_count"] = len(DISEASE_CLASSES)
        return

    try:
        import tensorflow as tf
        from tensorflow import keras
        import numpy as np

        logger.info(f"Attempting to load model from {target_path} (TensorFlow {tf.__version__})...")
        AI_MODEL = keras.models.load_model(target_path, compile=False)

        # Warmup model inference
        dummy_input = np.zeros((1, 224, 224, 3), dtype=np.float32)
        _ = AI_MODEL.predict(dummy_input, verbose=0)

        MODEL_STATUS["connected"] = True
        MODEL_STATUS["model_file"] = os.path.basename(target_path)
        MODEL_STATUS["classes_count"] = len(DISEASE_CLASSES)
        MODEL_STATUS["message"] = f"AI Model successfully loaded ({os.path.basename(target_path)})."
        MODEL_STATUS["loaded_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"Model successfully loaded: {MODEL_STATUS['model_file']}")

    except ImportError:
        logger.error("TensorFlow is not installed in the Python environment.")
        MODEL_STATUS["connected"] = False
        MODEL_STATUS["message"] = "TensorFlow is not installed. Please run: pip install -r requirements.txt"
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        MODEL_STATUS["connected"] = False
        MODEL_STATUS["message"] = f"Error loading model: {str(e)}"

# Preprocessing helper
def allowed_file(filename):
    """Validates file extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_image_file(file_storage):
    """
    Validates file headers and PIL integrity to prevent malicious or non-image uploads.
    """
    if not file_storage or file_storage.filename == "":
        return False, "No file selected."

    if not allowed_file(file_storage.filename):
        return False, f"Unsupported file type. Please upload JPG, JPEG, PNG, or WEBP images."

    try:
        # Verify with PIL without consuming entire file stream permanently
        pos = file_storage.tell()
        img = Image.open(file_storage)
        img.verify()
        file_storage.seek(pos)
        return True, "Valid image"
    except Exception as e:
        return False, "Invalid image data. File could not be decoded as an image."

def preprocess_image_for_model(image_path, target_size=(224, 224)):
    """
    Preprocesses the uploaded skin lesion image using OpenCV and NumPy:
    1. Reads image via OpenCV (BGR).
    2. Converts BGR to RGB color space.
    3. Resizes image to target dimensions (224x224).
    4. Normalizes pixel intensities to [0.0, 1.0].
    5. Adds batch dimension (1, 224, 224, 3).
    """
    import numpy as np

    try:
        import cv2
        # Read image using OpenCV
        img = cv2.imread(image_path)
        if img is None:
            # Fallback to PIL if OpenCV fails to read direct path
            pil_img = Image.open(image_path).convert("RGB")
            img = np.array(pil_img)[:, :, ::-1]  # RGB to BGR for cv2 pipeline

        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize to model input shape
        img_resized = cv2.resize(img_rgb, target_size, interpolation=cv2.INTER_AREA)

        # Convert to float32 and normalize [0, 1]
        img_normalized = img_resized.astype(np.float32) / 255.0

        # Expand batch dimension
        img_batch = np.expand_dims(img_normalized, axis=0)
        return img_batch

    except Exception as e:
        logger.warning(f"OpenCV processing error, falling back to PIL: {e}")
        # Pure PIL fallback
        pil_img = Image.open(image_path).convert("RGB")
        pil_img = pil_img.resize(target_size, Image.Resampling.LANCZOS)
        arr = np.array(pil_img, dtype=np.float32) / 255.0
        return np.expand_dims(arr, axis=0)


# ============================================================================
# Page Routes (Frontend HTML Views)
# ============================================================================

@app.route("/")
def index():
    """1. Home Page: Project Overview, Workflow, Features, Tech Stack, and Disclaimer."""
    return render_template("index.html", model_status=MODEL_STATUS, classes=DISEASE_CLASSES)

@app.route("/analyze")
def analyze():
    """2. Image Analysis Page: Drag & drop upload area, file validation, analyze button."""
    return render_template("analyze.html", model_status=MODEL_STATUS)

@app.route("/result/<int:prediction_id>")
def result(prediction_id):
    """3. Result Page: Displays image, prediction, confidence, disease background, disclaimer."""
    record = get_prediction_by_id(prediction_id)
    if not record:
        return render_template("result.html", record=None, error="Prediction record not found.")

    # Match with class metadata
    class_meta = next(
        (c for c in DISEASE_CLASSES if c["code"] == record["prediction_code"]),
        None
    )

    return render_template(
        "result.html",
        record=record,
        class_meta=class_meta,
        model_status=MODEL_STATUS
    )

@app.route("/history")
def history():
    """4. Prediction History Page: Shows past predictions with search, filter, and delete."""
    return render_template("history.html", classes=DISEASE_CLASSES)

@app.route("/about")
def about():
    """5. About Project Page: Problem statement, CNN, Transfer Learning, HAM10000/ISIC, ethics."""
    return render_template("about.html", classes=DISEASE_CLASSES)

@app.route("/model-status")
def model_status_page():
    """7. Admin/Model Status Page: Health check, model connection status, database statistics."""
    db_stats = get_database_stats()
    return render_template(
        "model-status.html",
        model_status=MODEL_STATUS,
        classes=DISEASE_CLASSES,
        stats=db_stats
    )


# ============================================================================
# REST API Endpoints
# ============================================================================

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    POST /api/predict
    Accepts skin image file, performs OpenCV preprocessing and Keras inference,
    saves record to SQLite, and returns JSON prediction response.
    """
    import numpy as np

    # Check if AI model is connected
    if not MODEL_STATUS["connected"] or AI_MODEL is None:
        return jsonify({
            "success": False,
            "model_connected": False,
            "error": "AI Model is not connected. Please ensure model/model.keras exists or run create_sample_model.py.",
            "instructions": "Place your trained Keras model inside the 'model/' directory or check /model-status."
        }), 503

    if "image" not in request.files:
        return jsonify({
            "success": False,
            "error": "No image file provided in the request form data ('image')."
        }), 400

    file = request.files["image"]
    is_valid, val_msg = validate_image_file(file)
    if not is_valid:
        return jsonify({"success": False, "error": val_msg}), 400

    try:
        # Generate safe unique filename
        ext = file.filename.rsplit(".", 1)[1].lower()
        unique_filename = f"{uuid.uuid4().hex}_{int(time.time())}.{ext}"
        saved_file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
        rel_image_path = f"/static/uploads/{unique_filename}"

        # Save uploaded image to disk
        file.save(saved_file_path)

        # Preprocess with OpenCV
        input_tensor = preprocess_image_for_model(saved_file_path, target_size=(224, 224))

        # Model Inference
        raw_predictions = AI_MODEL.predict(input_tensor, verbose=0)[0]
        top_idx = int(np.argmax(raw_predictions))
        top_confidence = float(raw_predictions[top_idx])

        # Match predicted class with metadata
        if top_idx < len(DISEASE_CLASSES):
            predicted_class = DISEASE_CLASSES[top_idx]
            pred_code = predicted_class["code"]
            pred_name = predicted_class["name"]
            risk_level = predicted_class.get("risk_level", "Moderate")
            category = predicted_class.get("category", "General")
            description = predicted_class.get("description", "")
            clinical_notes = predicted_class.get("clinical_notes", "")
        else:
            pred_code = f"class_{top_idx}"
            pred_name = f"Class {top_idx}"
            risk_level = "Moderate"
            category = "General"
            description = "Standard dermatological classification."
            clinical_notes = "Consult a dermatologist for confirmation."

        # Determine confidence status
        is_low_confidence = top_confidence < 0.60
        status_label = "Low Confidence - Dermatologist Consultation Strongly Advised" if is_low_confidence else "Analysis Complete"

        # Build full class distribution breakdown for frontend charts
        class_distribution = []
        for i, score in enumerate(raw_predictions):
            class_name = DISEASE_CLASSES[i]["short_name"] if i < len(DISEASE_CLASSES) else f"Class {i}"
            class_distribution.append({
                "class_id": i,
                "name": class_name,
                "probability": float(score),
                "percentage": round(float(score) * 100, 2)
            })
        class_distribution.sort(key=lambda x: x["probability"], reverse=True)

        # Log prediction into SQLite database
        record_id = save_prediction(
            image_filename=unique_filename,
            image_path=rel_image_path,
            prediction_code=pred_code,
            prediction_name=pred_name,
            confidence=top_confidence,
            risk_level=risk_level,
            category=category,
            status=status_label
        )

        return jsonify({
            "success": True,
            "id": record_id,
            "prediction": pred_name,
            "prediction_code": pred_code,
            "confidence": round(top_confidence, 4),
            "confidence_percentage": round(top_confidence * 100, 1),
            "risk_level": risk_level,
            "category": category,
            "status": status_label,
            "is_low_confidence": is_low_confidence,
            "description": description,
            "clinical_notes": clinical_notes,
            "class_distribution": class_distribution,
            "image_url": rel_image_path,
            "result_url": f"/result/{record_id}",
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "disclaimer": "This result is for preliminary screening only and is not a medical diagnosis. Please consult a qualified dermatologist."
        }), 200

    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": f"An error occurred while analyzing the image: {str(e)}"
        }), 500

@app.route("/api/history", methods=["GET"])
def api_history():
    """
    GET /api/history
    Returns list of logged prediction records with optional search and filtering.
    """
    try:
        search = request.args.get("search", None)
        filter_class = request.args.get("class", None)
        limit = int(request.args.get("limit", 100))
        offset = int(request.args.get("offset", 0))

        records = get_all_predictions(limit=limit, offset=offset, search=search, filter_class=filter_class)
        return jsonify({
            "success": True,
            "total": len(records),
            "records": records
        }), 200
    except Exception as e:
        logger.error(f"Error fetching history: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/history/<int:prediction_id>", methods=["DELETE"])
def api_delete_history(prediction_id):
    """
    DELETE /api/history/<id>
    Deletes a prediction record and cleans up the associated uploaded image file.
    """
    try:
        deleted_record = delete_prediction(prediction_id)
        if not deleted_record:
            return jsonify({"success": False, "error": "Record not found"}), 404

        # Delete image file from static/uploads if exists
        img_filename = deleted_record.get("image_filename")
        if img_filename:
            file_on_disk = os.path.join(UPLOAD_FOLDER, img_filename)
            if os.path.exists(file_on_disk):
                try:
                    os.remove(file_on_disk)
                    logger.info(f"Removed deleted image file: {file_on_disk}")
                except Exception as file_err:
                    logger.warning(f"Could not remove image file {file_on_disk}: {file_err}")

        return jsonify({
            "success": True,
            "message": f"Prediction record #{prediction_id} deleted successfully."
        }), 200
    except Exception as e:
        logger.error(f"Error deleting history #{prediction_id}: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/prediction/<int:prediction_id>", methods=["GET"])
def api_get_prediction(prediction_id):
    """
    GET /api/prediction/<id>
    Returns JSON details for a specific prediction ID.
    """
    record = get_prediction_by_id(prediction_id)
    if not record:
        return jsonify({"success": False, "error": "Record not found"}), 404

    class_meta = next((c for c in DISEASE_CLASSES if c["code"] == record["prediction_code"]), None)
    return jsonify({
        "success": True,
        "record": record,
        "class_details": class_meta
    }), 200

@app.route("/api/model-status", methods=["GET"])
def api_model_status():
    """
    GET /api/model-status
    Returns the real-time AI model connection status, class counts, and database statistics.
    """
    db_stats = get_database_stats()
    return jsonify({
        "success": True,
        "model_status": MODEL_STATUS,
        "classes": DISEASE_CLASSES,
        "database_stats": db_stats,
        "server_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }), 200

@app.route("/api/reload-model", methods=["POST"])
def api_reload_model():
    """
    POST /api/reload-model
    Re-scans the model/ directory and attempts to reconnect without restarting Flask.
    """
    try:
        init_ai_model()
        return jsonify({
            "success": True,
            "model_status": MODEL_STATUS
        }), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ============================================================================
# Error Handlers
# ============================================================================

@app.errorhandler(404)
def page_not_found(e):
    return render_template("base.html", not_found=True), 404

@app.errorhandler(413)
def file_too_large(e):
    return jsonify({
        "success": False,
        "error": "File size exceeds the 10MB limit. Please upload a smaller skin image."
    }), 413

@app.errorhandler(500)
def server_error(e):
    return jsonify({
        "success": False,
        "error": "Internal server error. Please check server logs."
    }), 500


# ============================================================================
# Application Entry Point
# ============================================================================

if __name__ == "__main__":
    # Initialize SQLite tables
    init_db()
    # Initialize / connect AI model
    init_ai_model()

    port = int(os.environ.get("PORT", 5000))
    print(f"\n=======================================================")
    print(f"  AI-Based Skin Disease Detection Web Application")
    print(f"  Running on: http://127.0.0.1:{port}")
    print(f"  Model Connected: {MODEL_STATUS['connected']}")
    print(f"=======================================================\n")
    app.run(host="0.0.0.0", port=port, debug=True)
