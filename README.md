# AI-Based Skin Disease Detection

An AI-powered dermatological screening web application designed for academic research and college project demonstration. The system enables users to upload photos of skin lesions and obtain preliminary deep learning-based classification across 7 disease categories using Computer Vision and Convolutional Neural Networks (CNNs) with Transfer Learning.

> [!IMPORTANT]
> **Medical Disclaimer**: This application is intended strictly for preliminary screening and academic research purposes. It does **NOT** provide medical diagnoses, treatment advice, or prescriptions, and does not replace a clinical examination by a licensed dermatologist.

---

## 🌟 Key Features

1. **Healthcare AI UI/UX**: Modern responsive interface built with clean medical aesthetics, soft gradients, glassmorphism touches, and interactive animations.
2. **Strict Non-Fake Model Policy**: If a trained TensorFlow model is missing, the platform transparently displays an **"AI Model Not Connected"** status instead of pretending to perform real predictions.
3. **Interactive Drag & Drop**: Easy image upload zone with file size (max 10MB) & extension (JPG, PNG, WEBP) validation and instant client-side preview.
4. **OpenCV Preprocessing Pipeline**: Automatic color profile conversion (BGR to RGB), spatial tensor resizing to 224&times;224 pixels, and intensity normalization (`[0, 1]`).
5. **Confidence & Risk Visualizations**: Live probability meters, 7-class distribution breakdowns, and automated warning banners for low-confidence (<60%) screenings.
6. **Prediction History & Audit Logs**: Local SQLite database logging past predictions with search, class filtering, and one-click record deletion.
7. **Admin & Model Telemetry Dashboard**: Live monitoring for model connectivity, loaded class mapping inspection, and API latency testing.
8. **Pluggable Model Architecture**: Easily swap in any MobileNetV2, ResNet50, EfficientNet, or VGG16 weights trained on HAM10000 / ISIC datasets.

---

## 🛠️ Tech Stack

- **Frontend**: HTML5, CSS3 (Vanilla Healthcare Design System), JavaScript (ES6+), Plus Jakarta Sans typography
- **Backend**: Python 3, Flask, REST API, Werkzeug
- **AI / Deep Learning**: TensorFlow, Keras (Transfer Learning), OpenCV, Pillow, NumPy
- **Database**: SQLite3 (`database/database.db`)

---

## 📁 Project Structure

```
ai-skin-disease-detection/
│
├── app.py                          # Flask application, REST API routing, & model inference
├── requirements.txt                # Python dependencies
├── create_sample_model.py          # Script to generate a ready-to-test MobileNetV2 .keras model
├── test_app.py                     # Automated unit and integration test suite
├── README.md                       # Comprehensive project documentation
│
├── model/
│   ├── model.keras                 # Place your trained Keras model here (.keras or .h5)
│   ├── classes.json                # Configurable mapping of disease classes & medical metadata
│   └── README.md                   # Model architecture and training guide
│
├── database/
│   ├── db.py                       # SQLite database manager (CRUD, logging, analytics)
│   └── database.db                 # Auto-generated SQLite database
│
├── static/
│   ├── css/
│   │   └── style.css               # Healthcare AI design system & animations
│   ├── js/
│   │   ├── main.js                 # Global navigation, mobile drawer, & toasts
│   │   ├── analyze.js              # Drag-and-drop upload & multi-stage progress loader
│   │   ├── history.js              # Prediction history filter, search, & delete
│   │   └── model-status.js         # Real-time health check & latency tester
│   ├── images/
│   │   ├── hero-illustration.svg   # Custom SVG vector graphics
│   │   └── placeholder.svg         # Fallback placeholder image
│   └── uploads/                    # Secure storage for uploaded lesion images
│
└── templates/
    ├── base.html                   # Base layout with header, disclaimer, and footer
    ├── index.html                  # 1. Home Page (Overview, 4-step workflow, tech, features)
    ├── analyze.html                # 2. Image Analysis Page (Upload dropzone & preview)
    ├── result.html                 # 3. Result Page (Class prediction, confidence, clinical info)
    ├── history.html                # 4. Prediction History Page (Table & delete actions)
    ├── about.html                  # 5. About Project Page (Problem, CNN, Transfer Learning, Ethics)
    └── model-status.html           # 7. Admin / Model Status Dashboard
```

---

## 🚀 Quick Start Guide (Run Locally)

### 1. Clone or Open the Project Directory
```bash
cd ai-skin-disease-detection
```

### 2. Create and Activate Virtual Environment
**On Windows:**
```cmd
python -m venv venv
venv\Scripts\activate
```

**On macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Initialize or Connect the AI Model
You have two easy options:

- **Option A (Generate Ready-to-Run Starter Model)**:
  Run the included script to create a clean MobileNetV2 model initialized with 7 HAM10000 output classes:
  ```bash
  python create_sample_model.py
  ```
- **Option B (Use Your Own Trained Model)**:
  Place your trained weights file in `model/model.keras` or `model/model.h5`.

### 5. Launch the Web Application
```bash
python app.py
```

### 6. Open in Browser
Visit **[http://127.0.0.1:5000](http://127.0.0.1:5000)** in your web browser.

---

## 🧠 Model Integration & Class Customization

### Where to Place Your Trained Model File
Place your model file directly in the `model/` folder:
- `model/model.keras` *(Recommended for Keras 3 / TF 2.16+)*
- `model/model.h5` *(Legacy Keras HDF5)*

The server automatically detects and loads the model upon startup or when you click **"Reload Model"** on the `/model-status` dashboard.

### How to Change Disease Class Names
The application reads all diagnostic categories from `model/classes.json`. To customize the classes or adjust medical descriptions:

1. Open `model/classes.json`.
2. Modify or reorder class entries to match your model's output neurons:
```json
{
  "id": 0,
  "code": "mel",
  "name": "Melanoma",
  "short_name": "Melanoma (MEL)",
  "category": "Malignant Skin Cancer",
  "risk_level": "Critical",
  "badge_color": "rose",
  "description": "Melanoma is an aggressive skin cancer originating in pigment-producing melanocytes...",
  "clinical_notes": "Requires immediate evaluation by a licensed dermatologist."
}
```
3. Save the file and restart `app.py` (or click **Reload Model** on the web UI).

---

## 📡 REST API Documentation

| Method | Endpoint | Description | Sample Response |
|---|---|---|---|
| `POST` | `/api/predict` | Uploads skin image for OpenCV preprocessing and Keras inference | `{"success": true, "prediction": "Melanoma", "confidence": 0.94, ...}` |
| `GET` | `/api/history` | Fetches logged screening records with optional search/filtering | `{"success": true, "total": 12, "records": [...]}` |
| `DELETE` | `/api/history/<id>` | Deletes a record and cleans up the image file from disk | `{"success": true, "message": "Record deleted."}` |
| `GET` | `/api/prediction/<id>` | Retrieves details of a specific screening report | `{"success": true, "record": {...}}` |
| `GET` | `/api/model-status` | Returns live AI model connection state and DB statistics | `{"success": true, "model_status": {...}}` |
| `POST` | `/api/reload-model` | Triggers a live re-scan of the `model/` directory | `{"success": true, "model_status": {...}}` |

---

## 📊 Dataset References (Academic Credits)

- **HAM10000 Dataset**: Tschandl, P., Rosendahl, C. & Kittler, H. *The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions.* Sci Data 5, 180161 (2018). [DOI: 10.1038/sdata.2018.161](https://doi.org/10.1038/sdata.2018.161)
- **ISIC Archive**: International Skin Imaging Collaboration. [https://www.isic-archive.com/](https://www.isic-archive.com/)

---

## 🧪 Running Automated Tests

Run the included automated test suite to verify routes, API endpoints, database operations, and error handling:
```bash
python test_app.py
```

---

## 📜 License & College Project Disclaimer
Developed for educational and academic project presentation. Free to adapt and extend for research and learning.
