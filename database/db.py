"""
database/db.py
--------------
SQLite database management module for the AI Skin Disease Detection application.
Handles initialization, prediction logging, history retrieval, deletion, and analytics.
"""

import os
import sqlite3
from datetime import datetime

# Path configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_DIR = os.path.join(BASE_DIR, "database")
DB_PATH = os.path.join(DB_DIR, "database.db")

def get_db_connection():
    """Establishes and returns a connection to SQLite database."""
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # Enables column-name dictionary-like access
    return conn

def init_db():
    """Initializes the database schema if tables do not exist."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_filename TEXT NOT NULL,
            image_path TEXT NOT NULL,
            prediction_code TEXT NOT NULL,
            prediction_name TEXT NOT NULL,
            confidence REAL NOT NULL,
            risk_level TEXT DEFAULT 'Moderate',
            category TEXT DEFAULT 'General',
            status TEXT DEFAULT 'Completed',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Create index for fast date sorting and search
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_created ON predictions(created_at DESC)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_code ON predictions(prediction_code)")

    conn.commit()
    conn.close()

def save_prediction(image_filename, image_path, prediction_code, prediction_name, confidence, risk_level="Moderate", category="General", status="Completed"):
    """
    Saves a new prediction record into SQLite.
    Returns the newly generated primary key id.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    now_iso = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cursor.execute("""
        INSERT INTO predictions 
        (image_filename, image_path, prediction_code, prediction_name, confidence, risk_level, category, status, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (image_filename, image_path, prediction_code, prediction_name, float(confidence), risk_level, category, status, now_iso))

    new_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return new_id

def get_all_predictions(limit=100, offset=0, search=None, filter_class=None):
    """Retrieves all past predictions with optional filtering and pagination."""
    conn = get_db_connection()
    cursor = conn.cursor()

    query = "SELECT * FROM predictions WHERE 1=1"
    params = []

    if search:
        query += " AND (prediction_name LIKE ? OR prediction_code LIKE ? OR image_filename LIKE ?)"
        term = f"%{search}%"
        params.extend([term, term, term])

    if filter_class and filter_class != "all":
        query += " AND prediction_code = ?"
        params.append(filter_class)

    query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()

    return [dict(row) for row in rows]

def get_prediction_by_id(prediction_id):
    """Retrieves a single prediction by its primary key."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM predictions WHERE id = ?", (prediction_id,))
    row = cursor.fetchone()
    conn.close()

    return dict(row) if row else None

def delete_prediction(prediction_id):
    """
    Deletes a prediction record by its ID.
    Returns the deleted record details so caller can clean up the file on disk.
    """
    record = get_prediction_by_id(prediction_id)
    if not record:
        return None

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM predictions WHERE id = ?", (prediction_id,))
    conn.commit()
    conn.close()

    return record

def get_database_stats():
    """Returns aggregated stats from the database for admin/status page."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) as total_count FROM predictions")
    total_count = cursor.fetchone()["total_count"]

    cursor.execute("SELECT AVG(confidence) as avg_conf FROM predictions")
    avg_conf_row = cursor.fetchone()["avg_conf"]
    avg_confidence = round(avg_conf_row * 100, 1) if avg_conf_row is not None else 0.0

    cursor.execute("SELECT COUNT(*) as high_risk_count FROM predictions WHERE risk_level IN ('Critical', 'High')")
    high_risk_count = cursor.fetchone()["high_risk_count"]

    cursor.execute("SELECT prediction_name, COUNT(*) as count FROM predictions GROUP BY prediction_name ORDER BY count DESC LIMIT 5")
    top_diseases = [dict(row) for row in cursor.fetchall()]

    conn.close()

    return {
        "total_records": total_count,
        "average_confidence_pct": avg_confidence,
        "high_risk_count": high_risk_count,
        "top_diseases": top_diseases
    }
