from flask import Flask, jsonify
import os
from flask_cors import CORS
import psycopg2
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Database connection parameters from environment variables
DB_PARAMS = {
    "dbname": os.environ.get("POSTGRES_DB"),
    "user": os.environ.get("POSTGRES_USER"),
    "password": os.environ.get("POSTGRES_PASSWORD"),
    "host": os.environ.get("POSTGRES_HOST", "db"),
    "port": 5432
}

def connect_db():
    return psycopg2.connect(**DB_PARAMS)

@app.route("/", methods=["GET"])
def home():
    return "HealthTrackr backend is running!"

@app.route("/log", methods=["POST"])
def log_health():
    data = {
        "date": datetime.utcnow().isoformat(),
        "water": "8 glasses",
        "sleep": "7 hours",
        "mood": "Feeling good",
        "steps": 8500
    }

    try:
        conn = connect_db()
        cur = conn.cursor()

        # Create table if it doesn't exist
        cur.execute("""
            CREATE TABLE IF NOT EXISTS HealthLogs (
                id SERIAL PRIMARY KEY,
                log_date TIMESTAMP,
                water TEXT,
                sleep TEXT,
                mood TEXT,
                steps INTEGER
            );
        """)

        # Insert log
        cur.execute("""
            INSERT INTO HealthLogs (log_date, water, sleep, mood, steps)
            VALUES (%s, %s, %s, %s, %s)
        """, (data["date"], data["water"], data["sleep"], data["mood"], data["steps"]))

        conn.commit()

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if cur: cur.close()
        if conn: conn.close()

    return jsonify({**data, "message": "Health log saved successfully!"})
