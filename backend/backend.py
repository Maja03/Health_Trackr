from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
from datetime import datetime
import psycopg2
import os
from dotenv import load_dotenv

# Load .env
load_dotenv()

app = Flask(__name__)
CORS(app)

# PostgreSQL config from env
conn_params = {
    'host': os.getenv('DB_HOST'),
    'dbname': os.getenv('POSTGRES_DB'),
    'user': os.getenv('POSTGRES_USER'),
    'password': os.getenv('POSTGRES_PASSWORD'),
    'port': os.getenv('DB_PORT', 5432)
}

def get_connection():
    return psycopg2.connect(**conn_params)

def create_tables():
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS HealthLogs (
                id SERIAL PRIMARY KEY,
                log_date TIMESTAMP,
                water VARCHAR(50),
                sleep VARCHAR(50),
                mood VARCHAR(100),
                steps INT
            );
        """)
        conn.commit()
        cur.close()
        conn.close()
        print("✔️ Tables ready.")
    except Exception as e:
        print(f"Table creation error: {e}")

def insert_log(entry):
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO HealthLogs (log_date, water, sleep, mood, steps)
            VALUES (%s, %s, %s, %s, %s)
        """, (entry['date'], entry['water'], entry['sleep'], entry['mood'], entry['steps']))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"nsert error: {e}")

@app.route('/')
def index():
    return send_from_directory('../frontend', 'index.html')

@app.route('/log', methods=['POST'])
def log_health():
    entry = {
        "date": datetime.utcnow(),
        "water": "8 glasses",
        "sleep": "7 hours",
        "mood": "Feeling good",
        "steps": 8500,
        "message": "Great job! You're staying on track!"
    }
    insert_log(entry)
    entry['date'] = entry['date'].isoformat()
    return jsonify(entry)

if __name__ == '__main__':
    create_tables()
    app.run(host='0.0.0.0', port=5000, debug=True)
