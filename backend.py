from flask import Flask, jsonify
from flask import Flask, jsonify, send_from_directory  # <-- added send_from_directory
from flask_cors import CORS
from datetime import datetime
import pyodbc

app = Flask(__name__)
CORS(app)

# SQL Server connection details
server = r'DESKTOP-B9NRMRB\MSSQLSERVER01'
database = 'HealthTrackrDB'
username = 'maja'   # replace with your username
password = 'Konwalia09'   # replace with your password
driver = '{ODBC Driver 17 for SQL Server}'  # or 18, depending on your system

connection_string = f'DRIVER={driver};SERVER={server};DATABASE={database};UID={username};PWD={password}'

def insert_log(entry):
    try:
        conn = pyodbc.connect(connection_string)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO HealthLogs (log_date, water, sleep, mood, steps)
            VALUES (?, ?, ?, ?, ?)
        """, entry['date'], entry['water'], entry['sleep'], entry['mood'], entry['steps'])
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"Database error: {e}")

@app.route('/')  # <-- NEW route
def index():
    return send_from_directory('', 'index.html')  # <-- serves your HTML file

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

    # Insert into the database
    insert_log(entry)

    # Format date nicely for frontend
    entry['date'] = entry['date'].isoformat()
    return jsonify(entry)

if __name__ == '__main__':
    app.run(debug=True)
