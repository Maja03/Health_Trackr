from flask import Flask, jsonify
from flask_cors import CORS
from datetime import datetime

app = Flask(__name__)
CORS(app)

@app.route('/log', methods=['POST'])
def log_health():
    entry = {
        "date": datetime.utcnow().isoformat(),
        "water": "8 glasses",
        "sleep": "7 hours",
        "mood": "Feeling good",
        "steps": 8500,
        "message": "Great job! You're staying on track "
    }
    return jsonify(entry)

if __name__ == '__main__':
    app.run(debug=True)



