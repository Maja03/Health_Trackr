from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from datetime import datetime
import os
import traceback

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # allow CORS from any origin

app.config["SQLALCHEMY_DATABASE_URI"] = (
    f"postgresql://{os.environ['POSTGRES_USER']}:{os.environ['POSTGRES_PASSWORD']}@"
    f"{os.environ['POSTGRES_HOST']}:{5432}/{os.environ['POSTGRES_DB']}"
)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)
migrate = Migrate(app, db)

class HealthLog(db.Model):
    __tablename__ = "health_logs"
    id = db.Column(db.Integer, primary_key=True)
    log_date = db.Column(db.DateTime, default=datetime.utcnow)
    water = db.Column(db.String)
    sleep = db.Column(db.String)
    mood = db.Column(db.String)
    steps = db.Column(db.Integer)

@app.route("/", methods=["GET"])
def home():
    return "HealthTrackr backend is running!"

@app.route("/log", methods=["POST"])
def log_health():
    try:
        print("Request Headers:", request.headers)
        data = request.get_json(silent=True)

        if data is None:
            return jsonify({"error": "Invalid or missing JSON in request body"}), 400

        required_fields = ["water", "sleep", "mood", "steps"]
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({"error": f"Missing fields: {', '.join(missing_fields)}"}), 400

        log = HealthLog(
            water=data.get("water"),
            sleep=data.get("sleep"),
            mood=data.get("mood"),
            steps=data.get("steps")
        )
        db.session.add(log)
        db.session.commit()

        return jsonify({
            "date": log.log_date.isoformat(),
            "water": log.water,
            "sleep": log.sleep,
            "mood": log.mood,
            "steps": log.steps,
            "message": "Health log saved successfully!"
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500