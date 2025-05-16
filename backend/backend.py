from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from datetime import datetime
import os
import traceback

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

app.config["SQLALCHEMY_DATABASE_URI"] = (
    f"postgresql://{os.environ.get('POSTGRES_USER')}:{os.environ.get('POSTGRES_PASSWORD')}@"
    f"{os.environ.get('POSTGRES_HOST', 'localhost')}:{os.environ.get('POSTGRES_PORT', 5432)}/"
    f"{os.environ.get('POSTGRES_DB')}"
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
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "Missing JSON body"}), 400

        required_fields = ["water", "sleep", "mood", "steps"]
        missing_fields = [f for f in required_fields if f not in data]
        if missing_fields:
            return jsonify({"error": f"Missing fields: {', '.join(missing_fields)}"}), 400

        log = HealthLog(
            water=data["water"],
            sleep=data["sleep"],
            mood=data["mood"],
            steps=int(data["steps"])
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

@app.route("/logs", methods=["GET"])
def get_logs():
    logs = HealthLog.query.order_by(HealthLog.log_date.desc()).all()
    return jsonify([
        {
            "id": log.id,
            "date": log.log_date.isoformat(),
            "water": log.water,
            "sleep": log.sleep,
            "mood": log.mood,
            "steps": log.steps
        } for log in logs
    ])

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
