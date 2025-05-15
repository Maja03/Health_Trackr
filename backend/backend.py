from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from datetime import datetime
import os

app = Flask(__name__)
CORS(app)

# Database configuration using environment variables
app.config["SQLALCHEMY_DATABASE_URI"] = (
    f"postgresql://{os.environ['POSTGRES_USER']}:{os.environ['POSTGRES_PASSWORD']}@"
    f"{os.environ['POSTGRES_HOST']}:{5432}/{os.environ['POSTGRES_DB']}"
)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# Initialize extensions
db = SQLAlchemy(app)
migrate = Migrate(app, db)

# Database model
class HealthLog(db.Model):
    __tablename__ = "health_logs"
    id = db.Column(db.Integer, primary_key=True)
    log_date = db.Column(db.DateTime, default=datetime.utcnow)
    water = db.Column(db.String)
    sleep = db.Column(db.String)
    mood = db.Column(db.String)
    steps = db.Column(db.Integer)

# Health check route
@app.route("/", methods=["GET"])
def home():
    return "HealthTrackr backend is running!"

# Add a health log entry dynamically
@app.route("/log", methods=["POST"])
def log_health():
    try:
        data = request.get_json()
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
        return jsonify({"error": str(e)}), 500

# View all health logs
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

# Global error handler
@app.errorhandler(Exception)
def handle_error(e):
    return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0")
