from flask import Flask, jsonify
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from datetime import datetime
import os

app = Flask(__name__)
CORS(app)

# SQLAlchemy Database URI setup
app.config["SQLALCHEMY_DATABASE_URI"] = (
    f"postgresql://{os.environ['POSTGRES_USER']}:{os.environ['POSTGRES_PASSWORD']}@"
    f"{os.environ['POSTGRES_HOST']}:{5432}/{os.environ['POSTGRES_DB']}"
)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

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

# Add a health log entry
@app.route("/log", methods=["POST"])
def log_health():
    log = HealthLog(
        water="8 glasses",
        sleep="7 hours",
        mood="Feeling good",
        steps=8500
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

if __name__ == "__main__":
    app.run(host="0.0.0.0")