FROM python:3.11-slim
WORKDIR /app
COPY backend.py .
RUN pip install flask flask-cors psycopg2-binary flask-migrate
ENV FLASK_APP=backend.py
CMD ["flask", "run", "--host=0.0.0.0"]