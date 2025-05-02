FROM python:3.11-slim
WORKDIR /app
COPY backend.py .
RUN pip install flask psycopg2-binary
ENV FLASK_APP=backend.py
CMD ["flask", "run", "--host=0.0.0.0"]
