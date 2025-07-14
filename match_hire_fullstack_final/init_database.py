#!/usr/bin/env python3
"""
Database initialization script
"""

from models import init_db, engine, Base
from sqlalchemy import text

def init_database():
    """Initialize the database with the updated schema"""
    print("Initializing database...")
    
    # Drop all tables and recreate them
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    
    print("Database initialized successfully!")
    
    # Test the connection
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table';"))
            tables = [row[0] for row in result]
            print(f"Tables created: {tables}")
    except Exception as e:
        print(f"Error testing database: {e}")

if __name__ == "__main__":
    init_database() 