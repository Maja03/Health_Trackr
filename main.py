# Import the FastAPI app from the simple_advanced_api module
from simple_advanced_api import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)