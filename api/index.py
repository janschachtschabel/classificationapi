# Vercel entry point for FastAPI
import sys
import os

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from main import app

# Export the FastAPI app for Vercel
handler = app
