"""Serverless-optimized FastAPI app for Vercel."""

import sys
import os

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Import core components
from core.config import settings
from core.logging import setup_logging

# Import API routes
from api.routes import classify, health
from api import scoring
from api.errors import (
    ClassificationAPIException,
    classification_api_exception_handler,
    general_exception_handler,
    http_exception_handler,
)

# Setup logging
setup_logging()

# Create FastAPI app without lifespan for serverless
app = FastAPI(
    title="Metadata Classification API",
    description="AI-powered metadata classification and scoring API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add exception handlers
app.add_exception_handler(HTTPException, http_exception_handler)
app.add_exception_handler(ClassificationAPIException, classification_api_exception_handler)
app.add_exception_handler(Exception, general_exception_handler)

# Include routers
app.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(classify.router, prefix="/classify", tags=["classification"])
app.include_router(scoring.router, prefix="/scoring", tags=["scoring"])

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Metadata Classification API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }
