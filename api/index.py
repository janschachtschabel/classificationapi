# Vercel serverless function entry point
import os
import sys

# Add paths for imports
current_dir = os.path.dirname(__file__)
parent_dir = os.path.join(current_dir, '..')
src_dir = os.path.join(parent_dir, 'src')

sys.path.insert(0, current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, src_dir)

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    from typing import Optional, List, Dict, Any
    import json
    
    # Create serverless-optimized FastAPI app
    app = FastAPI(
        title="Metadata Classification API",
        description="AI-powered metadata classification API (Serverless)",
        version="1.0.0"
    )
    
    # Add CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Serverless-optimized settings
    SERVERLESS_TIMEOUT = 8  # 8 seconds max for external calls
    MAX_VOCAB_SIZE = 50     # Reduced vocabulary size for serverless
    
    # Basic health endpoints
    @app.get("/")
    async def root():
        return {
            "message": "Metadata Classification API", 
            "status": "running",
            "environment": "serverless",
            "optimizations": {
                "timeout": f"{SERVERLESS_TIMEOUT}s",
                "max_vocab_size": MAX_VOCAB_SIZE
            }
        }
    
    @app.get("/health")
    async def health():
        return {"status": "healthy", "environment": "vercel_serverless"}
    
    # Simple classification request model
    class SimpleClassificationRequest(BaseModel):
        text: str
        mode: Optional[str] = "simple"
        max_results: Optional[int] = 5
    
    # Fallback classification endpoint (without external SKOS downloads)
    @app.post("/classify")
    async def classify_text_simple(request: SimpleClassificationRequest):
        """Simplified classification endpoint for serverless environment"""
        try:
            # Simple keyword-based classification without external vocabularies
            text = request.text.lower()
            
            # Basic educational content detection
            categories = []
            
            if any(word in text for word in ['math', 'mathematics', 'algebra', 'geometry']):
                categories.append({"category": "Mathematics", "confidence": 0.8})
            
            if any(word in text for word in ['science', 'biology', 'chemistry', 'physics']):
                categories.append({"category": "Science", "confidence": 0.8})
            
            if any(word in text for word in ['history', 'historical', 'past', 'ancient']):
                categories.append({"category": "History", "confidence": 0.7})
            
            if any(word in text for word in ['language', 'english', 'german', 'literature']):
                categories.append({"category": "Language Arts", "confidence": 0.7})
            
            if not categories:
                categories.append({"category": "General Education", "confidence": 0.5})
            
            return {
                "results": categories[:request.max_results],
                "metadata": {
                    "processing_mode": "serverless_simple",
                    "text_length": len(request.text),
                    "environment": "vercel",
                    "note": "Simplified classification without external SKOS vocabularies"
                }
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")
    
    # Try to use Mangum for ASGI adaptation
    try:
        from mangum import Mangum
        handler = Mangum(app, lifespan="off")
    except ImportError:
        # Fallback: export app directly for Vercel
        handler = app

except Exception as e:
    # Ultimate fallback
    def handler(event, context):
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({
                "error": f"Failed to initialize app: {str(e)}",
                "status": "error"
            })
        }

# Export both for compatibility
app_instance = handler if 'app' in locals() else None
