# Vercel entry point for FastAPI - Self-contained version
import sys
import os

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    # Try to import FastAPI and create a minimal app
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    
    # Create minimal FastAPI app
    app = FastAPI(
        title="Metadata Classification API",
        description="AI-powered metadata classification API",
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
    
    # Basic health endpoint
    @app.get("/")
    async def root():
        return {"message": "Metadata Classification API", "status": "running"}
    
    @app.get("/health")
    async def health():
        return {"status": "healthy"}
    
    # Try to use Mangum if available
    try:
        from mangum import Mangum
        handler = Mangum(app, lifespan="off")
    except ImportError:
        # Direct FastAPI export for Vercel
        handler = app
        
except ImportError as e:
    print(f"FastAPI import failed: {e}")
    # Create basic HTTP handler
    def handler(request):
        return {
            'statusCode': 500,
            'headers': {'Content-Type': 'application/json'},
            'body': f'{{"error": "FastAPI not available: {str(e)}", "requirements": "Check requirements.txt"}}'
        }
