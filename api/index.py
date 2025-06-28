# Ultra-minimal Vercel function - guaranteed to work
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional

# Create minimal app
app = FastAPI(title="Classification API")

@app.get("/")
def root():
    return {"message": "Classification API is running", "status": "ok"}

@app.get("/health")
def health():
    return {"status": "healthy"}

class ClassifyRequest(BaseModel):
    text: str
    mode: Optional[str] = "simple"

@app.post("/classify")
def classify(request: ClassifyRequest):
    text = request.text.lower()
    categories = []
    
    if any(word in text for word in ['math', 'mathematics']):
        categories.append({"category": "Mathematics", "confidence": 0.8})
    elif any(word in text for word in ['science', 'biology']):
        categories.append({"category": "Science", "confidence": 0.8})
    else:
        categories.append({"category": "General", "confidence": 0.5})
    
    return {
        "results": categories,
        "text_length": len(request.text),
        "status": "success"
    }

# Export for Vercel
try:
    from mangum import Mangum
    handler = Mangum(app)
except ImportError:
    handler = app
