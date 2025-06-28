# Vercel entry point for FastAPI
import sys
import os

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    # Import serverless-optimized app
    from app import app
    from mangum import Mangum
    
    # Use Mangum to adapt ASGI app for serverless
    handler = Mangum(app, lifespan="off")
    
except ImportError as e:
    print(f"Import error: {e}")
    # Fallback without Mangum
    try:
        from app import app
        handler = app
    except Exception as e2:
        print(f"Fallback import error: {e2}")
        def handler(event, context):
            return {
                'statusCode': 500,
                'headers': {'Content-Type': 'application/json'},
                'body': '{"error": "Failed to import FastAPI app"}'
            }
except Exception as e:
    print(f"General error: {e}")
    def handler(event, context):
        return {
            'statusCode': 500,
            'headers': {'Content-Type': 'application/json'},
            'body': f'{{"error": "{str(e)}"}}'  
        }
