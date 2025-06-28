# Render.com Deployment Guide

## 🚀 Deployment Steps

### 1. **Repository Setup**
- Ensure your code is pushed to GitHub
- Make sure `render.yaml`, `Dockerfile`, and `requirements.txt` are in the root directory

### 2. **Render.com Account Setup**
1. Go to [render.com](https://render.com)
2. Sign up/Login with your GitHub account
3. Connect your GitHub repository

### 3. **Deploy from Blueprint**
1. Click "New" → "Blueprint"
2. Connect your GitHub repository
3. Render will automatically detect the `render.yaml` file
4. Click "Apply" to start deployment

### 4. **Environment Variables**
Set these in the Render dashboard:

**Required:**
- `OPENAI_API_KEY` - Your OpenAI API key

**Optional (already configured in render.yaml):**
- `OPENAI_DEFAULT_MODEL=gpt-4o-mini`
- `API_HOST=0.0.0.0`
- `API_PORT=10000`
- `LOG_LEVEL=INFO`
- `LOG_FORMAT=json`

### 5. **Deployment Configuration**

The app is configured with:
- **Runtime**: Docker
- **Plan**: Starter (free tier)
- **Region**: Frankfurt
- **Port**: 10000 (Render standard)
- **Health Check**: `/health`
- **Start Command**: `uvicorn src.main:app --host 0.0.0.0 --port 10000`

### 6. **Features Available**

✅ **Full FastAPI Application**
- Complete metadata classification API
- SKOS vocabulary processing
- OpenAI integration
- Sentence transformers for semantic filtering
- Comprehensive logging and error handling

✅ **Production Ready**
- Docker containerization
- Health checks
- Proper environment variable handling
- Optimized for cloud deployment

### 7. **Testing Deployment**

Once deployed, test these endpoints:
- `GET /` - API information
- `GET /health` - Health check
- `GET /docs` - Interactive API documentation
- `POST /classify` - Classification endpoint

### 8. **Monitoring**

- Check Render dashboard for logs
- Monitor health check status
- View deployment metrics

## 🔧 Troubleshooting

**Build Issues:**
- Check Dockerfile syntax
- Verify all dependencies in requirements.txt
- Ensure Python version compatibility

**Runtime Issues:**
- Check environment variables
- Verify OpenAI API key is set
- Monitor application logs in Render dashboard

**Performance:**
- Starter plan has limited resources
- Consider upgrading to Standard plan for production
- Monitor memory and CPU usage

## 📝 Notes

- Render.com provides better support for full Python applications compared to serverless platforms
- Docker deployment ensures consistent environment
- All original API functionality is preserved
- No code modifications needed for deployment
