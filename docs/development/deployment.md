# Deployment Guide

Complete deployment guide for the Metadata Classification API in production environments.

## Production Deployment Options

### Docker Deployment (Recommended)

#### Single Container Deployment

```dockerfile
# Dockerfile (already exists in project)
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY evaluation_criteria/ ./evaluation_criteria/
COPY vocabularies/ ./vocabularies/

# Create non-root user
RUN useradd --create-home --shell /bin/bash app
USER app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start application
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### Build and Run

```bash
# Build Docker image
docker build -t classification-api:latest .

# Run container with environment variables
docker run -d \
    --name classification-api \
    -p 8000:8000 \
    -e OPENAI_API_KEY="your-api-key" \
    -e OPENAI_DEFAULT_MODEL="gpt-4.1-mini" \
    -e LOG_LEVEL="INFO" \
    -e API_PORT="8000" \
    --restart unless-stopped \
    classification-api:latest

# Check container status
docker ps
docker logs classification-api
```

### Docker Compose Deployment

```yaml
# docker-compose.yml
version: '3.8'

services:
  classification-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - OPENAI_DEFAULT_MODEL=${OPENAI_DEFAULT_MODEL:-gpt-4.1-mini}
      - LOG_LEVEL=${LOG_LEVEL:-INFO}
      - API_PORT=8000
    volumes:
      - ./logs:/app/logs
      - ./data:/app/data
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - classification-api
    restart: unless-stopped

volumes:
  logs:
  data:
```

#### Environment Configuration

```bash
# .env.production
OPENAI_API_KEY=your-production-api-key
OPENAI_DEFAULT_MODEL=gpt-4.1-mini
LOG_LEVEL=INFO
API_PORT=8000

# Optional advanced settings
OPENAI_TIMEOUT_SECONDS=30
DEFAULT_TEMPERATURE=0.1
DEFAULT_MAX_TOKENS=1000
```

#### Deploy with Docker Compose

```bash
# Deploy
docker-compose --env-file .env.production up -d

# Scale if needed
docker-compose up -d --scale classification-api=3

# Monitor
docker-compose logs -f
docker-compose ps
```

## Cloud Platform Deployments

### AWS Deployment

#### AWS ECS with Fargate

```json
{
  "family": "classification-api",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::account:role/ecsTaskRole",
  "containerDefinitions": [
    {
      "name": "classification-api",
      "image": "your-account.dkr.ecr.region.amazonaws.com/classification-api:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "LOG_LEVEL",
          "value": "INFO"
        }
      ],
      "secrets": [
        {
          "name": "OPENAI_API_KEY",
          "valueFrom": "arn:aws:secretsmanager:region:account:secret:openai-api-key"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/classification-api",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"],
        "interval": 30,
        "timeout": 5,
        "retries": 3,
        "startPeriod": 60
      }
    }
  ]
}
```

#### AWS Lambda Deployment

```python
# lambda_handler.py
import json
from mangum import Mangum
from src.main import app

# Create Lambda handler
handler = Mangum(app, lifespan="off")

def lambda_handler(event, context):
    """AWS Lambda handler function."""
    return handler(event, context)
```

```yaml
# serverless.yml
service: classification-api

provider:
  name: aws
  runtime: python3.11
  region: us-west-2
  timeout: 30
  memorySize: 1024
  environment:
    OPENAI_API_KEY: ${env:OPENAI_API_KEY}
    LOG_LEVEL: INFO

functions:
  api:
    handler: lambda_handler.handler
    events:
      - http:
          path: /{proxy+}
          method: ANY
          cors: true
      - http:
          path: /
          method: ANY
          cors: true

plugins:
  - serverless-python-requirements

custom:
  pythonRequirements:
    dockerizePip: true
    slim: true
```

### Google Cloud Platform

#### Cloud Run Deployment

```yaml
# cloudbuild.yaml
steps:
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', 'gcr.io/$PROJECT_ID/classification-api', '.']
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', 'gcr.io/$PROJECT_ID/classification-api']
  - name: 'gcr.io/cloud-builders/gcloud'
    args:
      - 'run'
      - 'deploy'
      - 'classification-api'
      - '--image'
      - 'gcr.io/$PROJECT_ID/classification-api'
      - '--region'
      - 'us-central1'
      - '--platform'
      - 'managed'
      - '--allow-unauthenticated'
      - '--port'
      - '8000'
      - '--memory'
      - '2Gi'
      - '--cpu'
      - '2'
      - '--max-instances'
      - '10'
      - '--set-env-vars'
      - 'LOG_LEVEL=INFO'
      - '--set-secrets'
      - 'OPENAI_API_KEY=openai-api-key:latest'
```

#### Deploy to Cloud Run

```bash
# Build and deploy
gcloud builds submit --config cloudbuild.yaml

# Or deploy directly
gcloud run deploy classification-api \
    --source . \
    --region us-central1 \
    --platform managed \
    --allow-unauthenticated \
    --port 8000 \
    --memory 2Gi \
    --cpu 2 \
    --max-instances 10 \
    --set-env-vars LOG_LEVEL=INFO \
    --set-secrets OPENAI_API_KEY=openai-api-key:latest
```

### Azure Deployment

#### Azure Container Instances

```bash
# Create resource group
az group create --name classification-api-rg --location eastus

# Create container instance
az container create \
    --resource-group classification-api-rg \
    --name classification-api \
    --image your-registry.azurecr.io/classification-api:latest \
    --cpu 2 \
    --memory 4 \
    --ports 8000 \
    --environment-variables LOG_LEVEL=INFO \
    --secure-environment-variables OPENAI_API_KEY=your-api-key \
    --restart-policy Always
```

#### Azure App Service

```yaml
# azure-pipelines.yml
trigger:
  branches:
    include:
      - main

pool:
  vmImage: 'ubuntu-latest'

variables:
  dockerRegistryServiceConnection: 'your-registry-connection'
  imageRepository: 'classification-api'
  containerRegistry: 'your-registry.azurecr.io'
  dockerfilePath: '$(Build.SourcesDirectory)/Dockerfile'
  tag: '$(Build.BuildId)'

stages:
- stage: Build
  displayName: Build and push stage
  jobs:
  - job: Build
    displayName: Build
    steps:
    - task: Docker@2
      displayName: Build and push an image to container registry
      inputs:
        command: buildAndPush
        repository: $(imageRepository)
        dockerfile: $(dockerfilePath)
        containerRegistry: $(dockerRegistryServiceConnection)
        tags: |
          $(tag)
          latest

- stage: Deploy
  displayName: Deploy stage
  dependsOn: Build
  jobs:
  - deployment: Deploy
    displayName: Deploy
    environment: 'production'
    strategy:
      runOnce:
        deploy:
          steps:
          - task: AzureWebAppContainer@1
            displayName: 'Azure Web App on Container Deploy'
            inputs:
              azureSubscription: 'your-subscription'
              appName: 'classification-api'
              containers: '$(containerRegistry)/$(imageRepository):$(tag)'
```

## Kubernetes Deployment

### Kubernetes Manifests

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: classification-api
---
# k8s/secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: api-secrets
  namespace: classification-api
type: Opaque
stringData:
  OPENAI_API_KEY: "your-api-key"
---
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: api-config
  namespace: classification-api
data:
  LOG_LEVEL: "INFO"
  API_PORT: "8000"
  OPENAI_DEFAULT_MODEL: "gpt-4.1-mini"
---
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: classification-api
  namespace: classification-api
  labels:
    app: classification-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: classification-api
  template:
    metadata:
      labels:
        app: classification-api
    spec:
      containers:
      - name: classification-api
        image: classification-api:latest
        ports:
        - containerPort: 8000
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-secrets
              key: OPENAI_API_KEY
        envFrom:
        - configMapRef:
            name: api-config
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: classification-api-service
  namespace: classification-api
spec:
  selector:
    app: classification-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: ClusterIP
---
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: classification-api-ingress
  namespace: classification-api
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/rate-limit: "100"
spec:
  tls:
  - hosts:
    - api.yourdomain.com
    secretName: api-tls
  rules:
  - host: api.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: classification-api-service
            port:
              number: 80
```

### Deploy to Kubernetes

```bash
# Apply all manifests
kubectl apply -f k8s/

# Check deployment status
kubectl get pods -n classification-api
kubectl get services -n classification-api
kubectl get ingress -n classification-api

# View logs
kubectl logs -f deployment/classification-api -n classification-api

# Scale deployment
kubectl scale deployment classification-api --replicas=5 -n classification-api
```

## Load Balancing and Reverse Proxy

### Nginx Configuration

```nginx
# nginx.conf
upstream classification_api {
    server classification-api-1:8000;
    server classification-api-2:8000;
    server classification-api-3:8000;
}

server {
    listen 80;
    server_name api.yourdomain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name api.yourdomain.com;

    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req zone=api burst=20 nodelay;

    # Security headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains";

    location / {
        proxy_pass http://classification_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts
        proxy_connect_timeout 30s;
        proxy_send_timeout 30s;
        proxy_read_timeout 30s;
        
        # Buffer settings
        proxy_buffering on;
        proxy_buffer_size 4k;
        proxy_buffers 8 4k;
    }

    location /health {
        access_log off;
        proxy_pass http://classification_api;
    }
}
```

### HAProxy Configuration

```
# haproxy.cfg
global
    daemon
    maxconn 4096

defaults
    mode http
    timeout connect 5000ms
    timeout client 50000ms
    timeout server 50000ms
    option httplog

frontend api_frontend
    bind *:80
    bind *:443 ssl crt /etc/ssl/certs/api.pem
    redirect scheme https if !{ ssl_fc }
    
    # Rate limiting
    stick-table type ip size 100k expire 30s store http_req_rate(10s)
    http-request track-sc0 src
    http-request reject if { sc_http_req_rate(0) gt 20 }
    
    default_backend api_backend

backend api_backend
    balance roundrobin
    option httpchk GET /health
    
    server api1 classification-api-1:8000 check
    server api2 classification-api-2:8000 check
    server api3 classification-api-3:8000 check
```

## Monitoring and Observability

### Prometheus Metrics

```python
# src/monitoring/metrics.py
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from fastapi import Response
import time

# Metrics
REQUEST_COUNT = Counter('api_requests_total', 'Total API requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('api_request_duration_seconds', 'Request duration', ['method', 'endpoint'])
ACTIVE_CONNECTIONS = Gauge('api_active_connections', 'Active connections')
OPENAI_REQUESTS = Counter('openai_requests_total', 'OpenAI API requests', ['model', 'status'])
CLASSIFICATION_CACHE_HITS = Counter('classification_cache_hits_total', 'Cache hits')

class MetricsMiddleware:
    """Middleware to collect metrics."""
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            start_time = time.time()
            
            # Track active connections
            ACTIVE_CONNECTIONS.inc()
            
            try:
                await self.app(scope, receive, send)
            finally:
                # Record metrics
                duration = time.time() - start_time
                method = scope["method"]
                path = scope["path"]
                
                REQUEST_DURATION.labels(method=method, endpoint=path).observe(duration)
                ACTIVE_CONNECTIONS.dec()
        else:
            await self.app(scope, receive, send)

# Metrics endpoint
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type="text/plain")
```

### Health Checks

```python
# Enhanced health check
from src.api.health import router
from fastapi import Depends
import asyncio

@router.get("/health/detailed")
async def detailed_health_check():
    """Detailed health check with dependency status."""
    
    checks = {}
    
    # Check OpenAI API
    try:
        # Simple API test
        response = await openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=1
        )
        checks["openai"] = {"status": "healthy", "response_time": "< 1s"}
    except Exception as e:
        checks["openai"] = {"status": "unhealthy", "error": str(e)}
    
    # Check vocabulary loading
    try:
        # Test vocabulary access
        vocab_loader = VocabularyLoader()
        test_vocab = vocab_loader.load_vocabulary("discipline")
        checks["vocabularies"] = {"status": "healthy", "count": len(test_vocab)}
    except Exception as e:
        checks["vocabularies"] = {"status": "unhealthy", "error": str(e)}
    
    # Overall status
    overall_status = "healthy" if all(
        check["status"] == "healthy" for check in checks.values()
    ) else "unhealthy"
    
    return {
        "status": overall_status,
        "timestamp": datetime.utcnow().isoformat(),
        "checks": checks
    }
```

### Logging Configuration

```python
# src/core/logging_config.py
import logging
import sys
from pythonjsonlogger import jsonlogger

def setup_production_logging():
    """Setup structured logging for production."""
    
    # JSON formatter
    formatter = jsonlogger.JsonFormatter(
        '%(asctime)s %(name)s %(levelname)s %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # File handler
    file_handler = logging.FileHandler('/app/logs/api.log')
    file_handler.setFormatter(formatter)
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    # Suppress noisy loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
```

## Security Considerations

### Environment Security

```bash
# Use secrets management
export OPENAI_API_KEY=$(aws secretsmanager get-secret-value --secret-id openai-api-key --query SecretString --output text)

# Or with Kubernetes secrets
kubectl create secret generic api-secrets \
    --from-literal=OPENAI_API_KEY="your-api-key" \
    --namespace=classification-api
```

### API Security

```python
# src/security/auth.py
from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API key authentication."""
    
    if credentials.credentials != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    
    return credentials.credentials

# Apply to endpoints
@router.post("/classify", dependencies=[Depends(verify_api_key)])
async def classify_text_endpoint(...):
    ...
```

### Network Security

```yaml
# Network policies for Kubernetes
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: classification-api-netpol
  namespace: classification-api
spec:
  podSelector:
    matchLabels:
      app: classification-api
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8000
  egress:
  - to: []
    ports:
    - protocol: TCP
      port: 443  # HTTPS outbound
    - protocol: TCP
      port: 53   # DNS
    - protocol: UDP
      port: 53   # DNS
```

## Deployment Automation

### CI/CD Pipeline

```yaml
# .github/workflows/deploy.yml
name: Deploy to Production

on:
  push:
    branches: [ main ]
    tags: [ 'v*' ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Run tests
      run: |
        pip install -r requirements.txt
        pytest --cov=src

  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Build Docker image
      run: |
        docker build -t classification-api:${{ github.sha }} .
        docker tag classification-api:${{ github.sha }} classification-api:latest
    
    - name: Push to registry
      run: |
        echo ${{ secrets.DOCKER_PASSWORD }} | docker login -u ${{ secrets.DOCKER_USERNAME }} --password-stdin
        docker push classification-api:${{ github.sha }}
        docker push classification-api:latest

  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
    - name: Deploy to production
      run: |
        # Update Kubernetes deployment
        kubectl set image deployment/classification-api \
          classification-api=classification-api:${{ github.sha }} \
          --namespace=classification-api
        
        # Wait for rollout
        kubectl rollout status deployment/classification-api \
          --namespace=classification-api
```

### Deployment Scripts

```bash
#!/bin/bash
# deploy.sh

set -e

# Configuration
IMAGE_TAG=${1:-latest}
NAMESPACE=${2:-classification-api}

echo "Deploying classification-api:$IMAGE_TAG to $NAMESPACE"

# Update image in deployment
kubectl set image deployment/classification-api \
    classification-api=classification-api:$IMAGE_TAG \
    --namespace=$NAMESPACE

# Wait for rollout to complete
kubectl rollout status deployment/classification-api \
    --namespace=$NAMESPACE \
    --timeout=300s

# Verify deployment
kubectl get pods -n $NAMESPACE
kubectl get services -n $NAMESPACE

# Run health check
EXTERNAL_IP=$(kubectl get service classification-api-service -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
curl -f http://$EXTERNAL_IP/health

echo "Deployment completed successfully!"
```

This comprehensive deployment guide covers all major deployment scenarios and production considerations for the Metadata Classification API.
