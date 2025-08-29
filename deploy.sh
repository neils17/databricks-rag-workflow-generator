#!/bin/bash

# Databricks RAG Workflow API Deployment Script
# This script builds and runs the Docker container

set -e

echo "🚀 Databricks RAG Workflow API Deployment"
echo "=========================================="

# Check if .env file exists
if [ ! -f .env ]; then
    echo "❌ Error: .env file not found!"
    echo "Please create a .env file with your configuration:"
    echo "cp config.env.example .env"
    echo "Then edit .env with your actual values."
    exit 1
fi

# Load environment variables
source .env

# Check required environment variables
required_vars=("OPENAI_API_KEY" "DATABRICKS_HOST" "DATABRICKS_TOKEN")
for var in "${required_vars[@]}"; do
    if [ -z "${!var}" ]; then
        echo "❌ Error: $var is not set in .env file"
        exit 1
    fi
done

echo "✅ Environment variables loaded"

# Build Docker image
echo "🔨 Building Docker image..."
docker build -t databricks-rag-api .

if [ $? -eq 0 ]; then
    echo "✅ Docker image built successfully"
else
    echo "❌ Docker build failed"
    exit 1
fi

# Run with Docker Compose
echo "🚀 Starting services with Docker Compose..."
docker-compose up -d

if [ $? -eq 0 ]; then
    echo "✅ Services started successfully"
    echo ""
    echo "🌐 API is running at: http://localhost:8000"
    echo "📚 API Documentation: http://localhost:8000/docs"
    echo "🔍 Health Check: http://localhost:8000/health"
    echo ""
    echo "📋 Available endpoints:"
    echo "  POST   /workflows                    - Create new workflow"
    echo "  GET    /workflows                    - List all workflows"
    echo "  GET    /workflows/{id}/status        - Get workflow status"
    echo "  POST   /workflows/{id}/query         - Query workflow"
    echo "  POST   /workflows/{id}/upload        - Upload documents"
    echo "  DELETE /workflows/{id}               - Delete workflow"
    echo ""
    echo "🔧 To stop services: docker-compose down"
    echo "📊 To view logs: docker-compose logs -f"
else
    echo "❌ Failed to start services"
    exit 1
fi
