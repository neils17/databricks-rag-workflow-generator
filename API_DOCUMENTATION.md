# Databricks RAG Workflow API Documentation

This document provides comprehensive documentation for the Databricks RAG Workflow API, available in both FastAPI and Azure Functions versions.

## Table of Contents

1. [Overview](#overview)
2. [Deployment Options](#deployment-options)
3. [API Endpoints](#api-endpoints)
4. [Authentication](#authentication)
5. [Examples](#examples)
6. [Error Handling](#error-handling)
7. [Rate Limiting](#rate-limiting)

## Overview

The Databricks RAG Workflow API provides a RESTful interface for creating and managing RAG (Retrieval-Augmented Generation) workflows in Databricks. It automates the creation of Databricks assets, document processing, and provides querying capabilities.

### Features

- ✅ Create RAG workflows with automatic Databricks asset generation
- ✅ Upload and process documents
- ✅ Query workflows with natural language
- ✅ MLflow experiment tracking
- ✅ Real-time status monitoring
- ✅ Background task processing

## Deployment Options

### Option 1: Docker Container (Recommended for Development)

```bash
# Clone the repository
git clone <your-repo-url>
cd databricks-rag-workflow

# Set up environment variables
cp config.env.example .env
# Edit .env with your credentials

# Deploy using Docker Compose
./deploy.sh
```

### Option 2: Azure Functions (Production)

```bash
# Install Azure CLI
# Login to Azure
az login

# Deploy to Azure Functions
./deploy_azure.sh
```

### Option 3: Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export OPENAI_API_KEY="your-key"
export DATABRICKS_HOST="your-host"
export DATABRICKS_TOKEN="your-token"

# Run the API
python app.py
```

## API Endpoints

### Base URLs

- **Docker/Local**: `http://localhost:8000`
- **Azure Functions**: `https://your-function-app.azurewebsites.net/api`

### Authentication

All endpoints require valid environment variables:
- `OPENAI_API_KEY`: Your OpenAI API key
- `DATABRICKS_HOST`: Your Databricks workspace URL
- `DATABRICKS_TOKEN`: Your Databricks personal access token

### Endpoints

#### 1. Health Check

**GET** `/health`

Check if the API is running and healthy.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "service": "databricks-rag-workflow-api"
}
```

#### 2. Create Workflow

**POST** `/workflows`

Create a new RAG workflow.

**Request Body:**
```json
{
  "workflow_name": "My RAG Workflow",
  "create_assets": true,
  "upload_documents": false,
  "document_description": "Optional description",
  "documents": [
    {
      "content": "Document content here...",
      "metadata": {
        "source": "manual",
        "type": "text"
      }
    }
  ]
}
```

**Response:**
```json
{
  "workflow_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "initializing",
  "message": "Workflow creation started",
  "created_at": "2024-01-15T10:30:00Z"
}
```

#### 3. Get Workflow Status

**GET** `/workflows/{workflow_id}/status`

Get the current status of a workflow.

**Response:**
```json
{
  "workflow_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "progress": "Workflow completed successfully",
  "error": null
}
```

#### 4. List Workflows

**GET** `/workflows`

List all workflows.

**Response:**
```json
{
  "workflows": [
    {
      "workflow_id": "550e8400-e29b-41d4-a716-446655440000",
      "status": "completed",
      "created_at": "2024-01-15T10:30:00Z",
      "progress": "Workflow completed successfully",
      "workflow_name": "My RAG Workflow"
    }
  ]
}
```

#### 5. Query Workflow

**POST** `/workflows/{workflow_id}/query`

Query a completed workflow.

**Request Body:**
```json
{
  "query": "What is the company's policy on remote work?"
}
```

**Response:**
```json
{
  "query": "What is the company's policy on remote work?",
  "response": "Based on the company policy document, employees are allowed to work remotely up to 3 days per week...",
  "context": [
    "Section 3.2: Remote Work Policy - Employees may work remotely up to 3 days per week...",
    "Section 3.3: Remote Work Requirements - Remote workers must maintain regular communication..."
  ],
  "similarity_score": 0.89,
  "response_time": 2.34
}
```

#### 6. Upload Documents

**POST** `/workflows/{workflow_id}/upload`

Upload documents to a workflow.

**Request (Multipart Form):**
```
files: [file1.txt, file2.pdf, ...]
description: "Optional description"
```

**Response:**
```json
{
  "workflow_id": "550e8400-e29b-41d4-a716-446655440000",
  "uploaded_files": [
    "/tmp/workflow_550e8400-e29b-41d4-a716-446655440000/document_1.txt",
    "/tmp/workflow_550e8400-e29b-41d4-a716-446655440000/document_2.pdf"
  ],
  "message": "Successfully uploaded 2 documents"
}
```

#### 7. Delete Workflow

**DELETE** `/workflows/{workflow_id}`

Delete a workflow.

**Response:**
```json
{
  "message": "Workflow 550e8400-e29b-41d4-a716-446655440000 deleted successfully"
}
```

## Examples

### Complete Workflow Example

```bash
# 1. Create a workflow
curl -X POST "http://localhost:8000/workflows" \
  -H "Content-Type: application/json" \
  -d '{
    "workflow_name": "Company Policy RAG",
    "create_assets": true,
    "upload_documents": true,
    "documents": [
      {
        "content": "Company Policy: Remote work is allowed up to 3 days per week...",
        "metadata": {"source": "hr", "type": "policy"}
      }
    ]
  }'

# Response: {"workflow_id": "abc123", "status": "initializing", ...}

# 2. Check status
curl "http://localhost:8000/workflows/abc123/status"

# 3. Wait for completion, then query
curl -X POST "http://localhost:8000/workflows/abc123/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the remote work policy?"}'
```

### Python Client Example

```python
import requests
import time

class RAGWorkflowClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def create_workflow(self, name, documents=None):
        payload = {
            "workflow_name": name,
            "create_assets": True,
            "upload_documents": documents is not None,
            "documents": documents or []
        }
        
        response = requests.post(f"{self.base_url}/workflows", json=payload)
        return response.json()
    
    def wait_for_completion(self, workflow_id, timeout=300):
        start_time = time.time()
        while time.time() - start_time < timeout:
            status = self.get_status(workflow_id)
            if status["status"] == "completed":
                return status
            elif status["status"] == "failed":
                raise Exception(f"Workflow failed: {status.get('error')}")
            time.sleep(5)
        raise Exception("Workflow timed out")
    
    def get_status(self, workflow_id):
        response = requests.get(f"{self.base_url}/workflows/{workflow_id}/status")
        return response.json()
    
    def query(self, workflow_id, query):
        payload = {"query": query}
        response = requests.post(f"{self.base_url}/workflows/{workflow_id}/query", json=payload)
        return response.json()

# Usage
client = RAGWorkflowClient()

# Create workflow
workflow = client.create_workflow("Test Workflow", [
    {"content": "This is a test document about AI and machine learning."}
])

# Wait for completion
client.wait_for_completion(workflow["workflow_id"])

# Query
result = client.query(workflow["workflow_id"], "What is this document about?")
print(result["response"])
```

### JavaScript/Node.js Example

```javascript
class RAGWorkflowClient {
    constructor(baseUrl = 'http://localhost:8000') {
        this.baseUrl = baseUrl;
    }

    async createWorkflow(name, documents = null) {
        const payload = {
            workflow_name: name,
            create_assets: true,
            upload_documents: documents !== null,
            documents: documents || []
        };

        const response = await fetch(`${this.baseUrl}/workflows`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        return response.json();
    }

    async getStatus(workflowId) {
        const response = await fetch(`${this.baseUrl}/workflows/${workflowId}/status`);
        return response.json();
    }

    async waitForCompletion(workflowId, timeout = 300000) {
        const startTime = Date.now();
        
        while (Date.now() - startTime < timeout) {
            const status = await this.getStatus(workflowId);
            
            if (status.status === 'completed') {
                return status;
            } else if (status.status === 'failed') {
                throw new Error(`Workflow failed: ${status.error}`);
            }
            
            await new Promise(resolve => setTimeout(resolve, 5000));
        }
        
        throw new Error('Workflow timed out');
    }

    async query(workflowId, query) {
        const response = await fetch(`${this.baseUrl}/workflows/${workflowId}/query`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query })
        });

        return response.json();
    }
}

// Usage
async function main() {
    const client = new RAGWorkflowClient();

    // Create workflow
    const workflow = await client.createWorkflow('Test Workflow', [
        { content: 'This is a test document about AI and machine learning.' }
    ]);

    // Wait for completion
    await client.waitForCompletion(workflow.workflow_id);

    // Query
    const result = await client.query(workflow.workflow_id, 'What is this document about?');
    console.log(result.response);
}

main().catch(console.error);
```

## Error Handling

The API returns standard HTTP status codes:

- **200**: Success
- **202**: Accepted (workflow creation started)
- **400**: Bad Request (invalid parameters)
- **404**: Not Found (workflow not found)
- **500**: Internal Server Error

Error responses include details:

```json
{
  "error": "Workflow not found",
  "details": "The specified workflow ID does not exist"
}
```

## Rate Limiting

- **Workflow Creation**: 10 requests per minute
- **Queries**: 100 requests per minute
- **Status Checks**: 200 requests per minute

## Monitoring and Logging

### Docker/Local
```bash
# View logs
docker-compose logs -f rag-api

# Health check
curl http://localhost:8000/health
```

### Azure Functions
```bash
# View logs
az functionapp logs tail --name your-function-app --resource-group your-rg

# Monitor in Azure Portal
# Go to your Function App > Monitor > Logs
```

## Security Considerations

1. **Environment Variables**: Store sensitive credentials in environment variables
2. **HTTPS**: Use HTTPS in production
3. **Authentication**: Consider adding API key authentication for production
4. **Input Validation**: All inputs are validated using Pydantic models
5. **Rate Limiting**: Implement rate limiting for production deployments

## Troubleshooting

### Common Issues

1. **Authentication Errors**
   - Verify your Databricks token is valid and not expired
   - Check that your Databricks host URL is correct

2. **Workflow Timeout**
   - Increase timeout values in deployment configuration
   - Check Databricks workspace connectivity

3. **Document Upload Failures**
   - Ensure documents are in supported formats (txt, pdf, md)
   - Check file size limits

4. **Query Failures**
   - Ensure workflow is in "completed" status before querying
   - Verify OpenAI API key is valid

### Debug Mode

Enable debug logging by setting the environment variable:
```bash
export LOG_LEVEL=DEBUG
```

## Support

For issues and questions:
1. Check the logs for detailed error messages
2. Verify your environment configuration
3. Test with the health check endpoint
4. Review the Databricks and OpenAI API documentation
