# Ultimate RAG Workflow - Databricks Integration

A complete end-to-end RAG (Retrieval-Augmented Generation) system that creates Databricks notebooks, enables document upload, provides interactive querying, tracks everything in MLflow, and offers both CLI and API interfaces.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Databricks workspace with API access
- OpenAI API key

### Setup
1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Databrick_PoC
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**
   ```bash
   cp config.env.example .env
   # Edit .env with your credentials:
   # DATABRICKS_HOST=your-databricks-host
   # DATABRICKS_TOKEN=your-databricks-token
   # OPENAI_API_KEY=your-openai-api-key
   ```

4. **Choose your deployment method:**

   **Option A: CLI Interface (Original)**
   ```bash
   python3 ultimate_rag_workflow.py
   ```

   **Option B: Docker Container (Recommended)**
   ```bash
   ./deploy.sh
   ```

   **Option C: Azure Functions (Production)**
   ```bash
   ./deploy_azure.sh
   ```

## 📁 Project Structure

### Core Files
- **`ultimate_rag_workflow.py`** - Main CLI workflow that creates everything
- **`app.py`** - FastAPI application for REST API
- **`enhanced_workflow_fixed.py`** - Enhanced RAG system with MLflow tracking
- **`enhanced_rag_client.py`** - Local RAG client for interactive querying
- **`natural_language_uploader.py`** - Upload documents using natural language
- **`document_loader.py`** - Load documents from various sources

### API & Containerization
- **`Dockerfile`** - Docker container configuration
- **`docker-compose.yml`** - Multi-service deployment with Redis
- **`azure_function/`** - Azure Functions deployment files
- **`deploy.sh`** - Docker deployment script
- **`deploy_azure.sh`** - Azure Functions deployment script

### Configuration
- **`config.env.example`** - Environment variables template
- **`requirements.txt`** - Python dependencies
- **`API_DOCUMENTATION.md`** - Complete API documentation
- **`README.md`** - This file

### Sample Data
- **`sample_company_policy.txt`** - Sample company policy document
- **`api_technical_guide.md`** - Sample technical documentation
- **`pdfs/`** - Directory containing sample PDF files

## 🎯 Features

### ✅ Complete End-to-End Solution
- **Databricks Notebook Creation** - Automated notebook generation
- **Document Upload** - Multiple upload methods (natural language, manual, file loader)
- **Interactive Querying** - Real-time document Q&A
- **MLflow Tracking** - Complete experiment tracking and analytics
- **Performance Monitoring** - Response times, similarity scores, metrics
- **REST API** - Full RESTful API for programmatic access
- **Containerization** - Docker support for easy deployment
- **Azure Functions** - Serverless deployment option

### 📚 Document Upload Methods
1. **Natural Language** - "Upload all PDF files"
2. **Manual Entry** - Type documents directly
3. **File Loader** - Load from text files, JSON, or manual entry
4. **Skip Upload** - Upload later in Databricks
5. **API Upload** - Programmatic document upload via REST API

### 🔍 Querying Capabilities
- **Interactive Session** - Real-time Q&A with documents
- **Single Queries** - One-off questions
- **API Queries** - REST API for programmatic querying
- **MLflow Integration** - Every query logged with metrics
- **Performance Analytics** - Response times, similarity scores, context utilization

## 🚀 Usage

### 1. CLI Interface (Original)
```bash
python3 ultimate_rag_workflow.py
```

### 2. Docker Container (Recommended)
```bash
# Deploy with Docker Compose
./deploy.sh

# API will be available at http://localhost:8000
# Documentation at http://localhost:8000/docs
```

### 3. Azure Functions (Production)
```bash
# Deploy to Azure Functions
./deploy_azure.sh

# API will be available at https://your-function-app.azurewebsites.net/api
```

### 4. REST API Usage

#### Create a Workflow
```bash
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
```

#### Query a Workflow
```bash
curl -X POST "http://localhost:8000/workflows/{workflow_id}/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the remote work policy?"}'
```

#### Check Workflow Status
```bash
curl "http://localhost:8000/workflows/{workflow_id}/status"
```

### 5. Python Client Example
```python
import requests

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
    
    def query(self, workflow_id, query):
        payload = {"query": query}
        response = requests.post(f"{self.base_url}/workflows/{workflow_id}/query", json=payload)
        return response.json()

# Usage
client = RAGWorkflowClient()
workflow = client.create_workflow("Test Workflow", [
    {"content": "This is a test document about AI and machine learning."}
])
result = client.query(workflow["workflow_id"], "What is this document about?")
print(result["response"])
```

## 📊 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/workflows` | Create new workflow |
| GET | `/workflows` | List all workflows |
| GET | `/workflows/{id}/status` | Get workflow status |
| POST | `/workflows/{id}/query` | Query workflow |
| POST | `/workflows/{id}/upload` | Upload documents |
| DELETE | `/workflows/{id}` | Delete workflow |

## 🐳 Docker Deployment

### Quick Start
```bash
# Build and run with Docker Compose
./deploy.sh

# Or manually:
docker-compose up -d
```

### Services
- **rag-api**: FastAPI application (port 8000)
- **redis**: Session storage and caching (port 6379)

### Environment Variables
The Docker container uses the same `.env` file as the CLI version.

## ☁️ Azure Functions Deployment

### Prerequisites
- Azure CLI installed
- Azure subscription
- Valid environment variables

### Deploy
```bash
# Login to Azure
az login

# Deploy to Azure Functions
./deploy_azure.sh
```

### Features
- Serverless scaling
- Built-in monitoring
- Integration with Azure services
- Automatic HTTPS

## 📈 MLflow Tracking

### What's Tracked:
- **Document Uploads**: Count, size, processing time
- **Queries**: Response time, similarity scores, context length
- **Performance**: Response length, context utilization ratio
- **Artifacts**: Full prompts, responses, retrieved documents

### View Results:
1. **Databricks MLflow UI**: `https://your-databricks-host/ml/experiments/[experiment-id]`
2. **MLflow Tracking Notebook**: Run the tracking notebook in Databricks
3. **Local Analytics**: Use the enhanced RAG client for local tracking

## 🔧 Advanced Usage

### Local RAG Client
```bash
python3 enhanced_rag_client.py
```
- Interactive document upload and querying
- MLflow tracking integration
- Real-time performance metrics

### Natural Language Uploader
```bash
python3 natural_language_uploader.py
```
- Describe documents in plain English
- Automatic file discovery and upload
- Smart keyword matching

### Document Loader
```bash
python3 document_loader.py
```
- Load from text files, JSON, or manual entry
- Multiple document formats supported
- Save documents for later use

## 🎯 Use Cases

### Business Applications:
- **Company Policy Q&A** - Query employee handbooks and policies
- **Technical Documentation** - Search API docs and technical guides
- **Knowledge Management** - Organize and query company knowledge base
- **Research Assistant** - Query research papers and documents
- **API Integration** - Embed RAG capabilities into existing applications

### Technical Features:
- **Vector Similarity Search** - Advanced document retrieval
- **Automatic Chunking** - Smart document processing
- **Performance Optimization** - Track and improve query performance
- **Scalable Architecture** - Works with large document collections
- **REST API** - Programmatic access and integration
- **Containerization** - Easy deployment and scaling

## 🛠️ Customization

### Modify Upload Methods:
- Edit `natural_language_uploader.py` for custom document discovery
- Modify `document_loader.py` for different file formats
- Customize chunking parameters in the RAG system

### Adjust Query Parameters:
- Change similarity thresholds
- Modify context length
- Adjust response generation parameters

### MLflow Customization:
- Add custom metrics
- Create custom visualizations
- Export data for external analysis

### API Customization:
- Add new endpoints in `app.py`
- Modify request/response models
- Add authentication and authorization
- Implement rate limiting

## 📈 Performance Monitoring

### Key Metrics:
- **Response Time**: Query processing speed
- **Similarity Scores**: Document relevance
- **Context Utilization**: How much context is used
- **Document Coverage**: Which documents are retrieved

### Optimization Tips:
- Monitor similarity scores for document quality
- Adjust chunk sizes based on document types
- Use MLflow experiments to compare different configurations
- Track query patterns to improve document organization

## 🔒 Security Considerations

### Environment Variables
- Store sensitive credentials in `.env` files
- Never commit credentials to version control
- Use different credentials for development and production

### API Security
- Use HTTPS in production
- Consider adding API key authentication
- Implement rate limiting for production deployments
- Validate all inputs using Pydantic models

### Container Security
- Run containers as non-root users
- Use minimal base images
- Regularly update dependencies
- Scan images for vulnerabilities

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with both CLI and API interfaces
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and questions:
1. Check the API documentation in `API_DOCUMENTATION.md`
2. Review MLflow experiment logs
3. Check the README for common solutions
4. Open an issue in the repository

## 📚 Additional Documentation

- **[API Documentation](API_DOCUMENTATION.md)** - Complete API reference with examples
- **[Docker Guide](Dockerfile)** - Container configuration details
- **[Azure Functions Guide](azure_function/)** - Serverless deployment guide

---

**🎉 Ready to build your own RAG system? Choose your preferred method:**

- **CLI**: `python3 ultimate_rag_workflow.py`
- **Docker**: `./deploy.sh`
- **Azure**: `./deploy_azure.sh`
- **API**: Check out the API documentation!
