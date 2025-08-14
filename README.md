# RAG Workflow - Databricks Integration

A complete end-to-end RAG (Retrieval-Augmented Generation) system that creates Databricks notebooks, enables document upload, provides interactive querying, and tracks everything in MLflow.

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

4. **Run the Ultimate RAG Workflow**
   ```bash
   python3 ultimate_rag_workflow.py
   ```

## 📁 Project Structure

### Core Files
- **`ultimate_rag_workflow.py`** - Main workflow that creates everything
- **`enhanced_workflow_fixed.py`** - Enhanced RAG system with MLflow tracking
- **`enhanced_rag_client.py`** - Local RAG client for interactive querying
- **`natural_language_uploader.py`** - Upload documents using natural language
- **`document_loader.py`** - Load documents from various sources

### Configuration
- **`config.env.example`** - Environment variables template
- **`requirements.txt`** - Python dependencies
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

### 📚 Document Upload Methods
1. **Natural Language** - "Upload all PDF files"
2. **Manual Entry** - Type documents directly
3. **File Loader** - Load from text files, JSON, or manual entry
4. **Skip Upload** - Upload later in Databricks

### 🔍 Querying Capabilities
- **Interactive Session** - Real-time Q&A with documents
- **Single Queries** - One-off questions
- **MLflow Integration** - Every query logged with metrics
- **Performance Analytics** - Response times, similarity scores, context utilization

## 🚀 Usage

### 1. Run the Ultimate Workflow
```bash
python3 ultimate_rag_workflow.py
```

This will:
- ✅ Create Databricks notebooks with RAG system
- ✅ Set up MLflow experiment tracking
- ✅ Provide interactive document upload
- ✅ Enable real-time querying
- ✅ Generate usage guides

### 2. What You'll Get in Databricks

#### 📓 Notebooks Created:
- **`/Shared/Ultimate_RAG_[timestamp]/Ultimate_RAG_System`** - Main RAG system
- **`/Shared/Ultimate_RAG_[timestamp]/MLflow_Tracking`** - Performance dashboard
- **`/Shared/Ultimate_RAG_[timestamp]/Usage_Guide`** - Complete usage instructions

#### 📊 MLflow Experiment:
- **`/Shared/Ultimate_RAG_[timestamp]`** - Tracks all operations
- **Metrics**: Response times, similarity scores, document counts
- **Artifacts**: Prompts, responses, retrieved context
- **Visualizations**: Performance plots and analytics

### 3. Interactive Features

#### Document Upload:
```python
# In Databricks notebook
upload_documents_manual()  # Manual entry
upload_documents_natural_language('Upload all PDF files')  # Natural language
```

#### Querying:
```python
# Single query
result = query_documents('What are the company policies?')
print(result['response'])

# Interactive session
start_interactive_querying()
```

## 📊 MLflow Tracking

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

### Technical Features:
- **Vector Similarity Search** - Advanced document retrieval
- **Automatic Chunking** - Smart document processing
- **Performance Optimization** - Track and improve query performance
- **Scalable Architecture** - Works with large document collections

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

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with the ultimate workflow
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and questions:
1. Check the usage guide in Databricks
2. Review MLflow experiment logs
3. Check the README for common solutions
4. Open an issue in the repository

---

**🎉 Ready to build your own RAG system? Run `python3 ultimate_rag_workflow.py` and start querying your documents!**
