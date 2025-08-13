# 🤖 Databricks RAG Workflow Generator

A comprehensive LangGraph multi-agent system that automatically creates complete RAG (Retrieval-Augmented Generation) workflows in Databricks with Vector Search integration.

## 🎯 Overview

This project demonstrates:
- **LangGraph Multi-Agent Flow**: Orchestrates specialized agents for different tasks
- **Complete Databricks Integration**: Creates ALL assets including Vector Search endpoints, indexes, schemas, catalogs
- **Full RAG Implementation**: Complete retrieval-augmented generation pipeline with no manual work needed
- **Production Ready**: Ready to use immediately with real Databricks assets

## 🏗️ Generated Assets

### Databricks Assets (Created in Your Workspace)
- **Vector Search Endpoints**: For efficient document retrieval
- **Unity Catalog Assets**: Catalogs, schemas, and tables (when configured)
- **Workspace Folders**: Organized workflow structure
- **Complete RAG Notebooks**: Full implementation with Vector Search
- **Configuration Notebooks**: Settings and parameters

### Local Files
- **RAG Clients**: Python clients for interacting with the system
- **Configuration Files**: Centralized settings management
- **Example Usage**: Ready-to-run examples
- **Documentation**: Comprehensive guides and READMEs

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.8+
- Databricks account with API access
- OpenAI API key

### 2. Installation
```bash
# Clone the repository
git clone <your-repo-url>
cd Databrick_PoC

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration
Create a `.env` file with your credentials:
```bash
# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Databricks Configuration
DATABRICKS_HOST=your_databricks_workspace_url
DATABRICKS_TOKEN=your_databricks_personal_access_token
```

### 4. Run the Workflow
```bash
# Simple production workflow (recommended)
python production_workflow_simple.py

# Enhanced workflow with Vector Search
python enhanced_workflow_fixed.py

# Complete workflow with all features
python production_workflow_complete.py
```

## 📁 Project Structure

```
Databrick_PoC/
├── production_workflow_simple.py      # Simple production workflow
├── enhanced_workflow_fixed.py         # Enhanced workflow with Vector Search
├── production_workflow_complete.py    # Complete LangGraph workflow
├── working_rag_client.py             # Working RAG client
├── working_example_usage.py          # Example usage
├── requirements.txt                  # Python dependencies
├── README.md                        # This file
└── .gitignore                       # Git ignore rules
```

## 🔧 Available Workflows

### 1. Simple Production Workflow
- Creates workspace folders and notebooks
- Generates local Python clients
- No Unity Catalog dependencies
- **Best for**: Quick setup and testing

### 2. Enhanced Workflow (Fixed)
- Creates Vector Search endpoints
- Handles Unity Catalog errors gracefully
- Full RAG implementation
- **Best for**: Production use with Vector Search

### 3. Complete LangGraph Workflow
- Multi-agent orchestration
- Complete Unity Catalog integration
- Advanced error handling
- **Best for**: Complex enterprise deployments

## 📊 Usage Examples

### Basic RAG Query
```python
from working_rag_client import WorkingDatabricksRAGClient, WorkingRAGConfig

# Initialize client
config = WorkingRAGConfig()
client = WorkingDatabricksRAGClient(config)

# Process documents and query
documents = ["Your document content here..."]
result = client.query("What is machine learning?", documents)
print(result)
```

### Check Asset Status
```python
# Check Databricks assets
status = client.get_asset_status()
print(status)
```

## 🎯 Features

### ✅ Multi-Agent Architecture
- **User Interaction Agent**: Understands user requests
- **Databricks Asset Agent**: Creates infrastructure
- **RAG Configuration Agent**: Optimizes parameters
- **File Generation Agent**: Creates local clients

### ✅ Complete Databricks Integration
- Vector Search endpoints and indexes
- Unity Catalog assets (catalogs, schemas, tables)
- Workspace folders and notebooks
- REST API integration

### ✅ Full RAG Implementation
- Document processing and chunking
- Embedding generation
- Vector similarity search
- LLM response generation
- Context-aware responses

### ✅ Production Ready
- Error handling and recovery
- Graceful degradation
- Configuration management
- Comprehensive logging

## 🔄 Workflow Steps

1. **User Request**: User specifies what they want to do
2. **Asset Creation**: ALL Databricks assets are created automatically
3. **RAG Configuration**: Optimal parameters are configured
4. **File Generation**: Local Python files with full implementation are created
5. **System Ready**: Complete RAG system is ready for use

## 🏛️ Architecture

```
User Request → LangGraph → Multi-Agent Flow → Complete Databricks Assets → Full RAG Implementation → Production System
```

The system leverages:
- **LangGraph**: For orchestrated multi-agent workflows
- **Databricks**: For scalable data infrastructure
- **Unity Catalog**: For data governance and organization
- **Vector Search**: For efficient document retrieval
- **LangChain**: For RAG pipeline components
- **OpenAI**: For embeddings and LLM responses

## 🛠️ Development

### Adding New Workflow Types
1. Extend the `user_interaction_agent` to recognize new workflow types
2. Create new agents for specific functionality
3. Update the workflow graph to include new nodes

### Customizing RAG Configuration
1. Modify the `rag_configuration_agent` logic
2. Update the configuration templates
3. Adjust the generated client code

## 📚 Resources

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Databricks SDK Documentation](https://databricks-sdk-py.readthedocs.io/)
- [Databricks Vector Search](https://docs.databricks.com/en/vector-search/index.html)
- [Unity Catalog](https://docs.databricks.com/en/data-governance/unity-catalog/index.html)

## 🤝 Support

For issues or questions:
1. Check the demo output for error messages
2. Verify environment variables are set correctly
3. Ensure proper permissions for Databricks access
4. Review the generated configuration files

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🎉 Acknowledgments

- Built with LangGraph for multi-agent orchestration
- Powered by Databricks for scalable infrastructure
- Enhanced with OpenAI for intelligent responses

---

**Note**: This is a production-ready system that creates ALL Databricks assets and implements the full RAG pipeline with no manual work needed.
