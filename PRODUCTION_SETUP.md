# Production Setup Guide

This guide will help you run the LangGraph Multi-Agent Databricks RAG Workflow in production mode to create real Databricks assets.

## 🚀 Quick Start

### Step 1: Create Environment File

Create a `.env` file in the project root with your credentials:

```bash
# OpenAI API Configuration
OPENAI_API_KEY=sk-your-actual-openai-api-key

# Databricks Configuration
DATABRICKS_HOST=https://your-workspace.cloud.databricks.com
DATABRICKS_TOKEN=dapi-your-actual-databricks-token

# Optional: Databricks Catalog and Schema (will be auto-generated if not specified)
DATABRICKS_CATALOG=your_catalog_name
DATABRICKS_SCHEMA=your_schema_name
```

### Step 2: Run Production Setup

```bash
python3 production_setup.py
```

This script will:
- ✅ Validate your environment variables
- ✅ Install required dependencies
- ✅ Test Databricks connection
- ✅ Test OpenAI connection
- ✅ Run the production workflow

### Step 3: View Created Assets

After successful execution, you can view your assets in Databricks:

1. **Unity Catalog Assets**:
   - Go to `Data > Unity Catalog`
   - Look for your created catalog and schema

2. **Workspace Notebook**:
   - Go to `Workspace > Shared`
   - Find your RAG workflow notebook

3. **Document Volume**:
   - Go to `Data > Volumes`
   - Navigate to your catalog/schema to see the documents volume

## 🔧 Required Credentials

### OpenAI API Key
- Get from: https://platform.openai.com/api-keys
- Format: `sk-...`
- Required for: LLM interactions and embeddings

### Databricks Credentials
- **Workspace URL**: Your Databricks workspace URL
  - Format: `https://your-workspace.cloud.databricks.com`
  - Get from: Your Databricks workspace URL bar

- **Personal Access Token**:
  - Get from: Databricks workspace → User Settings → Developer → Personal Access Tokens
  - Format: `dapi...`
  - Required permissions: Unity Catalog admin, workspace access

## 📋 Prerequisites

### Databricks Requirements
- ✅ Unity Catalog enabled workspace
- ✅ Unity Catalog admin privileges
- ✅ Personal Access Token with appropriate permissions
- ✅ Workspace access

### Local Requirements
- ✅ Python 3.8+
- ✅ pip package manager
- ✅ Internet connection

## 🛠️ Manual Setup (Alternative)

If you prefer to set up manually:

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Environment Variables
```bash
export OPENAI_API_KEY="sk-your-actual-openai-api-key"
export DATABRICKS_HOST="https://your-workspace.cloud.databricks.com"
export DATABRICKS_TOKEN="dapi-your-actual-databricks-token"
```

### 3. Run Production Workflow
```bash
python3 databricks_rag_flow.py
```

## 🔍 What Will Be Created

### Databricks Assets
- **Catalog**: `rag_catalog_YYYYMMDD_HHMMSS`
- **Schema**: `rag_schema`
- **Volume**: `documents` (for document storage)
- **Notebook**: `/Shared/RAG_Workflow_YYYYMMDD_HHMMSS`

### Local Files
- `rag_client.py` - Production RAG client
- `rag_config.py` - Configuration with real credentials
- `example_usage.py` - Example usage with real Databricks
- `README.md` - Updated documentation

## 🎯 Next Steps After Creation

1. **Set up Vector Search**:
   - Go to Databricks → Vector Search
   - Create a new endpoint
   - Update the endpoint in `rag_config.py`

2. **Upload Documents**:
   - Use the generated `rag_client.py`
   - Upload documents to the created volume
   - Process and embed documents

3. **Test the System**:
   ```bash
   python3 example_usage.py
   ```

## 🐛 Troubleshooting

### Common Issues

#### "Insufficient privileges to create catalog"
- **Solution**: Ensure you have Unity Catalog admin privileges
- **Check**: Go to Databricks → Admin Console → Unity Catalog

#### "Invalid token"
- **Solution**: Generate a new Personal Access Token
- **Check**: Token format should be `dapi...`

#### "Connection failed"
- **Solution**: Verify your workspace URL
- **Check**: URL should be `https://your-workspace.cloud.databricks.com`

#### "OpenAI API key invalid"
- **Solution**: Check your OpenAI API key
- **Check**: Key should start with `sk-`

### Debug Mode
Enable detailed logging:
```bash
export DEBUG=1
python3 production_setup.py
```

## 🔒 Security Notes

- ✅ Never commit `.env` files to version control
- ✅ Use Personal Access Tokens, not username/password
- ✅ Rotate tokens regularly
- ✅ Use workspace-specific tokens when possible
- ✅ Monitor API usage and costs

## 📞 Support

If you encounter issues:

1. Check the error messages in the console
2. Verify your credentials are correct
3. Ensure you have the required Databricks permissions
4. Check the troubleshooting section above

---

**Ready to create real Databricks assets? Run `python3 production_setup.py` and follow the prompts!**
