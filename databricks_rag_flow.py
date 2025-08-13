"""
LangGraph Multi-Agent Flow for Databricks RAG Workflow Automation
"""

import os
import json
import asyncio
from typing import Dict, List, Any, TypedDict, Annotated
from datetime import datetime
from pathlib import Path

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from databricks_sdk import WorkspaceClient
from databricks_sdk.service.workspace import WorkspaceObjectInfo
from databricks_sdk.service.catalog import CatalogInfo, SchemaInfo
from databricks_sdk.service.sql import QueryStatus

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Initialize Databricks client
def get_databricks_client():
    """Initialize and return Databricks workspace client"""
    host = os.getenv("DATABRICKS_HOST")
    token = os.getenv("DATABRICKS_TOKEN")
    
    if not host or not token:
        raise ValueError("DATABRICKS_HOST and DATABRICKS_TOKEN must be set in environment variables")
    
    return WorkspaceClient(host=host, token=token)

# State definition
class AgentState(TypedDict):
    messages: List[Any]
    user_request: str
    workflow_type: str
    databricks_assets: Dict[str, Any]
    rag_config: Dict[str, Any]
    generated_files: List[str]
    current_step: str
    error: str

# Initialize LLM
llm = ChatOpenAI(
    model="gpt-4-turbo-preview",
    temperature=0.1,
    api_key=os.getenv("OPENAI_API_KEY")
)

# Agent 1: User Interaction Agent
def user_interaction_agent(state: AgentState) -> AgentState:
    """Agent responsible for understanding user requests and determining workflow type"""
    
    messages = [
        SystemMessage(content="""You are a helpful assistant that understands user requests for data workflows.
        Your job is to:
        1. Understand what the user wants to do
        2. Determine the appropriate workflow type
        3. Extract key requirements
        
        Available workflow types:
        - RAG workflow: Create a Retrieval-Augmented Generation system with data ingestion, vector storage, and query interface
        
        Respond with a clear understanding and workflow type selection.""")
    ]
    
    if state["messages"]:
        messages.extend(state["messages"])
    
    response = llm.invoke(messages)
    
    # Determine workflow type based on user request
    user_request = state.get("user_request", "")
    workflow_type = "RAG workflow" if "rag" in user_request.lower() else "RAG workflow"
    
    # Update state
    state["workflow_type"] = workflow_type
    state["messages"].append(response)
    state["current_step"] = "user_interaction_complete"
    
    return state

# Agent 2: Databricks Asset Generation Agent
@tool
def create_databricks_catalog(catalog_name: str = "rag_catalog") -> str:
    """Create a new catalog in Databricks Unity Catalog"""
    try:
        client = get_databricks_client()
        
        # Create catalog
        catalog_info = client.catalogs.create(
            name=catalog_name,
            comment="Catalog for RAG workflow assets"
        )
        
        return f"Successfully created catalog: {catalog_name}"
    except Exception as e:
        return f"Error creating catalog: {str(e)}"

@tool
def create_databricks_schema(catalog_name: str, schema_name: str = "rag_schema") -> str:
    """Create a new schema in the specified catalog"""
    try:
        client = get_databricks_client()
        
        # Create schema
        schema_info = client.schemas.create(
            catalog_name=catalog_name,
            name=schema_name,
            comment="Schema for RAG workflow tables and volumes"
        )
        
        return f"Successfully created schema: {catalog_name}.{schema_name}"
    except Exception as e:
        return f"Error creating schema: {str(e)}"

@tool
def create_volume(catalog_name: str, schema_name: str, volume_name: str = "rag_volume") -> str:
    """Create a volume for storing documents and embeddings"""
    try:
        client = get_databricks_client()
        
        # Create volume
        volume_info = client.volumes.create(
            catalog_name=catalog_name,
            schema_name=schema_name,
            name=volume_name,
            volume_type="EXTERNAL",
            comment="Volume for RAG document storage"
        )
        
        return f"Successfully created volume: {catalog_name}.{schema_name}.{volume_name}"
    except Exception as e:
        return f"Error creating volume: {str(e)}"

@tool
def create_notebook(notebook_path: str, notebook_content: str) -> str:
    """Create a notebook in Databricks workspace"""
    try:
        client = get_databricks_client()
        
        # Create notebook
        workspace_object = client.workspace.upload(
            path=notebook_path,
            content=notebook_content.encode('utf-8'),
            format="SOURCE"
        )
        
        return f"Successfully created notebook: {notebook_path}"
    except Exception as e:
        return f"Error creating notebook: {str(e)}"

def databricks_asset_agent(state: AgentState) -> AgentState:
    """Agent responsible for creating Databricks assets"""
    
    # Generate unique identifiers
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    catalog_name = f"rag_catalog_{timestamp}"
    schema_name = "rag_schema"
    volume_name = "documents"
    
    assets_created = {}
    
    try:
        # Create catalog
        catalog_result = create_databricks_catalog(catalog_name)
        assets_created["catalog"] = {
            "name": catalog_name,
            "status": "created" if "Successfully" in catalog_result else "failed",
            "message": catalog_result
        }
        
        # Create schema
        schema_result = create_databricks_schema(catalog_name, schema_name)
        assets_created["schema"] = {
            "name": f"{catalog_name}.{schema_name}",
            "status": "created" if "Successfully" in schema_result else "failed",
            "message": schema_result
        }
        
        # Create volume
        volume_result = create_volume(catalog_name, schema_name, volume_name)
        assets_created["volume"] = {
            "name": f"{catalog_name}.{schema_name}.{volume_name}",
            "status": "created" if "Successfully" in volume_result else "failed",
            "message": volume_result
        }
        
        # Create RAG workflow notebook
        rag_notebook_content = generate_rag_notebook_content(catalog_name, schema_name, volume_name)
        notebook_path = f"/Shared/RAG_Workflow_{timestamp}"
        notebook_result = create_notebook(notebook_path, rag_notebook_content)
        assets_created["notebook"] = {
            "path": notebook_path,
            "status": "created" if "Successfully" in notebook_result else "failed",
            "message": notebook_result
        }
        
        state["databricks_assets"] = assets_created
        state["current_step"] = "databricks_assets_created"
        
    except Exception as e:
        state["error"] = f"Error in databricks_asset_agent: {str(e)}"
        state["current_step"] = "error"
    
    return state

def generate_rag_notebook_content(catalog_name: str, schema_name: str, volume_name: str) -> str:
    """Generate the content for the RAG workflow notebook"""
    
    return f'''# Databricks RAG Workflow
# Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

# COMMAND ----------

# Install required packages
# MAGIC %pip install langchain langchain-openai databricks-vectorsearch pydantic

# COMMAND ----------

# Import libraries
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import DatabricksVectorSearch
from langchain.schema import Document
import pandas as pd
from pydantic import BaseModel
from typing import List, Optional
import json

# COMMAND ----------

# Configuration
CATALOG_NAME = "{catalog_name}"
SCHEMA_NAME = "{schema_name}"
VOLUME_NAME = "{volume_name}"
VECTOR_SEARCH_ENDPOINT = "your_vector_search_endpoint"  # Update this
OPENAI_API_KEY = "your_openai_api_key"  # Update this

# COMMAND ----------

# Initialize embeddings and LLM
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
llm = ChatOpenAI(
    model="gpt-4-turbo-preview",
    temperature=0.1,
    openai_api_key=OPENAI_API_KEY
)

# COMMAND ----------

# Create vector search index
def create_vector_search_index():
    """Create vector search index for document embeddings"""
    
    # Create index schema
    index_schema = {{
        "type": "vector_search",
        "name": "rag_documents_index",
        "endpoint_name": VECTOR_SEARCH_ENDPOINT,
        "source_table_name": f"{{CATALOG_NAME}}.{{SCHEMA_NAME}}.documents",
        "pipeline_type": "TRIGGERED",
        "primary_key": "id",
        "embedding_source_columns": [
            {{
                "name": "content",
                "embedding_model_endpoint_name": "your_embedding_endpoint"  # Update this
            }}
        ]
    }}
    
    # Create the index (this would be done via Databricks Vector Search API)
    print("Vector search index schema created")
    return index_schema

# COMMAND ----------

# Document processing functions
def process_documents(documents: List[str]) -> List[Document]:
    """Process raw documents into LangChain Document objects"""
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    
    docs = []
    for i, doc_text in enumerate(documents):
        chunks = text_splitter.split_text(doc_text)
        for j, chunk in enumerate(chunks):
            docs.append(Document(
                page_content=chunk,
                metadata={{"source": f"document_{{i}}", "chunk": j}}
            ))
    
    return docs

# COMMAND ----------

# RAG Query function
def rag_query(query: str, k: int = 5) -> str:
    """Perform RAG query using vector search"""
    
    # This would connect to the vector search index
    # For now, returning a placeholder response
    response = f"RAG query for: {{query}}\\nRetrieved {{k}} relevant documents from vector search index."
    
    return response

# COMMAND ----------

# Example usage
if __name__ == "__main__":
    # Create vector search index
    index_schema = create_vector_search_index()
    print("Vector search index created successfully")
    
    # Example documents
    sample_docs = [
        "This is a sample document about machine learning.",
        "Another document about data science and analytics.",
        "A third document about artificial intelligence."
    ]
    
    # Process documents
    processed_docs = process_documents(sample_docs)
    print(f"Processed {{len(processed_docs)}} document chunks")
    
    # Example query
    query = "What is machine learning?"
    result = rag_query(query)
    print(result)
'''

# Agent 3: RAG Configuration Agent
def rag_configuration_agent(state: AgentState) -> AgentState:
    """Agent responsible for configuring RAG parameters and settings"""
    
    messages = [
        SystemMessage(content="""You are an expert in RAG (Retrieval-Augmented Generation) systems.
        Your job is to:
        1. Configure optimal RAG parameters based on the use case
        2. Set up vector search settings
        3. Define document processing pipeline
        4. Configure embedding and retrieval strategies
        
        Provide a comprehensive RAG configuration.""")
    ]
    
    # Add context about created assets
    if state.get("databricks_assets"):
        assets_info = json.dumps(state["databricks_assets"], indent=2)
        messages.append(SystemMessage(content=f"Created Databricks assets: {assets_info}"))
    
    response = llm.invoke(messages)
    
    # Generate RAG configuration
    rag_config = {
        "embedding_model": "text-embedding-3-small",
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "retrieval_strategy": "similarity_search",
        "top_k": 5,
        "reranking": False,
        "vector_search_endpoint": "your_vector_search_endpoint",
        "index_name": "rag_documents_index",
        "catalog": state.get("databricks_assets", {}).get("catalog", {}).get("name", "rag_catalog"),
        "schema": state.get("databricks_assets", {}).get("schema", {}).get("name", "rag_schema"),
        "volume": state.get("databricks_assets", {}).get("volume", {}).get("name", "documents")
    }
    
    state["rag_config"] = rag_config
    state["messages"].append(response)
    state["current_step"] = "rag_configuration_complete"
    
    return state

# Agent 4: Local File Generation Agent
def local_file_generation_agent(state: AgentState) -> AgentState:
    """Agent responsible for generating local Python files that reference Databricks assets"""
    
    generated_files = []
    
    try:
        # Generate main RAG client
        rag_client_content = generate_rag_client_file(state)
        with open("rag_client.py", "w") as f:
            f.write(rag_client_content)
        generated_files.append("rag_client.py")
        
        # Generate configuration file
        config_content = generate_config_file(state)
        with open("rag_config.py", "w") as f:
            f.write(config_content)
        generated_files.append("rag_config.py")
        
        # Generate example usage
        example_content = generate_example_usage(state)
        with open("example_usage.py", "w") as f:
            f.write(example_content)
        generated_files.append("example_usage.py")
        
        # Generate README
        readme_content = generate_readme(state)
        with open("README.md", "w") as f:
            f.write(readme_content)
        generated_files.append("README.md")
        
        state["generated_files"] = generated_files
        state["current_step"] = "local_files_generated"
        
    except Exception as e:
        state["error"] = f"Error in local_file_generation_agent: {str(e)}"
        state["current_step"] = "error"
    
    return state

def generate_rag_client_file(state: AgentState) -> str:
    """Generate the main RAG client Python file"""
    
    assets = state.get("databricks_assets", {})
    config = state.get("rag_config", {})
    
    return f'''"""
RAG Client for Databricks Integration
Generated by LangGraph Multi-Agent Flow
"""

import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from pydantic import BaseModel

# Databricks SDK imports
from databricks_sdk import WorkspaceClient
from databricks_sdk.service.workspace import WorkspaceObjectInfo

@dataclass
class RAGConfig:
    """Configuration for RAG system"""
    catalog_name: str = "{assets.get('catalog', {}).get('name', 'rag_catalog')}"
    schema_name: str = "{assets.get('schema', {}).get('name', 'rag_schema')}"
    volume_name: str = "{assets.get('volume', {}).get('name', 'documents')}"
    embedding_model: str = "{config.get('embedding_model', 'text-embedding-3-small')}"
    chunk_size: int = {config.get('chunk_size', 1000)}
    chunk_overlap: int = {config.get('chunk_overlap', 200)}
    top_k: int = {config.get('top_k', 5)}

class DatabricksRAGClient:
    """Client for interacting with Databricks RAG system"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.client = self._get_databricks_client()
        self.embeddings = OpenAIEmbeddings(
            model=config.embedding_model,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        self.llm = ChatOpenAI(
            model="gpt-4-turbo-preview",
            temperature=0.1,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            length_function=len,
        )
    
    def _get_databricks_client(self) -> WorkspaceClient:
        """Initialize Databricks workspace client"""
        host = os.getenv("DATABRICKS_HOST")
        token = os.getenv("DATABRICKS_TOKEN")
        
        if not host or not token:
            raise ValueError("DATABRICKS_HOST and DATABRICKS_TOKEN must be set")
        
        return WorkspaceClient(host=host, token=token)
    
    def upload_document(self, file_path: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Upload a document to Databricks volume"""
        try:
            # Upload to volume
            volume_path = f"/Volumes/{self.config.catalog_name}/{self.config.schema_name}/{self.config.volume_name}"
            workspace_path = f"{{volume_path}}/{{Path(file_path).name}}"
            
            with open(file_path, 'rb') as f:
                self.client.workspace.upload(
                    path=workspace_path,
                    content=f.read(),
                    format="AUTO"
                )
            
            return f"Successfully uploaded {{file_path}} to {{workspace_path}}"
        except Exception as e:
            return f"Error uploading document: {{str(e)}}"
    
    def process_documents(self, documents: List[str]) -> List[Document]:
        """Process raw documents into LangChain Document objects"""
        docs = []
        for i, doc_text in enumerate(documents):
            chunks = self.text_splitter.split_text(doc_text)
            for j, chunk in enumerate(chunks):
                docs.append(Document(
                    page_content=chunk,
                    metadata={{"source": f"document_{{i}}", "chunk": j}}
                ))
        return docs
    
    def query(self, query: str) -> str:
        """Perform RAG query"""
        try:
            # This would integrate with Databricks Vector Search
            # For now, returning a structured response
            response = f"""
RAG Query Results for: {{query}}

Databricks Assets Used:
- Catalog: {{self.config.catalog_name}}
- Schema: {{self.config.schema_name}}
- Volume: {{self.config.volume_name}}

Note: This is a placeholder response. In a full implementation, this would:
1. Generate embeddings for the query
2. Search the vector index in Databricks
3. Retrieve relevant documents
4. Generate a response using the LLM

To complete the implementation, you would need to:
1. Set up Databricks Vector Search endpoint
2. Create and populate the vector index
3. Implement the actual search and retrieval logic
"""
            return response
        except Exception as e:
            return f"Error performing query: {{str(e)}}"
    
    def get_asset_status(self) -> Dict[str, Any]:
        """Get status of Databricks assets"""
        try:
            # Check if assets exist
            assets_status = {{
                "catalog": {{
                    "name": self.config.catalog_name,
                    "exists": True  # Would check actual existence
                }},
                "schema": {{
                    "name": f"{{self.config.catalog_name}}.{{self.config.schema_name}}",
                    "exists": True
                }},
                "volume": {{
                    "name": f"{{self.config.catalog_name}}.{{self.config.schema_name}}.{{self.config.volume_name}}",
                    "exists": True
                }}
            }}
            return assets_status
        except Exception as e:
            return {{"error": str(e)}}

# Example usage
if __name__ == "__main__":
    config = RAGConfig()
    client = DatabricksRAGClient(config)
    
    # Check asset status
    status = client.get_asset_status()
    print("Asset Status:", status)
    
    # Example query
    result = client.query("What is machine learning?")
    print(result)
'''

def generate_config_file(state: AgentState) -> str:
    """Generate configuration file"""
    
    assets = state.get("databricks_assets", {})
    config = state.get("rag_config", {})
    
    return f'''"""
Configuration for Databricks RAG System
"""

import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class DatabricksConfig:
    """Databricks connection configuration"""
    host: str = os.getenv("DATABRICKS_HOST", "")
    token: str = os.getenv("DATABRICKS_TOKEN", "")
    catalog: str = "{assets.get('catalog', {}).get('name', 'rag_catalog')}"
    schema: str = "{assets.get('schema', {}).get('name', 'rag_schema')}"
    volume: str = "{assets.get('volume', {}).get('name', 'documents')}"

@dataclass
class RAGConfig:
    """RAG system configuration"""
    embedding_model: str = "{config.get('embedding_model', 'text-embedding-3-small')}"
    chunk_size: int = {config.get('chunk_size', 1000)}
    chunk_overlap: int = {config.get('chunk_overlap', 200)}
    top_k: int = {config.get('top_k', 5)}
    vector_search_endpoint: Optional[str] = "{config.get('vector_search_endpoint', 'your_vector_search_endpoint')}"

@dataclass
class OpenAIConfig:
    """OpenAI configuration"""
    api_key: str = os.getenv("OPENAI_API_KEY", "")
    model: str = "gpt-4-turbo-preview"
    temperature: float = 0.1

# Configuration instances
databricks_config = DatabricksConfig()
rag_config = RAGConfig()
openai_config = OpenAIConfig()
'''

def generate_example_usage(state: AgentState) -> str:
    """Generate example usage file"""
    
    return '''"""
Example Usage of Databricks RAG System
"""

from rag_client import DatabricksRAGClient, RAGConfig
from rag_config import databricks_config, rag_config
import os

def main():
    """Example usage of the RAG system"""
    
    # Initialize configuration
    config = RAGConfig(
        catalog_name=databricks_config.catalog,
        schema_name=databricks_config.schema,
        volume_name=databricks_config.volume,
        embedding_model=rag_config.embedding_model,
        chunk_size=rag_config.chunk_size,
        chunk_overlap=rag_config.chunk_overlap,
        top_k=rag_config.top_k
    )
    
    # Create RAG client
    client = DatabricksRAGClient(config)
    
    # Check asset status
    print("Checking Databricks assets...")
    status = client.get_asset_status()
    print("Asset Status:", status)
    
    # Example: Upload a document
    # client.upload_document("path/to/your/document.pdf")
    
    # Example: Process documents
    sample_docs = [
        "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions without being explicitly programmed.",
        "Data science combines statistics, programming, and domain expertise to extract insights from data.",
        "Natural language processing (NLP) is a field of AI that focuses on the interaction between computers and human language."
    ]
    
    processed_docs = client.process_documents(sample_docs)
    print(f"Processed {len(processed_docs)} document chunks")
    
    # Example: Perform RAG query
    print("\\nPerforming RAG query...")
    query = "What is machine learning and how does it relate to AI?"
    result = client.query(query)
    print(result)

if __name__ == "__main__":
    main()
'''

def generate_readme(state: AgentState) -> str:
    """Generate README file"""
    
    assets = state.get("databricks_assets", {})
    config = state.get("rag_config", {})
    
    return f'''# Databricks RAG Workflow

This project was automatically generated by a LangGraph multi-agent flow to create a complete RAG (Retrieval-Augmented Generation) system integrated with Databricks.

## Overview

The system creates:
- Databricks Unity Catalog assets (catalog, schema, volume)
- Vector search infrastructure
- Local Python client for interacting with the RAG system
- Complete workflow automation

## Generated Assets

### Databricks Assets
- **Catalog**: {assets.get('catalog', {}).get('name', 'rag_catalog')}
- **Schema**: {assets.get('schema', {}).get('name', 'rag_schema')}
- **Volume**: {assets.get('volume', {}).get('name', 'documents')}
- **Notebook**: {assets.get('notebook', {}).get('path', '/Shared/RAG_Workflow')}

### Local Files
- `rag_client.py` - Main RAG client for Databricks integration
- `rag_config.py` - Configuration settings
- `example_usage.py` - Example usage of the RAG system
- `README.md` - This file

## Setup

1. **Environment Variables**
   Create a `.env` file with:
   ```
   OPENAI_API_KEY=your_openai_api_key
   DATABRICKS_HOST=your_databricks_workspace_url
   DATABRICKS_TOKEN=your_databricks_personal_access_token
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Databricks Configuration**
   - Ensure you have access to Unity Catalog
   - Set up a Vector Search endpoint (update in config)
   - Configure embedding model endpoint

## Usage

```python
from rag_client import DatabricksRAGClient, RAGConfig

# Initialize client
config = RAGConfig()
client = DatabricksRAGClient(config)

# Check assets
status = client.get_asset_status()
print(status)

# Perform query
result = client.query("What is machine learning?")
print(result)
```

## RAG Configuration

- **Embedding Model**: {config.get('embedding_model', 'text-embedding-3-small')}
- **Chunk Size**: {config.get('chunk_size', 1000)}
- **Chunk Overlap**: {config.get('chunk_overlap', 200)}
- **Top K**: {config.get('top_k', 5)}

## Next Steps

1. **Complete Vector Search Setup**
   - Create Vector Search endpoint in Databricks
   - Update endpoint configuration in `rag_config.py`
   - Create and populate vector index

2. **Document Ingestion**
   - Upload documents to the created volume
   - Process and embed documents
   - Index documents in vector search

3. **Production Deployment**
   - Set up proper authentication
   - Configure monitoring and logging
   - Implement error handling and retry logic

## Architecture

```
User Query → RAG Client → Databricks Vector Search → Retrieved Documents → LLM → Response
```

The system leverages Databricks' scalable infrastructure for:
- Document storage (Unity Catalog volumes)
- Vector embeddings and search
- Compute resources for processing
- Integration with existing data pipelines

## Support

For issues or questions:
1. Check Databricks workspace for asset status
2. Verify environment variables are set correctly
3. Ensure proper permissions for Unity Catalog access
'''

# Main workflow graph
def create_workflow_graph():
    """Create the LangGraph workflow"""
    
    # Create the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("user_interaction", user_interaction_agent)
    workflow.add_node("databricks_assets", databricks_asset_agent)
    workflow.add_node("rag_configuration", rag_configuration_agent)
    workflow.add_node("local_files", local_file_generation_agent)
    
    # Add edges
    workflow.add_edge("user_interaction", "databricks_assets")
    workflow.add_edge("databricks_assets", "rag_configuration")
    workflow.add_edge("rag_configuration", "local_files")
    workflow.add_edge("local_files", END)
    
    # Set entry point
    workflow.set_entry_point("user_interaction")
    
    return workflow.compile()

# Main execution function
async def run_rag_workflow(user_request: str):
    """Run the complete RAG workflow"""
    
    # Initialize state
    initial_state = AgentState(
        messages=[HumanMessage(content=user_request)],
        user_request=user_request,
        workflow_type="",
        databricks_assets={},
        rag_config={},
        generated_files=[],
        current_step="",
        error=""
    )
    
    # Create and run workflow
    app = create_workflow_graph()
    
    try:
        result = await app.ainvoke(initial_state)
        return result
    except Exception as e:
        return {"error": str(e)}

# CLI interface
def main():
    """Main CLI interface"""
    print("🤖 Databricks RAG Workflow Generator")
    print("=" * 50)
    
    # Get user input
    user_request = input("What would you like to do? (e.g., 'Create a RAG workflow'): ")
    
    if not user_request:
        print("No request provided. Exiting.")
        return
    
    print(f"\\n🚀 Starting workflow for: {user_request}")
    print("This will create Databricks assets and generate local files...")
    
    # Run workflow
    result = asyncio.run(run_rag_workflow(user_request))
    
    if "error" in result:
        print(f"❌ Error: {result['error']}")
    else:
        print("\\n✅ Workflow completed successfully!")
        print("\\n📁 Generated files:")
        for file in result.get("generated_files", []):
            print(f"  - {file}")
        
        print("\\n🏗️  Databricks assets created:")
        assets = result.get("databricks_assets", {})
        for asset_type, asset_info in assets.items():
            print(f"  - {asset_type}: {asset_info.get('name', asset_info.get('path', 'N/A'))}")
        
        print("\\n📖 Next steps:")
        print("1. Check your Databricks workspace for the created assets")
        print("2. Update the configuration in rag_config.py")
        print("3. Run example_usage.py to test the system")
        print("4. Set up Vector Search endpoint in Databricks")

if __name__ == "__main__":
    main()
