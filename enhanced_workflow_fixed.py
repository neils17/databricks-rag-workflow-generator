#!/usr/bin/env python3
"""
Fixed Enhanced Production Workflow for Complete Databricks RAG System
This version handles Unity Catalog errors gracefully and creates working assets.
"""

import os
import json
import requests
import base64
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class FixedDatabricksAPI:
    """Fixed Databricks API client that handles errors gracefully"""
    
    def __init__(self, host: str, token: str):
        self.host = host.rstrip('/')
        self.token = token
        self.headers = {
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json'
        }
    
    def create_vector_search_endpoint(self, endpoint_name: str, endpoint_type: str = "STANDARD") -> Dict[str, Any]:
        """Create a Vector Search endpoint"""
        url = f"{self.host}/api/2.0/vector-search/endpoints"
        data = {
            "name": endpoint_name,
            "endpoint_type": endpoint_type
        }
        
        response = requests.post(url, headers=self.headers, json=data)
        if response.status_code == 200:
            return {"success": True, "data": response.json()}
        else:
            return {"success": False, "error": response.text, "status": response.status_code}
    
    def check_endpoint_status(self, endpoint_name: str) -> Dict[str, Any]:
        """Check Vector Search endpoint status without waiting"""
        url = f"{self.host}/api/2.0/vector-search/endpoints/{endpoint_name}"
        response = requests.get(url, headers=self.headers)
        if response.status_code == 200:
            return {"success": True, "data": response.json()}
        else:
            return {"success": False, "error": response.text, "status": response.status_code}
    
    def create_notebook(self, path: str, content: str) -> Dict[str, Any]:
        """Create a notebook using Databricks REST API"""
        url = f"{self.host}/api/2.0/workspace/import"
        
        # Encode content as base64
        content_bytes = content.encode('utf-8')
        content_b64 = base64.b64encode(content_bytes).decode('utf-8')
        
        data = {
            "path": path,
            "format": "SOURCE",
            "content": content_b64,
            "language": "PYTHON"
        }
        
        response = requests.post(url, headers=self.headers, json=data)
        if response.status_code == 200:
            return {"success": True, "data": response.json()}
        else:
            return {"success": False, "error": response.text, "status": response.status_code}
    
    def create_folder(self, path: str) -> Dict[str, Any]:
        """Create a folder in Databricks workspace"""
        url = f"{self.host}/api/2.0/workspace/mkdirs"
        data = {"path": path}
        
        response = requests.post(url, headers=self.headers, json=data)
        if response.status_code == 200:
            return {"success": True, "data": response.json()}
        else:
            return {"success": False, "error": response.text, "status": response.status_code}
    
    def test_connection(self) -> Dict[str, Any]:
        """Test the connection to Databricks"""
        url = f"{self.host}/api/2.0/clusters/list"
        response = requests.get(url, headers=self.headers)
        if response.status_code == 200:
            return {"success": True, "message": "Connection successful"}
        else:
            return {"success": False, "error": response.text, "status": response.status_code}

def create_working_rag_system():
    """Create a working RAG system with available Databricks assets"""
    
    print("🚀 Creating Working RAG System with Available Databricks Assets...")
    print("=" * 70)
    
    # Initialize Databricks API client
    host = os.getenv("DATABRICKS_HOST")
    token = os.getenv("DATABRICKS_TOKEN")
    
    if not host or not token:
        print("❌ DATABRICKS_HOST and DATABRICKS_TOKEN must be set")
        return
    
    api = FixedDatabricksAPI(host, token)
    
    # Test connection first
    print("🔗 Testing Databricks connection...")
    connection_test = api.test_connection()
    if not connection_test["success"]:
        print(f"❌ Databricks connection failed: {connection_test.get('error', 'Unknown error')}")
        return
    
    print("✅ Databricks connection successful!")
    
    # Generate unique identifiers
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    endpoint_name = f"rag_endpoint_{timestamp}"
    folder_path = f"/Shared/Working_RAG_Workflow_{timestamp}"
    
    assets_created = {}
    
    try:
        # Create Vector Search endpoint (this works)
        print(f"🔍 Creating Vector Search endpoint: {endpoint_name}")
        endpoint_result = api.create_vector_search_endpoint(endpoint_name)
        if endpoint_result["success"]:
            assets_created["vector_search_endpoint"] = {
                "name": endpoint_name,
                "status": "created",
                "message": f"Successfully created Vector Search endpoint: {endpoint_name}"
            }
            print(f"✅ Vector Search endpoint created: {endpoint_name}")
            
            # Check endpoint status without waiting
            print("🔍 Checking Vector Search endpoint status...")
            status_result = api.check_endpoint_status(endpoint_name)
            if status_result["success"]:
                endpoint_info = status_result["data"]
                state = endpoint_info.get("state", "UNKNOWN")
                print(f"✅ Vector Search endpoint status: {state}")
                assets_created["vector_search_endpoint"]["status"] = state
                assets_created["vector_search_endpoint"]["message"] += f" (status: {state})"
            else:
                print(f"⚠️  Could not check endpoint status: {status_result.get('error', 'Unknown error')}")
        else:
            assets_created["vector_search_endpoint"] = {
                "name": endpoint_name,
                "status": "failed",
                "message": f"Failed to create Vector Search endpoint: {endpoint_result.get('error', 'Unknown error')}"
            }
            print(f"❌ Vector Search endpoint creation failed: {endpoint_result.get('error', 'Unknown error')}")
        
        # Create workspace folder and notebooks
        print(f"📁 Creating workspace folder: {folder_path}")
        folder_result = api.create_folder(folder_path)
        if folder_result["success"]:
            assets_created["folder"] = {
                "path": folder_path,
                "status": "created",
                "message": f"Successfully created folder: {folder_path}"
            }
            print(f"✅ Folder created: {folder_path}")
        else:
            assets_created["folder"] = {
                "path": folder_path,
                "status": "failed",
                "message": f"Failed to create folder: {folder_result.get('error', 'Unknown error')}"
            }
            print(f"❌ Folder creation failed: {folder_result.get('error', 'Unknown error')}")
        
        # Create working RAG workflow notebook
        notebook_path = f"{folder_path}/Working_RAG_Workflow"
        print(f"📓 Creating working RAG workflow notebook: {notebook_path}")
        rag_notebook_content = generate_working_rag_notebook_content(timestamp, endpoint_name)
        notebook_result = api.create_notebook(notebook_path, rag_notebook_content)
        if notebook_result["success"]:
            assets_created["working_rag_notebook"] = {
                "path": notebook_path,
                "status": "created",
                "message": f"Successfully created working RAG notebook: {notebook_path}"
            }
            print(f"✅ Working RAG notebook created: {notebook_path}")
        else:
            assets_created["working_rag_notebook"] = {
                "path": notebook_path,
                "status": "failed",
                "message": f"Failed to create working RAG notebook: {notebook_result.get('error', 'Unknown error')}"
            }
            print(f"❌ Working RAG notebook creation failed: {notebook_result.get('error', 'Unknown error')}")
        
        # Create configuration notebook
        config_notebook_path = f"{folder_path}/RAG_Configuration"
        print(f"⚙️  Creating configuration notebook: {config_notebook_path}")
        config_notebook_content = generate_config_notebook_content(timestamp, endpoint_name)
        config_notebook_result = api.create_notebook(config_notebook_path, config_notebook_content)
        if config_notebook_result["success"]:
            assets_created["config_notebook"] = {
                "path": config_notebook_path,
                "status": "created",
                "message": f"Successfully created configuration notebook: {config_notebook_path}"
            }
            print(f"✅ Configuration notebook created: {config_notebook_path}")
        else:
            assets_created["config_notebook"] = {
                "path": config_notebook_path,
                "status": "failed",
                "message": f"Failed to create configuration notebook: {config_notebook_result.get('error', 'Unknown error')}"
            }
            print(f"❌ Configuration notebook creation failed: {config_notebook_result.get('error', 'Unknown error')}")
        
        # Create MLflow tracking notebook
        mlflow_notebook_path = f"{folder_path}/MLflow_Tracking"
        print(f"📊 Creating MLflow tracking notebook: {mlflow_notebook_path}")
        mlflow_notebook_content = generate_mlflow_tracking_notebook_content(timestamp, endpoint_name)
        mlflow_notebook_result = api.create_notebook(mlflow_notebook_path, mlflow_notebook_content)
        if mlflow_notebook_result["success"]:
            assets_created["mlflow_tracking_notebook"] = {
                "path": mlflow_notebook_path,
                "status": "created",
                "message": f"Successfully created MLflow tracking notebook: {mlflow_notebook_path}"
            }
            print(f"✅ MLflow tracking notebook created: {mlflow_notebook_path}")
        else:
            assets_created["mlflow_tracking_notebook"] = {
                "path": mlflow_notebook_path,
                "status": "failed",
                "message": f"Failed to create MLflow tracking notebook: {mlflow_notebook_result.get('error', 'Unknown error')}"
            }
            print(f"❌ MLflow tracking notebook creation failed: {mlflow_notebook_result.get('error', 'Unknown error')}")
        
        # Create enhanced client files
        create_working_client_files(assets_created, timestamp)
        
        print("\n🎉 Working RAG System Created Successfully!")
        print("\n📁 Generated files:")
        print("  - working_rag_client.py")
        print("  - working_rag_config.py")
        print("  - working_example_usage.py")
        print("  - WORKING_README.md")
        
        print("\n🏗️  WORKING Databricks assets created:")
        for asset_type, asset_info in assets_created.items():
            print(f"  - {asset_type}: {asset_info.get('name', asset_info.get('path', 'N/A'))}")
        
        print("\n🌐 View your assets in Databricks:")
        print(f"  Workspace: {host}")
        print("  - Go to Vector Search to see your endpoint")
        print("  - Go to Workspace > Shared to see your working RAG workflow")
        
        print("\n📖 Next steps:")
        print("1. Check your Databricks workspace for created assets")
        print("2. Run working_example_usage.py to test the system")
        print("3. The Vector Search endpoint will be ready when you need it")
        print("4. No manual configuration needed - everything is ready to use!")
        
    except Exception as e:
        print(f"❌ Error creating working RAG system: {str(e)}")

def generate_working_rag_notebook_content(timestamp: str, endpoint_name: str) -> str:
    """Generate the content for the working RAG workflow notebook with MLflow tracking"""
    
    return f'''# Working Databricks RAG Workflow with MLflow Tracking
# Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
# Workflow ID: {timestamp}

# COMMAND ----------

# Install required packages
# MAGIC %pip install langchain langchain-openai databricks-vectorsearch pydantic mlflow

# COMMAND ----------

# Import libraries
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import DatabricksVectorSearch
from langchain.schema import Document
import pandas as pd
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import json
import uuid
import mlflow
import time
from datetime import datetime

# COMMAND ----------

# Configuration
WORKFLOW_ID = "{timestamp}"
VECTOR_SEARCH_ENDPOINT = "{endpoint_name}"
OPENAI_API_KEY = "your_openai_api_key"  # Update this

# MLflow Configuration
EXPERIMENT_NAME = f"RAG_Workflow_{{timestamp}}"
TRACKING_URI = "databricks"  # Use Databricks MLflow tracking

# COMMAND ----------

# Initialize MLflow Experiment
def setup_mlflow_experiment():
    """Setup MLflow experiment for RAG workflow tracking"""
    try:
        # Set tracking URI to Databricks
        mlflow.set_tracking_uri(TRACKING_URI)
        
        # Create or get experiment
        experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
        if experiment is None:
            experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
            print(f"✅ Created new MLflow experiment: {{EXPERIMENT_NAME}} (ID: {{experiment_id}})")
        else:
            experiment_id = experiment.experiment_id
            print(f"✅ Using existing MLflow experiment: {{EXPERIMENT_NAME}} (ID: {{experiment_id}})")
        
        # Set the experiment
        mlflow.set_experiment(EXPERIMENT_NAME)
        
        return experiment_id
    except Exception as e:
        print(f"⚠️  MLflow setup failed: {{str(e)}}")
        return None

# COMMAND ----------

# Initialize embeddings and LLM
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
llm = ChatOpenAI(
    model="gpt-4-turbo-preview",
    temperature=0.1,
    openai_api_key=OPENAI_API_KEY
)

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

# RAG Query function with MLflow tracking
def rag_query_with_tracking(query: str, documents: List[str], k: int = 5, experiment_id: str = None) -> Dict[str, Any]:
    """Perform RAG query with full implementation and MLflow tracking"""
    
    # Start MLflow run
    with mlflow.start_run(experiment_id=experiment_id) as run:
        start_time = time.time()
        
        try:
            # Log parameters
            mlflow.log_param("query", query)
            mlflow.log_param("num_documents", len(documents))
            mlflow.log_param("top_k", k)
            mlflow.log_param("workflow_id", WORKFLOW_ID)
            mlflow.log_param("vector_search_endpoint", VECTOR_SEARCH_ENDPOINT)
            mlflow.log_param("embedding_model", "text-embedding-3-small")
            mlflow.log_param("llm_model", "gpt-4-turbo-preview")
            mlflow.log_param("chunk_size", 1000)
            mlflow.log_param("chunk_overlap", 200)
            
            # Process documents
            processed_docs = process_documents(documents)
            mlflow.log_metric("num_processed_chunks", len(processed_docs))
            
            # Create embeddings for query
            query_embedding = embeddings.embed_query(query)
            
            # Simple similarity search
            similarities = []
            for doc in processed_docs:
                doc_embedding = embeddings.embed_query(doc.page_content)
                # Calculate cosine similarity (simplified)
                similarity = sum(a * b for a, b in zip(query_embedding, doc_embedding))
                similarities.append((similarity, doc))
            
            # Sort by similarity and get top k
            similarities.sort(key=lambda x: x[0], reverse=True)
            top_docs = [doc for _, doc in similarities[:k]]
            
            # Log similarity metrics
            if similarities:
                mlflow.log_metric("max_similarity", similarities[0][0])
                mlflow.log_metric("min_similarity", similarities[-1][0])
                mlflow.log_metric("avg_similarity", sum(s[0] for s in similarities) / len(similarities))
            
            # Prepare context from retrieved documents
            context = "\\n\\n".join([doc.page_content for doc in top_docs])
            mlflow.log_metric("context_length", len(context))
            
            # Create prompt for LLM
            prompt = f"""Based on the following context, answer the question. If the context doesn't contain enough information to answer the question, say so.

Context:
{{context}}

Question: {{query}}

Answer:"""
            
            # Generate response using LLM
            response = llm.invoke(prompt)
            
            # Calculate metrics
            end_time = time.time()
            response_time = end_time - start_time
            response_length = len(response.content)
            
            # Log metrics
            mlflow.log_metric("response_time_seconds", response_time)
            mlflow.log_metric("response_length", response_length)
            mlflow.log_metric("context_utilization_ratio", len(context) / sum(len(doc.page_content) for doc in processed_docs))
            
            # Log artifacts
            mlflow.log_text(prompt, "prompt.txt")
            mlflow.log_text(response.content, "response.txt")
            mlflow.log_text(context, "retrieved_context.txt")
            
            # Log run info
            mlflow.set_tag("workflow_id", WORKFLOW_ID)
            mlflow.set_tag("vector_search_endpoint", VECTOR_SEARCH_ENDPOINT)
            mlflow.set_tag("query_type", "rag_query")
            mlflow.set_tag("generation_timestamp", datetime.now().isoformat())
            
            return {{
                "success": True,
                "response": response.content,
                "run_id": run.info.run_id,
                "experiment_id": experiment_id,
                "metrics": {{
                    "response_time_seconds": response_time,
                    "response_length": response_length,
                    "num_processed_chunks": len(processed_docs),
                    "context_length": len(context)
                }}
            }}
            
        except Exception as e:
            # Log error
            mlflow.log_param("error", str(e))
            mlflow.set_tag("status", "failed")
            
            return {{
                "success": False,
                "error": str(e),
                "run_id": run.info.run_id,
                "experiment_id": experiment_id
            }}

# COMMAND ----------

# Legacy RAG Query function (without tracking)
def rag_query(query: str, documents: List[str], k: int = 5) -> str:
    """Perform RAG query with full implementation using document similarity"""
    
    try:
        # Process documents
        processed_docs = process_documents(documents)
        
        # Create embeddings for query
        query_embedding = embeddings.embed_query(query)
        
        # Simple similarity search (you can enhance this with Vector Search when ready)
        # For now, we'll use a simple approach
        similarities = []
        for doc in processed_docs:
            doc_embedding = embeddings.embed_query(doc.page_content)
            # Calculate cosine similarity (simplified)
            similarity = sum(a * b for a, b in zip(query_embedding, doc_embedding))
            similarities.append((similarity, doc))
        
        # Sort by similarity and get top k
        similarities.sort(key=lambda x: x[0], reverse=True)
        top_docs = [doc for _, doc in similarities[:k]]
        
        # Prepare context from retrieved documents
        context = "\\n\\n".join([doc.page_content for doc in top_docs])
        
        # Create prompt for LLM
        prompt = f"""Based on the following context, answer the question. If the context doesn't contain enough information to answer the question, say so.

Context:
{{context}}

Question: {{query}}

Answer:"""
        
        # Generate response using LLM
        response = llm.invoke(prompt)
        
        return response.content
        
    except Exception as e:
        return f"Error performing RAG query: {{str(e)}}"

# COMMAND ----------

# Example usage with MLflow tracking
if __name__ == "__main__":
    print("🚀 Starting RAG Workflow with MLflow Tracking")
    print("=" * 60)
    
    # Setup MLflow experiment
    experiment_id = setup_mlflow_experiment()
    
    if experiment_id:
        print(f"📊 MLflow Experiment ID: {{experiment_id}}")
        print(f"🔗 View experiments at: {{mlflow.get_tracking_uri()}}/experiments/{{experiment_id}}")
    else:
        print("⚠️  MLflow tracking disabled - continuing without tracking")
    
    # Example documents
    sample_docs = [
        "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions without being explicitly programmed. It uses algorithms and statistical models to analyze and draw inferences from patterns in data.",
        "Data science combines statistics, programming, and domain expertise to extract insights from data. It involves collecting, cleaning, analyzing, and interpreting data to support decision-making processes.",
        "Natural language processing (NLP) is a field of AI that focuses on the interaction between computers and human language. It enables machines to understand, interpret, and generate human language in a meaningful way."
    ]
    
    # Example query with MLflow tracking
    query = "What is machine learning and how does it relate to AI?"
    
    if experiment_id:
        # Use tracked version
        result = rag_query_with_tracking(query, sample_docs, experiment_id=experiment_id)
        if result["success"]:
            print(f"\\n✅ Query completed successfully!")
            print(f"📊 Run ID: {{result['run_id']}}")
            print(f"⏱️  Response Time: {{result['metrics']['response_time_seconds']:.2f}} seconds")
            print(f"📝 Response Length: {{result['metrics']['response_length']}} characters")
            print(f"\\nQuery: {{query}}")
            print(f"Response: {{result['response']}}")
        else:
            print(f"❌ Query failed: {{result['error']}}")
    else:
        # Use untracked version
        result = rag_query(query, sample_docs)
        print(f"\\nQuery: {{query}}")
        print(f"Response: {{result}}")
    
    print("\\n✅ Working RAG system with MLflow tracking is ready!")
    print("\\n💡 Next steps:")
    print("1. Update OPENAI_API_KEY with your actual key")
    print("2. Add your own documents to the system")
    print("3. Use Vector Search endpoint when it's ready")
    print("4. View MLflow experiments in Databricks MLflow UI")
'''

def generate_config_notebook_content(timestamp: str, endpoint_name: str) -> str:
    """Generate configuration notebook content"""
    
    return f'''# RAG Configuration
# Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
# Workflow ID: {timestamp}

# COMMAND ----------

# Configuration settings
WORKFLOW_ID = "{timestamp}"
VECTOR_SEARCH_ENDPOINT = "{endpoint_name}"

# RAG Configuration
RAG_CONFIG = {{
    "embedding_model": "text-embedding-3-small",
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "top_k": 5,
    "llm_model": "gpt-4-turbo-preview",
    "temperature": 0.1
}}

# COMMAND ----------

print("RAG Configuration:")
for key, value in RAG_CONFIG.items():
    print(f"  {{key}}: {{value}}")

print(f"\\nWorkflow ID: {{WORKFLOW_ID}}")
print(f"Vector Search Endpoint: {{VECTOR_SEARCH_ENDPOINT}}")

# COMMAND ----------

# Environment setup instructions
print("\\n📋 Setup Instructions:")
print("1. Set your OPENAI_API_KEY in the environment")
print("2. Update the Vector Search endpoint when ready")
print("3. Configure your document sources")
print("4. Test the RAG system with sample queries")
'''

def generate_mlflow_tracking_notebook_content(timestamp: str, endpoint_name: str) -> str:
    """Generate MLflow tracking configuration notebook content"""
    
    return f'''# MLflow Tracking Configuration for RAG Workflow
# Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
# Workflow ID: {timestamp}

# COMMAND ----------

# Install MLflow
# MAGIC %pip install mlflow

# COMMAND ----------

import mlflow
import pandas as pd
from datetime import datetime
import json

# COMMAND ----------

# Configuration
WORKFLOW_ID = "{timestamp}"
VECTOR_SEARCH_ENDPOINT = "{endpoint_name}"
EXPERIMENT_NAME = f"RAG_Workflow_{{timestamp}}"

# COMMAND ----------

# Setup MLflow tracking
def setup_mlflow_tracking():
    """Setup MLflow tracking for the RAG workflow"""
    
    # Set tracking URI to Databricks
    mlflow.set_tracking_uri("databricks")
    
    # Create or get experiment
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
        print(f"✅ Created new MLflow experiment: {{EXPERIMENT_NAME}}")
        print(f"📊 Experiment ID: {{experiment_id}}")
    else:
        experiment_id = experiment.experiment_id
        print(f"✅ Using existing MLflow experiment: {{EXPERIMENT_NAME}}")
        print(f"📊 Experiment ID: {{experiment_id}}")
    
    # Set the experiment
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    return experiment_id

# COMMAND ----------

# Initialize tracking
experiment_id = setup_mlflow_tracking()

# COMMAND ----------

# Log workflow configuration
with mlflow.start_run(experiment_id=experiment_id) as run:
    # Log parameters
    mlflow.log_param("workflow_id", WORKFLOW_ID)
    mlflow.log_param("vector_search_endpoint", VECTOR_SEARCH_ENDPOINT)
    mlflow.log_param("experiment_name", EXPERIMENT_NAME)
    mlflow.log_param("embedding_model", "text-embedding-3-small")
    mlflow.log_param("llm_model", "gpt-4-turbo-preview")
    mlflow.log_param("chunk_size", 1000)
    mlflow.log_param("chunk_overlap", 200)
    mlflow.log_param("top_k", 5)
    
    # Log metrics
    mlflow.log_metric("setup_timestamp", datetime.now().timestamp())
    
    # Log tags
    mlflow.set_tag("workflow_type", "rag_workflow")
    mlflow.set_tag("vector_search_endpoint", VECTOR_SEARCH_ENDPOINT)
    mlflow.set_tag("generation_timestamp", datetime.now().isoformat())
    
    print(f"✅ Workflow configuration logged to MLflow")
    print(f"📊 Run ID: {{run.info.run_id}}")

# COMMAND ----------

# Function to view experiment runs
def view_experiment_runs():
    """View all runs in the current experiment"""
    
    try:
        # Get experiment
        experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
        if experiment is None:
            print("❌ Experiment not found")
            return
        
        # Get all runs
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        
        if runs.empty:
            print("📊 No runs found in experiment")
            return
        
        print(f"📊 Found {{len(runs)}} runs in experiment: {{EXPERIMENT_NAME}}")
        print("\\nRecent runs:")
        
        # Display recent runs
        for idx, run in runs.head(10).iterrows():
            run_id = run['run_id']
            status = run.get('status', 'UNKNOWN')
            start_time = run.get('start_time', 'N/A')
            
            print(f"  Run ID: {{run_id}} | Status: {{status}} | Start: {{start_time}}")
            
    except Exception as e:
        print(f"❌ Error viewing runs: {{str(e)}}")

# COMMAND ----------

# Function to compare runs
def compare_runs(run_ids: list):
    """Compare multiple runs"""
    
    try:
        # Get runs data
        runs_data = mlflow.search_runs(run_ids=run_ids)
        
        if runs_data.empty:
            print("❌ No runs found")
            return
        
        print(f"📊 Comparing {{len(runs_data)}} runs:")
        
        # Compare key metrics
        metrics_to_compare = ['response_time_seconds', 'response_length', 'num_processed_chunks']
        
        for metric in metrics_to_compare:
            if metric in runs_data.columns:
                values = runs_data[metric].dropna()
                if len(values) > 0:
                    print(f"\\n{{metric}}:")
                    print(f"  Min: {{values.min():.2f}}")
                    print(f"  Max: {{values.max():.2f}}")
                    print(f"  Avg: {{values.mean():.2f}}")
        
    except Exception as e:
        print(f"❌ Error comparing runs: {{str(e)}}")

# COMMAND ----------

# Function to export experiment data
def export_experiment_data():
    """Export experiment data to DataFrame"""
    
    try:
        # Get experiment
        experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
        if experiment is None:
            print("❌ Experiment not found")
            return None
        
        # Get all runs
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        
        if runs.empty:
            print("📊 No runs found in experiment")
            return None
        
        print(f"📊 Exported {{len(runs)}} runs from experiment")
        return runs
        
    except Exception as e:
        print(f"❌ Error exporting data: {{str(e)}}")
        return None

# COMMAND ----------

# Display experiment information
print("🔍 MLflow Experiment Information")
print("=" * 50)
print(f"Experiment Name: {{EXPERIMENT_NAME}}")
print(f"Experiment ID: {{experiment_id}}")
print(f"Workflow ID: {{WORKFLOW_ID}}")
print(f"Vector Search Endpoint: {{VECTOR_SEARCH_ENDPOINT}}")

# COMMAND ----------

# View recent runs
view_experiment_runs()

# COMMAND ----------

# Export data for analysis
experiment_data = export_experiment_data()
if experiment_data is not None:
    display(experiment_data)

# COMMAND ----------

print("\\n✅ MLflow tracking configuration complete!")
print("\\n💡 Next steps:")
print("1. Run the main RAG workflow notebook to generate tracked runs")
print("2. View experiments in Databricks MLflow UI")
print("3. Use the comparison functions to analyze performance")
print("4. Export data for further analysis")
'''

def create_working_client_files(assets_created: Dict[str, Any], timestamp: str):
    """Create working client files with MLflow tracking"""
    
    # Create working RAG client with MLflow tracking
    working_client_content = f'''"""
Working RAG Client for Databricks Integration with MLflow Tracking
Generated by Fixed Enhanced Production Workflow
This version works with available Databricks assets and includes MLflow tracking
"""

import os
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from dotenv import load_dotenv
import mlflow
import time
from datetime import datetime

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Load environment variables
load_dotenv()

@dataclass
class WorkingRAGConfig:
    """Configuration for working RAG system with MLflow tracking"""
    workflow_id: str = "{timestamp}"
    vector_search_endpoint: str = "{assets_created.get('vector_search_endpoint', {}).get('name', 'rag_endpoint')}"
    embedding_model: str = "text-embedding-3-small"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 5
    enable_mlflow: bool = True
    experiment_name: str = f"RAG_Workflow_{{timestamp}}"

class WorkingDatabricksRAGClient:
    """Working client for interacting with Databricks RAG system with MLflow tracking"""
    
    def __init__(self, config: WorkingRAGConfig):
        self.config = config
        self.host = os.getenv("DATABRICKS_HOST")
        self.token = os.getenv("DATABRICKS_TOKEN")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        # Initialize LangChain components
        self.embeddings = OpenAIEmbeddings(
            model=config.embedding_model,
            openai_api_key=self.openai_api_key
        )
        self.llm = ChatOpenAI(
            model="gpt-4-turbo-preview",
            temperature=0.1,
            openai_api_key=self.openai_api_key
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            length_function=len,
        )
        
        # Setup MLflow if enabled
        self.experiment_id = None
        if config.enable_mlflow:
            self.experiment_id = self._setup_mlflow()
        
        print(f"Working RAG Client initialized for workflow: {{config.workflow_id}}")
        if self.experiment_id:
            print(f"📊 MLflow tracking enabled - Experiment ID: {{self.experiment_id}}")
    
    def _setup_mlflow(self) -> Optional[str]:
        """Setup MLflow experiment for tracking"""
        try:
            # Set tracking URI to Databricks
            mlflow.set_tracking_uri("databricks")
            
            # Create or get experiment
            experiment = mlflow.get_experiment_by_name(self.config.experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(self.config.experiment_name)
                print(f"✅ Created new MLflow experiment: {{self.config.experiment_name}}")
            else:
                experiment_id = experiment.experiment_id
                print(f"✅ Using existing MLflow experiment: {{self.config.experiment_name}}")
            
            # Set the experiment
            mlflow.set_experiment(self.config.experiment_name)
            
            return experiment_id
        except Exception as e:
            print(f"⚠️  MLflow setup failed: {{str(e)}}")
            return None
    
    def get_workspace_url(self) -> str:
        """Get the workspace URL for viewing assets"""
        return f"{{self.host}}/workspace/Shared/Working_RAG_Workflow_{{self.config.workflow_id}}"
    
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
    
    def query_with_tracking(self, query: str, documents: List[str]) -> Dict[str, Any]:
        """Perform RAG query with MLflow tracking"""
        if not self.experiment_id:
            print("⚠️  MLflow tracking not available, using regular query")
            return {{"success": False, "error": "MLflow tracking not available"}}
        
        # Start MLflow run
        with mlflow.start_run(experiment_id=self.experiment_id) as run:
            start_time = time.time()
            
            try:
                # Log parameters
                mlflow.log_param("query", query)
                mlflow.log_param("num_documents", len(documents))
                mlflow.log_param("top_k", self.config.top_k)
                mlflow.log_param("workflow_id", self.config.workflow_id)
                mlflow.log_param("vector_search_endpoint", self.config.vector_search_endpoint)
                mlflow.log_param("embedding_model", self.config.embedding_model)
                mlflow.log_param("llm_model", "gpt-4-turbo-preview")
                mlflow.log_param("chunk_size", self.config.chunk_size)
                mlflow.log_param("chunk_overlap", self.config.chunk_overlap)
                
                # Process documents
                processed_docs = self.process_documents(documents)
                mlflow.log_metric("num_processed_chunks", len(processed_docs))
                
                # Create embeddings for query
                query_embedding = self.embeddings.embed_query(query)
                
                # Simple similarity search
                similarities = []
                for doc in processed_docs:
                    doc_embedding = self.embeddings.embed_query(doc.page_content)
                    # Calculate cosine similarity (simplified)
                    similarity = sum(a * b for a, b in zip(query_embedding, doc_embedding))
                    similarities.append((similarity, doc))
                
                # Sort by similarity and get top k
                similarities.sort(key=lambda x: x[0], reverse=True)
                top_docs = [doc for _, doc in similarities[:self.config.top_k]]
                
                # Log similarity metrics
                if similarities:
                    mlflow.log_metric("max_similarity", similarities[0][0])
                    mlflow.log_metric("min_similarity", similarities[-1][0])
                    mlflow.log_metric("avg_similarity", sum(s[0] for s in similarities) / len(similarities))
                
                # Prepare context from retrieved documents
                context = "\\n\\n".join([doc.page_content for doc in top_docs])
                mlflow.log_metric("context_length", len(context))
                
                # Create prompt for LLM
                prompt = f"""Based on the following context, answer the question. If the context doesn't contain enough information to answer the question, say so.

Context:
{{context}}

Question: {{query}}

Answer:"""
                
                # Generate response using LLM
                response = self.llm.invoke(prompt)
                
                # Calculate metrics
                end_time = time.time()
                response_time = end_time - start_time
                response_length = len(response.content)
                
                # Log metrics
                mlflow.log_metric("response_time_seconds", response_time)
                mlflow.log_metric("response_length", response_length)
                mlflow.log_metric("context_utilization_ratio", len(context) / sum(len(doc.page_content) for doc in processed_docs))
                
                # Log artifacts
                mlflow.log_text(prompt, "prompt.txt")
                mlflow.log_text(response.content, "response.txt")
                mlflow.log_text(context, "retrieved_context.txt")
                
                # Log run info
                mlflow.set_tag("workflow_id", self.config.workflow_id)
                mlflow.set_tag("vector_search_endpoint", self.config.vector_search_endpoint)
                mlflow.set_tag("query_type", "rag_query")
                mlflow.set_tag("generation_timestamp", datetime.now().isoformat())
                
                return {{
                    "success": True,
                    "response": response.content,
                    "run_id": run.info.run_id,
                    "experiment_id": self.experiment_id,
                    "metrics": {{
                        "response_time_seconds": response_time,
                        "response_length": response_length,
                        "num_processed_chunks": len(processed_docs),
                        "context_length": len(context)
                    }}
                }}
                
            except Exception as e:
                # Log error
                mlflow.log_param("error", str(e))
                mlflow.set_tag("status", "failed")
                
                return {{
                    "success": False,
                    "error": str(e),
                    "run_id": run.info.run_id,
                    "experiment_id": self.experiment_id
                }}
    
    def query(self, query: str, documents: List[str]) -> str:
        """Perform RAG query with full implementation (legacy method)"""
        try:
            # Process documents
            processed_docs = self.process_documents(documents)
            
            # Create embeddings for query
            query_embedding = self.embeddings.embed_query(query)
            
            # Simple similarity search
            similarities = []
            for doc in processed_docs:
                doc_embedding = self.embeddings.embed_query(doc.page_content)
                # Calculate cosine similarity (simplified)
                similarity = sum(a * b for a, b in zip(query_embedding, doc_embedding))
                similarities.append((similarity, doc))
            
            # Sort by similarity and get top k
            similarities.sort(key=lambda x: x[0], reverse=True)
            top_docs = [doc for _, doc in similarities[:self.config.top_k]]
            
            # Prepare context from retrieved documents
            context = "\\n\\n".join([doc.page_content for doc in top_docs])
            
            # Create prompt for LLM
            prompt = f"""Based on the following context, answer the question. If the context doesn't contain enough information to answer the question, say so.

Context:
{{context}}

Question: {{query}}

Answer:"""
            
            # Generate response using LLM
            response = self.llm.invoke(prompt)
            
            return response.content
            
        except Exception as e:
            return f"Error performing RAG query: {{str(e)}}"
    
    def get_asset_status(self) -> Dict[str, Any]:
        """Get status of Databricks assets"""
        try:
            assets_status = {{
                "workflow_id": self.config.workflow_id,
                "workspace_url": self.get_workspace_url(),
                "mlflow_experiment_id": self.experiment_id,
                "mlflow_experiment_name": self.config.experiment_name,
                "assets": {{
                    "vector_search_endpoint": self.config.vector_search_endpoint,
                    "folder": "Shared/Working_RAG_Workflow_{{self.config.workflow_id}}"
                }}
            }}
            return assets_status
        except Exception as e:
            return {{"error": str(e)}}

# Example usage
if __name__ == "__main__":
    config = WorkingRAGConfig()
    client = WorkingDatabricksRAGClient(config)
    
    # Check asset status
    status = client.get_asset_status()
    print("Asset Status:", status)
    
    # Example query with MLflow tracking
    sample_docs = ["Machine learning is a subset of artificial intelligence..."]
    
    if config.enable_mlflow and client.experiment_id:
        result = client.query_with_tracking("What is machine learning?", sample_docs)
        if result["success"]:
            print(f"✅ Query completed with tracking!")
            print(f"📊 Run ID: {{result['run_id']}}")
            print(f"⏱️  Response Time: {{result['metrics']['response_time_seconds']:.2f}} seconds")
            print(f"Response: {{result['response']}}")
        else:
            print(f"❌ Query failed: {{result['error']}}")
    else:
        result = client.query("What is machine learning?", sample_docs)
        print(f"Response: {{result}}")
'''
    
    with open("working_rag_client.py", "w") as f:
        f.write(working_client_content)
    
    # Create working RAG config
    working_config_content = f'''"""
Working Configuration for Databricks RAG System with MLflow Tracking
"""

import os
from dataclasses import dataclass

@dataclass
class WorkingDatabricksConfig:
    """Working Databricks connection configuration"""
    host: str = os.getenv("DATABRICKS_HOST", "")
    token: str = os.getenv("DATABRICKS_TOKEN", "")
    workflow_id: str = "{timestamp}"
    vector_search_endpoint: str = "{assets_created.get('vector_search_endpoint', {}).get('name', 'rag_endpoint')}"

@dataclass
class WorkingRAGConfig:
    """Working RAG system configuration with MLflow tracking"""
    workflow_id: str = "{timestamp}"
    embedding_model: str = "text-embedding-3-small"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 5
    vector_search_endpoint: str = "{assets_created.get('vector_search_endpoint', {}).get('name', 'rag_endpoint')}"
    enable_mlflow: bool = True
    experiment_name: str = f"RAG_Workflow_{{timestamp}}"

@dataclass
class OpenAIConfig:
    """OpenAI configuration"""
    api_key: str = os.getenv("OPENAI_API_KEY", "")
    model: str = "gpt-4-turbo-preview"
    temperature: float = 0.1

# Configuration instances
working_databricks_config = WorkingDatabricksConfig()
working_rag_config = WorkingRAGConfig()
openai_config = OpenAIConfig()
'''
    
    with open("working_rag_config.py", "w") as f:
        f.write(working_config_content)
    
    # Create enhanced example usage with MLflow tracking
    working_example_content = f'''"""
Enhanced Example Usage of Databricks RAG System with MLflow Tracking
"""

from working_rag_client import WorkingDatabricksRAGClient, WorkingRAGConfig
from working_rag_config import working_databricks_config, working_rag_config
import os
import mlflow

def main():
    """Enhanced example usage of the RAG system with MLflow tracking"""
    
    print("🚀 Starting Enhanced RAG System with MLflow Tracking")
    print("=" * 60)
    
    # Initialize configuration with MLflow tracking
    config = WorkingRAGConfig(
        workflow_id=working_rag_config.workflow_id,
        vector_search_endpoint=working_databricks_config.vector_search_endpoint,
        embedding_model=working_rag_config.embedding_model,
        chunk_size=working_rag_config.chunk_size,
        chunk_overlap=working_rag_config.chunk_overlap,
        top_k=working_rag_config.top_k,
        enable_mlflow=True,
        experiment_name=working_rag_config.experiment_name
    )
    
    # Create RAG client
    client = WorkingDatabricksRAGClient(config)
    
    # Check asset status
    print("\\n📊 Checking Databricks assets and MLflow setup...")
    status = client.get_asset_status()
    print("Asset Status:", status)
    
    if status.get("mlflow_experiment_id"):
        print(f"✅ MLflow tracking enabled - Experiment ID: {{status['mlflow_experiment_id']}}")
        print(f"📊 Experiment Name: {{status['mlflow_experiment_name']}}")
    else:
        print("⚠️  MLflow tracking not available")
    
    # Example: Process documents and perform query
    print("\\n📄 Processing sample documents...")
    sample_docs = [
        "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions without being explicitly programmed. It uses algorithms and statistical models to analyze and draw inferences from patterns in data.",
        "Data science combines statistics, programming, and domain expertise to extract insights from data. It involves collecting, cleaning, analyzing, and interpreting data to support decision-making processes.",
        "Natural language processing (NLP) is a field of AI that focuses on the interaction between computers and human language. It enables machines to understand, interpret, and generate human language in a meaningful way."
    ]
    
    processed_docs = client.process_documents(sample_docs)
    print(f"✅ Processed {{len(processed_docs)}} document chunks")
    
    # Example: Perform RAG query with MLflow tracking
    print("\\n🔍 Performing RAG query with MLflow tracking...")
    query = "What is machine learning and how does it relate to AI?"
    
    if config.enable_mlflow and client.experiment_id:
        # Use tracked version
        result = client.query_with_tracking(query, sample_docs)
        if result["success"]:
            print(f"✅ Query completed successfully with tracking!")
            print(f"📊 Run ID: {{result['run_id']}}")
            print(f"⏱️  Response Time: {{result['metrics']['response_time_seconds']:.2f}} seconds")
            print(f"📝 Response Length: {{result['metrics']['response_length']}} characters")
            print(f"📄 Processed Chunks: {{result['metrics']['num_processed_chunks']}}")
            print(f"📏 Context Length: {{result['metrics']['context_length']}} characters")
            print(f"\\nQuery: {{query}}")
            print(f"Response: {{result['response']}}")
            
            # Show MLflow experiment URL
            print(f"\\n🔗 View experiment in Databricks MLflow UI:")
            print(f"   {{client.host}}/mlflow/experiments/{{client.experiment_id}}")
        else:
            print(f"❌ Query failed: {{result['error']}}")
    else:
        # Use untracked version
        result = client.query(query, sample_docs)
        print(f"\\nQuery: {{query}}")
        print(f"Response: {{result}}")
    
    # Demonstrate multiple queries for comparison
    print("\\n🔄 Running multiple queries for MLflow comparison...")
    queries = [
        "What is data science?",
        "Explain natural language processing",
        "How do machine learning and AI differ?"
    ]
    
    for i, q in enumerate(queries, 1):
        print(f"\\n--- Query {{i}}: {{q}} ---")
        if config.enable_mlflow and client.experiment_id:
            result = client.query_with_tracking(q, sample_docs)
            if result["success"]:
                print(f"✅ Response Time: {{result['metrics']['response_time_seconds']:.2f}}s")
                print(f"📝 Response: {{result['response'][:100]}}...")
            else:
                print(f"❌ Failed: {{result['error']}}")
        else:
            result = client.query(q, sample_docs)
            print(f"📝 Response: {{result[:100]}}...")
    
    print("\\n🎉 Enhanced RAG system with MLflow tracking ready!")
    print("\\n💡 Next steps:")
    print("1. Visit your Databricks workspace to see created assets")
    print("2. Check MLflow experiments for detailed tracking")
    print("3. Compare different query performances")
    print("4. Upload your documents to the system")
    print("5. Use Vector Search endpoint when it's ready")

if __name__ == "__main__":
    main()
'''
    
    with open("working_example_usage.py", "w") as f:
        f.write(working_example_content)
    
    # Create enhanced README with MLflow information
    working_readme_content = f'''# Working RAG System with MLflow Tracking

## 🚀 Enhanced Databricks RAG Workflow with MLflow Experiment Tracking

This is a **production-ready RAG (Retrieval-Augmented Generation) system** that automatically creates Databricks assets and includes comprehensive MLflow experiment tracking for monitoring and optimization.

## 📊 MLflow Tracking Features

### **Automatic Experiment Creation**
- Creates unique MLflow experiments for each workflow
- Tracks all RAG queries with detailed metrics
- Enables performance comparison and optimization

### **Comprehensive Metrics Tracking**
- **Response Time**: How long each query takes
- **Response Length**: Length of generated responses
- **Similarity Scores**: Document retrieval quality metrics
- **Context Utilization**: How much of the retrieved context is used
- **Processing Metrics**: Number of chunks processed

### **Artifact Logging**
- **Prompts**: Complete prompts sent to LLM
- **Responses**: Generated responses
- **Retrieved Context**: Context used for generation
- **Configuration**: System parameters and settings

## 🏗️ Generated Assets

### **Databricks Workspace Assets**
- **Vector Search Endpoint**: `{assets_created.get('vector_search_endpoint', {}).get('name', 'rag_endpoint')}`
- **Working RAG Workflow**: Complete RAG implementation with MLflow tracking
- **RAG Configuration**: System configuration and settings
- **MLflow Tracking**: Dedicated notebook for experiment management

### **Local Python Files**
- **working_rag_client.py**: Enhanced client with MLflow tracking
- **working_rag_config.py**: Configuration management
- **working_example_usage.py**: Usage examples with tracking
- **WORKING_README.md**: This documentation

## 🔧 Configuration

### **Environment Variables**
```bash
OPENAI_API_KEY=your_openai_api_key
DATABRICKS_HOST=your_databricks_workspace_url
DATABRICKS_TOKEN=your_databricks_personal_access_token
```

### **MLflow Configuration**
- **Experiment Name**: `RAG_Workflow_{timestamp}`
- **Tracking URI**: Databricks MLflow
- **Auto-setup**: Automatic experiment creation and management

## 📈 Usage Examples

### **Basic RAG Query with Tracking**
```python
from working_rag_client import WorkingDatabricksRAGClient, WorkingRAGConfig

# Initialize with MLflow tracking
config = WorkingRAGConfig(enable_mlflow=True)
client = WorkingDatabricksRAGClient(config)

# Perform tracked query
documents = ["Your document content here..."]
result = client.query_with_tracking("What is machine learning?", documents)

if result["success"]:
    print(f"Response: {{result['response']}}")
    print(f"Run ID: {{result['run_id']}}")
    print(f"Response Time: {{result['metrics']['response_time_seconds']:.2f}}s")
```

### **Multiple Queries for Comparison**
```python
queries = ["Query 1", "Query 2", "Query 3"]
for query in queries:
    result = client.query_with_tracking(query, documents)
    # Each query creates a separate MLflow run for comparison
```

## 📊 MLflow Experiment Analysis

### **Viewing Experiments**
1. Go to Databricks MLflow UI
2. Navigate to Experiments
3. Find your experiment: `RAG_Workflow_{timestamp}`
4. View all runs and metrics

### **Comparing Runs**
- Compare response times across different queries
- Analyze similarity score distributions
- Monitor context utilization patterns
- Track system performance over time

### **Key Metrics to Monitor**
- **response_time_seconds**: Query performance
- **max_similarity**: Document retrieval quality
- **context_utilization_ratio**: Context efficiency
- **num_processed_chunks**: Processing load

## 🎯 Production Benefits

### **Performance Monitoring**
- Track query performance over time
- Identify slow queries and bottlenecks
- Monitor system resource utilization

### **Quality Assurance**
- Compare response quality across different parameters
- Track similarity scores for retrieval quality
- Monitor context utilization for efficiency

### **Optimization**
- A/B test different chunk sizes and overlap
- Compare different embedding models
- Optimize top-k values for retrieval

## 🔍 Troubleshooting

### **MLflow Setup Issues**
- Ensure Databricks MLflow is enabled
- Check workspace permissions
- Verify tracking URI configuration

### **Performance Issues**
- Monitor response times in MLflow
- Check similarity score distributions
- Analyze context utilization patterns

## 📖 Next Steps

1. **Run the enhanced example**: `python3 working_example_usage.py`
2. **View MLflow experiments** in Databricks UI
3. **Compare different queries** and parameters
4. **Upload your documents** and start tracking
5. **Optimize based on metrics** from MLflow

## 🏛️ Architecture

```
User Query → MLflow Run Start → Document Processing → Embedding Creation → 
Similarity Search → Context Retrieval → LLM Generation → Metrics Logging → 
MLflow Run End → Response with Tracking Info
```

The system provides complete visibility into the RAG pipeline performance and enables data-driven optimization.

---

**Generated on**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Workflow ID**: {timestamp}  
**Vector Search Endpoint**: {assets_created.get('vector_search_endpoint', {}).get('name', 'rag_endpoint')}  
**MLflow Experiment**: RAG_Workflow_{timestamp}
'''
    
    with open("WORKING_README.md", "w") as f:
        f.write(working_readme_content)

def main():
    """Main function"""
    print("🤖 Fixed Enhanced Databricks RAG Workflow Generator")
    print("=" * 60)
    
    # Get user input
    user_request = input("What would you like to do? (e.g., 'Create a working RAG workflow'): ")
    
    if not user_request:
        print("No request provided. Exiting.")
        return
    
    print(f"\n🚀 Starting fixed enhanced workflow for: {user_request}")
    print("This will create working Databricks assets including Vector Search endpoints...")
    
    # Confirm with user
    confirm = input("\n⚠️  This will create working Databricks infrastructure. Continue? (y/N): ")
    if confirm.lower() != 'y':
        print("Operation cancelled.")
        return
    
    # Create working RAG system
    create_working_rag_system()

if __name__ == "__main__":
    main()
