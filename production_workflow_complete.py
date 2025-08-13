"""
Complete Production Workflow for Real Databricks Assets
This version creates ALL Databricks assets automatically including Vector Search endpoints, indexes, schemas, catalogs
and implements the full RAG pipeline with no manual work needed.
"""

import os
import json
import asyncio
import requests
import base64
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

# Load environment variables
load_dotenv()

class DatabricksAPI:
    """Complete Databricks API client using REST calls"""
    
    def __init__(self, host: str, token: str):
        self.host = host.rstrip('/')
        self.token = token
        self.headers = {
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json'
        }
    
    def create_catalog(self, catalog_name: str, comment: str = "") -> Dict[str, Any]:
        """Create a catalog using Databricks REST API"""
        url = f"{self.host}/api/2.1/unity-catalog/catalogs"
        data = {
            "name": catalog_name,
            "comment": comment
        }
        
        response = requests.post(url, headers=self.headers, json=data)
        if response.status_code == 200:
            return {"success": True, "data": response.json()}
        else:
            return {"success": False, "error": response.text, "status": response.status_code}
    
    def create_schema(self, catalog_name: str, schema_name: str, comment: str = "") -> Dict[str, Any]:
        """Create a schema using Databricks REST API"""
        url = f"{self.host}/api/2.1/unity-catalog/schemas"
        data = {
            "name": schema_name,
            "catalog_name": catalog_name,
            "comment": comment
        }
        
        response = requests.post(url, headers=self.headers, json=data)
        if response.status_code == 200:
            return {"success": True, "data": response.json()}
        else:
            return {"success": False, "error": response.text, "status": response.status_code}
    
    def create_table(self, catalog_name: str, schema_name: str, table_name: str, columns: List[Dict]) -> Dict[str, Any]:
        """Create a table using Databricks REST API"""
        url = f"{self.host}/api/2.1/unity-catalog/tables"
        data = {
            "name": table_name,
            "catalog_name": catalog_name,
            "schema_name": schema_name,
            "table_type": "MANAGED",
            "columns": columns
        }
        
        response = requests.post(url, headers=self.headers, json=data)
        if response.status_code == 200:
            return {"success": True, "data": response.json()}
        else:
            return {"success": False, "error": response.text, "status": response.status_code}
    
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
    
    def create_vector_search_index(self, index_name: str, endpoint_name: str, source_table_name: str, 
                                 primary_key: str, embedding_source_columns: List[Dict]) -> Dict[str, Any]:
        """Create a Vector Search index"""
        url = f"{self.host}/api/2.0/vector-search/indexes"
        data = {
            "name": index_name,
            "endpoint_name": endpoint_name,
            "source_table_name": source_table_name,
            "pipeline_type": "TRIGGERED",
            "primary_key": primary_key,
            "embedding_source_columns": embedding_source_columns
        }
        
        response = requests.post(url, headers=self.headers, json=data)
        if response.status_code == 200:
            return {"success": True, "data": response.json()}
        else:
            return {"success": False, "error": response.text, "status": response.status_code}
    
    def wait_for_endpoint_ready(self, endpoint_name: str, timeout: int = 300) -> Dict[str, Any]:
        """Wait for Vector Search endpoint to be ready"""
        url = f"{self.host}/api/2.0/vector-search/endpoints/{endpoint_name}"
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            response = requests.get(url, headers=self.headers)
            if response.status_code == 200:
                endpoint_info = response.json()
                if endpoint_info.get("state") == "ONLINE":
                    return {"success": True, "data": endpoint_info}
                elif endpoint_info.get("state") in ["FAILED", "ERROR"]:
                    return {"success": False, "error": f"Endpoint failed: {endpoint_info}"}
            time.sleep(10)
        
        return {"success": False, "error": "Timeout waiting for endpoint to be ready"}
    
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

# Agent 2: Complete Databricks Asset Generation Agent
def databricks_asset_agent(state: AgentState) -> AgentState:
    """Agent responsible for creating ALL Databricks assets including Vector Search"""
    
    print("🏗️  Creating COMPLETE Databricks assets (including Vector Search)...")
    
    # Initialize Databricks API client
    host = os.getenv("DATABRICKS_HOST")
    token = os.getenv("DATABRICKS_TOKEN")
    
    if not host or not token:
        state["error"] = "DATABRICKS_HOST and DATABRICKS_TOKEN must be set"
        return state
    
    api = DatabricksAPI(host, token)
    
    # Test connection first
    print("🔗 Testing Databricks connection...")
    connection_test = api.test_connection()
    if not connection_test["success"]:
        state["error"] = f"Databricks connection failed: {connection_test.get('error', 'Unknown error')}"
        return state
    
    print("✅ Databricks connection successful!")
    
    # Generate unique identifiers
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    catalog_name = f"rag_catalog_{timestamp}"
    schema_name = "rag_schema"
    table_name = "documents"
    endpoint_name = f"rag_endpoint_{timestamp}"
    index_name = f"rag_index_{timestamp}"
    folder_path = f"/Shared/RAG_Workflow_{timestamp}"
    
    assets_created = {}
    
    try:
        # Create catalog
        print(f"📚 Creating catalog: {catalog_name}")
        catalog_result = api.create_catalog(catalog_name, "Catalog for RAG workflow assets")
        if catalog_result["success"]:
            assets_created["catalog"] = {
                "name": catalog_name,
                "status": "created",
                "message": f"Successfully created catalog: {catalog_name}"
            }
            print(f"✅ Catalog created: {catalog_name}")
        else:
            assets_created["catalog"] = {
                "name": catalog_name,
                "status": "failed",
                "message": f"Failed to create catalog: {catalog_result.get('error', 'Unknown error')}"
            }
            print(f"❌ Catalog creation failed: {catalog_result.get('error', 'Unknown error')}")
        
        # Create schema
        print(f"📁 Creating schema: {catalog_name}.{schema_name}")
        schema_result = api.create_schema(catalog_name, schema_name, "Schema for RAG workflow tables")
        if schema_result["success"]:
            assets_created["schema"] = {
                "name": f"{catalog_name}.{schema_name}",
                "status": "created",
                "message": f"Successfully created schema: {catalog_name}.{schema_name}"
            }
            print(f"✅ Schema created: {catalog_name}.{schema_name}")
        else:
            assets_created["schema"] = {
                "name": f"{catalog_name}.{schema_name}",
                "status": "failed",
                "message": f"Failed to create schema: {schema_result.get('error', 'Unknown error')}"
            }
            print(f"❌ Schema creation failed: {schema_result.get('error', 'Unknown error')}")
        
        # Create documents table
        print(f"📋 Creating documents table: {catalog_name}.{schema_name}.{table_name}")
        columns = [
            {"name": "id", "type": "STRING", "nullable": False},
            {"name": "content", "type": "STRING", "nullable": False},
            {"name": "metadata", "type": "STRING", "nullable": True},
            {"name": "embedding", "type": "ARRAY<FLOAT>", "nullable": True}
        ]
        table_result = api.create_table(catalog_name, schema_name, table_name, columns)
        if table_result["success"]:
            assets_created["table"] = {
                "name": f"{catalog_name}.{schema_name}.{table_name}",
                "status": "created",
                "message": f"Successfully created table: {catalog_name}.{schema_name}.{table_name}"
            }
            print(f"✅ Table created: {catalog_name}.{schema_name}.{table_name}")
        else:
            assets_created["table"] = {
                "name": f"{catalog_name}.{schema_name}.{table_name}",
                "status": "failed",
                "message": f"Failed to create table: {table_result.get('error', 'Unknown error')}"
            }
            print(f"❌ Table creation failed: {table_result.get('error', 'Unknown error')}")
        
        # Create Vector Search endpoint
        print(f"🔍 Creating Vector Search endpoint: {endpoint_name}")
        endpoint_result = api.create_vector_search_endpoint(endpoint_name)
        if endpoint_result["success"]:
            assets_created["vector_search_endpoint"] = {
                "name": endpoint_name,
                "status": "created",
                "message": f"Successfully created Vector Search endpoint: {endpoint_name}"
            }
            print(f"✅ Vector Search endpoint created: {endpoint_name}")
            
            # Wait for endpoint to be ready
            print("⏳ Waiting for Vector Search endpoint to be ready...")
            ready_result = api.wait_for_endpoint_ready(endpoint_name)
            if ready_result["success"]:
                assets_created["vector_search_endpoint"]["status"] = "ready"
                assets_created["vector_search_endpoint"]["message"] += " (endpoint is ready)"
                print(f"✅ Vector Search endpoint is ready: {endpoint_name}")
            else:
                assets_created["vector_search_endpoint"]["status"] = "created_but_not_ready"
                assets_created["vector_search_endpoint"]["message"] += f" (endpoint created but not ready: {ready_result.get('error', 'Unknown error')})"
                print(f"⚠️  Vector Search endpoint created but not ready: {ready_result.get('error', 'Unknown error')}")
        else:
            assets_created["vector_search_endpoint"] = {
                "name": endpoint_name,
                "status": "failed",
                "message": f"Failed to create Vector Search endpoint: {endpoint_result.get('error', 'Unknown error')}"
            }
            print(f"❌ Vector Search endpoint creation failed: {endpoint_result.get('error', 'Unknown error')}")
        
        # Create Vector Search index
        if assets_created.get("vector_search_endpoint", {}).get("status") in ["created", "ready"]:
            print(f"🔍 Creating Vector Search index: {index_name}")
            embedding_source_columns = [
                {
                    "name": "embedding",
                    "embedding_model_endpoint_name": "databricks-bge-large-en-v1.5"
                }
            ]
            index_result = api.create_vector_search_index(
                index_name, 
                endpoint_name, 
                f"{catalog_name}.{schema_name}.{table_name}",
                "id",
                embedding_source_columns
            )
            if index_result["success"]:
                assets_created["vector_search_index"] = {
                    "name": index_name,
                    "status": "created",
                    "message": f"Successfully created Vector Search index: {index_name}"
                }
                print(f"✅ Vector Search index created: {index_name}")
            else:
                assets_created["vector_search_index"] = {
                    "name": index_name,
                    "status": "failed",
                    "message": f"Failed to create Vector Search index: {index_result.get('error', 'Unknown error')}"
                }
                print(f"❌ Vector Search index creation failed: {index_result.get('error', 'Unknown error')}")
        
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
        
        # Create complete RAG workflow notebook
        notebook_path = f"{folder_path}/Complete_RAG_Workflow"
        print(f"📓 Creating complete RAG workflow notebook: {notebook_path}")
        rag_notebook_content = generate_complete_rag_notebook_content(
            timestamp, catalog_name, schema_name, table_name, endpoint_name, index_name
        )
        notebook_result = api.create_notebook(notebook_path, rag_notebook_content)
        if notebook_result["success"]:
            assets_created["complete_rag_notebook"] = {
                "path": notebook_path,
                "status": "created",
                "message": f"Successfully created complete RAG notebook: {notebook_path}"
            }
            print(f"✅ Complete RAG notebook created: {notebook_path}")
        else:
            assets_created["complete_rag_notebook"] = {
                "path": notebook_path,
                "status": "failed",
                "message": f"Failed to create complete RAG notebook: {notebook_result.get('error', 'Unknown error')}"
            }
            print(f"❌ Complete RAG notebook creation failed: {notebook_result.get('error', 'Unknown error')}")
        
        state["databricks_assets"] = assets_created
        state["current_step"] = "databricks_assets_created"
        
    except Exception as e:
        state["error"] = f"Error in databricks_asset_agent: {str(e)}"
        state["current_step"] = "error"
    
    return state
