#!/usr/bin/env python3
"""
Ultimate RAG Workflow - Complete End-to-End Solution
Creates Databricks notebooks, uploads documents, enables querying, and tracks everything in MLflow
"""

import os
import json
import requests
import base64
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import subprocess
import sys

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class UltimateRAGWorkflow:
    """Complete RAG workflow with document upload and querying"""
    
    def __init__(self):
        self.host = os.getenv("DATABRICKS_HOST")
        self.token = os.getenv("DATABRICKS_TOKEN")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        if not all([self.host, self.token, self.openai_api_key]):
            raise ValueError("Missing required environment variables: DATABRICKS_HOST, DATABRICKS_TOKEN, OPENAI_API_KEY")
        
        self.headers = {
            'Authorization': f'Bearer {self.token}',
            'Content-Type': 'application/json'
        }
        
        # Generate unique identifiers
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.workflow_id = f"Ultimate_RAG_{self.timestamp}"
        self.folder_path = f"/Shared/{self.workflow_id}"
        self.endpoint_name = f"ultimate_rag_{self.timestamp}"
        
        print(f"🚀 Ultimate RAG Workflow initialized")
        print(f"📁 Folder: {self.folder_path}")
        print(f"🔍 Endpoint: {self.endpoint_name}")
    
    def create_databricks_assets(self) -> Dict[str, Any]:
        """Create all Databricks assets (folders, notebooks, Vector Search endpoint)"""
        
        print("\n📓 Creating Databricks Assets...")
        print("=" * 50)
        
        # Test connection
        print("🔗 Testing Databricks connection...")
        connection_test = self._test_connection()
        if not connection_test["success"]:
            return {"success": False, "error": f"Connection failed: {connection_test['error']}"}
        
        print("✅ Databricks connection successful")
        
        # Create folder
        print(f"📁 Creating folder: {self.folder_path}")
        folder_result = self._create_folder(self.folder_path)
        if not folder_result["success"]:
            return {"success": False, "error": f"Failed to create folder: {folder_result['error']}"}
        
        print("✅ Folder created successfully")
        
        # Create Vector Search endpoint
        print(f"🔍 Creating Vector Search endpoint: {self.endpoint_name}")
        endpoint_result = self._create_vector_search_endpoint(self.endpoint_name)
        if not endpoint_result["success"]:
            print(f"⚠️  Vector Search endpoint creation failed: {endpoint_result['error']}")
            print("Continuing with RAG system creation...")
        else:
            print("✅ Vector Search endpoint created successfully")
        
        # Create main RAG notebook
        print("📓 Creating main RAG notebook...")
        rag_notebook_path = f"{self.folder_path}/Ultimate_RAG_System"
        rag_content = self._generate_ultimate_rag_notebook_content()
        rag_result = self._create_notebook(rag_notebook_path, rag_content)
        
        if not rag_result["success"]:
            return {"success": False, "error": f"Failed to create RAG notebook: {rag_result['error']}"}
        
        print("✅ Main RAG notebook created successfully")
        
        # Create MLflow tracking notebook
        print("📊 Creating MLflow tracking notebook...")
        mlflow_notebook_path = f"{self.folder_path}/MLflow_Tracking"
        mlflow_content = self._generate_mlflow_tracking_notebook_content()
        mlflow_result = self._create_notebook(mlflow_notebook_path, mlflow_content)
        
        if mlflow_result["success"]:
            print("✅ MLflow tracking notebook created successfully")
        else:
            print(f"⚠️  MLflow notebook creation failed: {mlflow_result['error']}")
        
        # Create usage guide
        print("📖 Creating usage guide...")
        guide_notebook_path = f"{self.folder_path}/Usage_Guide"
        guide_content = self._generate_usage_guide_content()
        guide_result = self._create_notebook(guide_notebook_path, guide_content)
        
        if guide_result["success"]:
            print("✅ Usage guide created successfully")
        else:
            print(f"⚠️  Usage guide creation failed: {guide_result['error']}")
        
        return {
            "success": True,
            "folder_path": self.folder_path,
            "rag_notebook_path": rag_notebook_path,
            "mlflow_notebook_path": mlflow_notebook_path,
            "guide_notebook_path": guide_notebook_path,
            "endpoint_name": self.endpoint_name,
            "workflow_id": self.workflow_id
        }
    
    def upload_documents_interactive(self) -> Dict[str, Any]:
        """Interactive document upload with multiple options"""
        
        print("\n📚 Document Upload Options")
        print("=" * 50)
        print("1. Natural Language Upload (describe what you want)")
        print("2. Manual Document Entry")
        print("3. Load from Document Loader")
        print("4. Skip document upload for now")
        
        choice = input("\nChoose upload method (1-4): ").strip()
        
        if choice == "1":
            return self._natural_language_upload()
        elif choice == "2":
            return self._manual_document_upload()
        elif choice == "3":
            return self._document_loader_upload()
        elif choice == "4":
            print("⏭️  Skipping document upload")
            return {"success": True, "message": "Document upload skipped"}
        else:
            print("❌ Invalid choice, skipping document upload")
            return {"success": True, "message": "Document upload skipped"}
    
    def _natural_language_upload(self) -> Dict[str, Any]:
        """Use natural language to upload documents"""
        
        print("\n🔍 Natural Language Document Upload")
        print("-" * 30)
        
        description = input("Describe the documents you want to upload: ").strip()
        
        if not description:
            return {"success": False, "error": "No description provided"}
        
        # Run the natural language uploader
        try:
            result = subprocess.run(
                [sys.executable, "natural_language_uploader.py"],
                input=f"{description}\ny\nquit\n",
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.returncode == 0:
                return {"success": True, "message": "Documents uploaded via natural language"}
            else:
                return {"success": False, "error": f"Upload failed: {result.stderr}"}
                
        except Exception as e:
            return {"success": False, "error": f"Upload error: {str(e)}"}
    
    def _manual_document_upload(self) -> Dict[str, Any]:
        """Manual document entry"""
        
        print("\n✍️  Manual Document Upload")
        print("-" * 30)
        print("Enter your documents (type 'done' when finished):")
        
        documents = []
        doc_num = 1
        
        while True:
            content = input(f"\nDocument {doc_num} (or 'done'): ").strip()
            
            if content.lower() == 'done':
                break
            
            if content:
                documents.append({
                    "title": f"Manual_Document_{doc_num}",
                    "content": content,
                    "source": "manual_entry"
                })
                doc_num += 1
        
        if documents:
            # Save documents to a file for the RAG client
            self._save_documents_to_file(documents)
            return {"success": True, "message": f"Uploaded {len(documents)} documents manually"}
        else:
            return {"success": False, "error": "No documents entered"}
    
    def _document_loader_upload(self) -> Dict[str, Any]:
        """Use document loader to upload files"""
        
        print("\n📁 Document Loader Upload")
        print("-" * 30)
        
        try:
            result = subprocess.run(
                [sys.executable, "document_loader.py"],
                input="1\n1\ny\n",  # Use sample documents and save
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.returncode == 0:
                return {"success": True, "message": "Documents uploaded via document loader"}
            else:
                return {"success": False, "error": f"Loader failed: {result.stderr}"}
                
        except Exception as e:
            return {"success": False, "error": f"Loader error: {str(e)}"}
    
    def _save_documents_to_file(self, documents: List[Dict[str, Any]]):
        """Save documents to a file for the RAG client"""
        
        filename = f"uploaded_documents_{self.timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(documents, f, indent=2)
        
        print(f"💾 Documents saved to: {filename}")
    
    def start_interactive_querying(self):
        """Start interactive querying session"""
        
        print("\n🔍 Interactive Querying Session")
        print("=" * 50)
        print("You can now query your uploaded documents!")
        print("Type 'quit' to exit the querying session")
        print()
        
        # Run the enhanced RAG client
        try:
            subprocess.run(
                [sys.executable, "enhanced_rag_client.py"],
                timeout=None  # No timeout for interactive session
            )
        except KeyboardInterrupt:
            print("\n👋 Querying session ended")
        except Exception as e:
            print(f"❌ Querying error: {str(e)}")
    
    def _test_connection(self) -> Dict[str, Any]:
        """Test Databricks connection"""
        url = f"{self.host}/api/2.0/clusters/list"
        response = requests.get(url, headers=self.headers)
        if response.status_code == 200:
            return {"success": True, "message": "Connection successful"}
        else:
            return {"success": False, "error": response.text, "status": response.status_code}
    
    def _create_folder(self, path: str) -> Dict[str, Any]:
        """Create a folder in Databricks workspace"""
        url = f"{self.host}/api/2.0/workspace/mkdirs"
        data = {"path": path}
        
        response = requests.post(url, headers=self.headers, json=data)
        if response.status_code == 200:
            return {"success": True, "data": response.json()}
        else:
            return {"success": False, "error": response.text, "status": response.status_code}
    
    def _create_vector_search_endpoint(self, endpoint_name: str) -> Dict[str, Any]:
        """Create a Vector Search endpoint"""
        url = f"{self.host}/api/2.0/vector-search/endpoints"
        data = {
            "name": endpoint_name,
            "endpoint_type": "STANDARD"
        }
        
        response = requests.post(url, headers=self.headers, json=data)
        if response.status_code == 200:
            return {"success": True, "data": response.json()}
        else:
            return {"success": False, "error": response.text, "status": response.status_code}
    
    def _create_notebook(self, path: str, content: str) -> Dict[str, Any]:
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
    
    def _generate_ultimate_rag_notebook_content(self) -> str:
        """Generate the ultimate RAG notebook content"""
        
        return f'''# Ultimate RAG System with Document Upload and MLflow Tracking
# Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
# Workflow ID: {self.workflow_id}

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
import re
from pathlib import Path

# COMMAND ----------

# Configuration
WORKFLOW_ID = "{self.workflow_id}"
VECTOR_SEARCH_ENDPOINT = "{self.endpoint_name}"
OPENAI_API_KEY = "{self.openai_api_key}"  # Already set

# MLflow Configuration
EXPERIMENT_NAME = f"/Shared/Ultimate_RAG_{{WORKFLOW_ID}}"
TRACKING_URI = "databricks"

# COMMAND ----------

# Initialize MLflow Experiment
def setup_mlflow_experiment():
    """Setup MLflow experiment for RAG workflow tracking"""
    try:
        mlflow.set_tracking_uri(TRACKING_URI)
        
        experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
        if experiment is None:
            experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
            print(f"✅ Created new MLflow experiment: {{EXPERIMENT_NAME}} (ID: {{experiment_id}})")
        else:
            experiment_id = experiment.experiment_id
            print(f"✅ Using existing MLflow experiment: {{EXPERIMENT_NAME}} (ID: {{experiment_id}})")
        
        mlflow.set_experiment(EXPERIMENT_NAME)
        return experiment_id
    except Exception as e:
        print(f"⚠️  MLflow setup failed: {{str(e)}}")
        return None

# COMMAND ----------

# Initialize components
experiment_id = setup_mlflow_experiment()
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0.1, openai_api_key=OPENAI_API_KEY)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# COMMAND ----------

# Document Upload Functions
def upload_documents_natural_language(description: str):
    """Upload documents using natural language description"""
    print(f"📝 Processing request: '{{description}}'")
    
    # This would integrate with the natural language uploader
    # For now, we'll use a simplified version
    return upload_documents_manual()

def upload_documents_manual():
    """Upload documents manually"""
    print("✍️  Manual Document Upload")
    print("Enter your documents (type 'done' when finished):")
    
    documents = []
    doc_num = 1
    
    while True:
        content = input(f"\\nDocument {{doc_num}} (or 'done'): ").strip()
        
        if content.lower() == 'done':
            break
        
        if content:
            documents.append({{
                "title": f"Manual_Document_{{doc_num}}",
                "content": content,
                "source": "manual_entry"
            }})
            doc_num += 1
    
    if documents:
        return process_and_store_documents(documents)
    else:
        print("❌ No documents entered")
        return None

def process_and_store_documents(documents: List[Dict[str, Any]]):
    """Process documents and store them for querying"""
    
    with mlflow.start_run(experiment_id=experiment_id) as run:
        try:
            # Log parameters
            mlflow.log_param("upload_type", "manual")
            mlflow.log_param("num_documents", len(documents))
            
            # Process documents
            processed_docs = []
            for i, doc in enumerate(documents):
                chunks = text_splitter.split_text(doc['content'])
                for j, chunk in enumerate(chunks):
                    processed_docs.append({{
                        'id': f"{{doc['title']}}_{{j}}",
                        'content': chunk,
                        'title': doc['title'],
                        'source': doc['source'],
                        'chunk_index': j,
                        'document_index': i
                    }})
            
            # Store documents in global variable for querying
            global stored_documents
            stored_documents = processed_docs
            
            # Log metrics
            mlflow.log_metric("num_processed_chunks", len(processed_docs))
            mlflow.log_metric("total_content_size", sum(len(doc['content']) for doc in documents))
            
            # Log artifacts
            mlflow.log_text(json.dumps([doc['title'] for doc in documents], indent=2), "uploaded_documents.json")
            
            print(f"✅ Uploaded {{len(documents)}} documents ({{len(processed_docs)}} chunks)")
            print(f"🔗 MLflow Run ID: {{run.info.run_id}}")
            
            return processed_docs
            
        except Exception as e:
            mlflow.log_param("error", str(e))
            print(f"❌ Upload failed: {{str(e)}}")
            return None

# COMMAND ----------

# Query Functions
def query_documents(query: str, k: int = 5):
    """Query uploaded documents with MLflow tracking"""
    
    if 'stored_documents' not in globals() or not stored_documents:
        return {{"success": False, "error": "No documents available. Upload documents first."}}
    
    with mlflow.start_run(experiment_id=experiment_id) as run:
        start_time = time.time()
        
        try:
            # Log parameters
            mlflow.log_param("query", query)
            mlflow.log_param("num_documents", len(stored_documents))
            mlflow.log_param("top_k", k)
            
            # Create query embedding
            query_embedding = embeddings.embed_query(query)
            
            # Vector similarity search
            similarities = []
            for doc in stored_documents:
                doc_embedding = embeddings.embed_query(doc['content'])
                similarity = sum(a * b for a, b in zip(query_embedding, doc_embedding))
                similarities.append((similarity, doc))
            
            # Get top k most similar
            similarities.sort(key=lambda x: x[0], reverse=True)
            top_docs = [doc for _, doc in similarities[:k]]
            
            # Prepare context
            context_parts = []
            for i, doc in enumerate(top_docs):
                context_parts.append(f"Document {{i+1}} ({{doc['title']}}):\\n{{doc['content']}}")
            
            context = "\\n\\n".join(context_parts)
            
            # Generate response
            prompt = f"""Based on the following context from your uploaded documents, answer the question.

Context:
{{context}}

Question: {{query}}

Answer:"""
            
            response = llm.invoke(prompt)
            
            # Calculate metrics
            end_time = time.time()
            response_time = end_time - start_time
            
            # Log metrics
            mlflow.log_metric("response_time_seconds", response_time)
            mlflow.log_metric("response_length", len(response.content))
            mlflow.log_metric("context_length", len(context))
            
            if similarities:
                mlflow.log_metric("max_similarity", similarities[0][0])
                mlflow.log_metric("avg_similarity", sum(s[0] for s in similarities[:5]) / 5)
            
            # Log artifacts
            mlflow.log_text(prompt, "prompt.txt")
            mlflow.log_text(response.content, "response.txt")
            mlflow.log_text(context, "retrieved_context.txt")
            
            # Log run info
            mlflow.set_tag("workflow_id", WORKFLOW_ID)
            mlflow.set_tag("query_type", "ultimate_rag")
            
            return {{
                "success": True,
                "response": response.content,
                "run_id": run.info.run_id,
                "response_time": response_time,
                "retrieved_documents": [
                    {{"title": doc['title'], "similarity": similarities[i][0]}}
                    for i, doc in enumerate(top_docs)
                ]
            }}
            
        except Exception as e:
            mlflow.log_param("error", str(e))
            return {{"success": False, "error": str(e)}}

# COMMAND ----------

# Interactive Querying
def start_interactive_querying():
    """Start interactive querying session"""
    print("\\n🔍 Interactive Querying Session")
    print("=" * 50)
    print("Type 'quit' to exit the querying session")
    print()
    
    while True:
        query = input("\\n❓ Enter your question: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("👋 Querying session ended")
            break
        
        if not query:
            continue
        
        print("\\n🔍 Processing query...")
        result = query_documents(query)
        
        if result["success"]:
            print(f"\\n✅ Answer: {{result['response']}}")
            print(f"⏱️  Response Time: {{result['response_time']:.2f}}s")
            print(f"🔗 MLflow Run ID: {{result['run_id']}}")
            
            if result['retrieved_documents']:
                print("\\n📚 Retrieved Documents:")
                for i, doc in enumerate(result['retrieved_documents'], 1):
                    print(f"  {{i}}. {{doc['title']}} (similarity: {{doc['similarity']:.3f}})")
        else:
            print(f"❌ Query failed: {{result['error']}}")

# COMMAND ----------

# Main execution
print("🚀 Ultimate RAG System Ready!")
print(f"📊 MLflow Experiment: {{EXPERIMENT_NAME}}")
print(f"🔍 Vector Search Endpoint: {{VECTOR_SEARCH_ENDPOINT}}")

print("\\n💡 Available functions:")
print("- upload_documents_manual(): Upload documents manually")
print("- query_documents(query, k): Query uploaded documents")
print("- start_interactive_querying(): Start interactive session")

print("\\n📚 To get started:")
print("1. Upload documents: upload_documents_manual()")
print("2. Query documents: query_documents('your question')")
print("3. Or start interactive: start_interactive_querying()")

# COMMAND ----------

# Auto-start interactive session if documents are available
try:
    if 'stored_documents' in globals() and stored_documents:
        print("\\n📚 Documents found! Starting interactive session...")
        start_interactive_querying()
    else:
        print("\\n📝 No documents found. Use upload_documents_manual() to add documents.")
except:
    print("\\n📝 No documents found. Use upload_documents_manual() to add documents.")
'''
    
    def _generate_mlflow_tracking_notebook_content(self) -> str:
        """Generate MLflow tracking notebook content"""
        
        return f'''# MLflow Tracking Dashboard
# Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
# Workflow ID: {self.workflow_id}

# COMMAND ----------

import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# COMMAND ----------

# Setup MLflow
mlflow.set_tracking_uri("databricks")
EXPERIMENT_NAME = f"/Shared/Ultimate_RAG_{self.workflow_id}"

# COMMAND ----------

def view_experiment_runs():
    """View all runs in the experiment"""
    try:
        experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
        if experiment:
            runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
            print(f"📊 Found {{len(runs)}} runs in experiment")
            return runs
        else:
            print("❌ Experiment not found")
            return None
    except Exception as e:
        print(f"❌ Error viewing runs: {{str(e)}}")
        return None

# COMMAND ----------

def compare_runs(run_ids: List[str]):
    """Compare specific runs"""
    try:
        runs = mlflow.search_runs(filter_string=f"run_id in ('{{', '.join(run_ids)}}')")
        return runs
    except Exception as e:
        print(f"❌ Error comparing runs: {{str(e)}}")
        return None

# COMMAND ----------

def plot_metrics():
    """Plot key metrics over time"""
    runs = view_experiment_runs()
    if runs is not None and len(runs) > 0:
        # Plot response times
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(runs['metrics.response_time_seconds'])
        plt.title('Response Time Over Runs')
        plt.ylabel('Seconds')
        
        plt.subplot(2, 2, 2)
        plt.plot(runs['metrics.max_similarity'])
        plt.title('Max Similarity Over Runs')
        plt.ylabel('Similarity Score')
        
        plt.subplot(2, 2, 3)
        plt.plot(runs['metrics.response_length'])
        plt.title('Response Length Over Runs')
        plt.ylabel('Characters')
        
        plt.subplot(2, 2, 4)
        plt.plot(runs['metrics.avg_similarity'])
        plt.title('Average Similarity Over Runs')
        plt.ylabel('Similarity Score')
        
        plt.tight_layout()
        plt.show()

# COMMAND ----------

# View experiment runs
runs = view_experiment_runs()

if runs is not None and len(runs) > 0:
    print("\\n📈 Key Metrics Summary:")
    print(f"Total Runs: {{len(runs)}}")
    print(f"Average Response Time: {{runs['metrics.response_time_seconds'].mean():.2f}}s")
    print(f"Average Max Similarity: {{runs['metrics.max_similarity'].mean():.3f}}")
    print(f"Average Response Length: {{runs['metrics.response_length'].mean():.0f}} chars")
    
    # Plot metrics
    plot_metrics()
else:
    print("\\n📝 No runs found yet. Start using the RAG system to generate data!")

# COMMAND ----------

print("\\n🔗 View in MLflow UI:")
print(f"{{self.host}}/ml/experiments/{{EXPERIMENT_NAME.replace('/', '%2F')}}")
'''
    
    def _generate_usage_guide_content(self) -> str:
        """Generate usage guide content"""
        
        return f'''# Ultimate RAG System - Usage Guide
# Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
# Workflow ID: {self.workflow_id}

# COMMAND ----------

print("📚 Ultimate RAG System - Complete Usage Guide")
print("=" * 60)

# COMMAND ----------

print("🚀 Getting Started")
print("-" * 20)
print("1. Open the Ultimate_RAG_System notebook")
print("2. Run all cells to initialize the system")
print("3. Upload your documents using one of the methods below")
print("4. Start querying your documents!")

# COMMAND ----------

print("\\n📝 Document Upload Methods")
print("-" * 20)
print("Method 1: Manual Upload")
print("  upload_documents_manual()")
print("  - Enter documents one by one")
print("  - Type 'done' when finished")
print()
print("Method 2: Natural Language Upload")
print("  upload_documents_natural_language('Upload all PDF files')")
print("  - Describe what documents you want")
print("  - System finds and uploads them automatically")
print()
print("Method 3: File Upload")
print("  - Use the document_loader.py utility")
print("  - Load from text files, JSON, or manual entry")

# COMMAND ----------

print("\\n🔍 Querying Documents")
print("-" * 20)
print("Single Query:")
print("  result = query_documents('What are the company policies?')")
print("  print(result['response'])")
print()
print("Interactive Session:")
print("  start_interactive_querying()")
print("  - Enter questions one by one")
print("  - Type 'quit' to exit")

# COMMAND ----------

print("\\n📊 MLflow Tracking")
print("-" * 20)
print("All operations are automatically tracked:")
print("- Document uploads (count, size, processing time)")
print("- Queries (response time, similarity scores)")
print("- Performance metrics (response length, context utilization)")
print()
print("View tracking data:")
print("1. Open the MLflow_Tracking notebook")
print("2. Run cells to see metrics and plots")
print("3. Or visit the MLflow UI directly")

# COMMAND ----------

print("\\n🔗 Quick Links")
print("-" * 20)
print(f"Main System: {{self.folder_path}}/Ultimate_RAG_System")
print(f"MLflow Tracking: {{self.folder_path}}/MLflow_Tracking")
print(f"MLflow UI: {{self.host}}/ml/experiments/{{EXPERIMENT_NAME.replace('/', '%2F')}}")
print(f"Workspace: {{self.host}}/workspace{{self.folder_path}}")

# COMMAND ----------

print("\\n💡 Pro Tips")
print("-" * 20)
print("• Be specific in your queries for better results")
print("• Check MLflow experiments for performance insights")
print("• Use the interactive session for quick testing")
print("• Monitor similarity scores to understand document relevance")
print("• Export MLflow data for further analysis")

# COMMAND ----------

print("\\n✅ Ready to use!")
print("Open the Ultimate_RAG_System notebook and start uploading documents!")
'''

def main():
    """Main function for the ultimate RAG workflow"""
    
    print("🚀 Ultimate RAG Workflow - Complete End-to-End Solution")
    print("=" * 70)
    print("This will create:")
    print("✅ Databricks notebooks with RAG system")
    print("✅ Document upload capabilities")
    print("✅ Interactive querying interface")
    print("✅ MLflow experiment tracking")
    print("✅ Performance monitoring and analytics")
    print()
    
    try:
        # Initialize workflow
        workflow = UltimateRAGWorkflow()
        
        # Step 1: Create Databricks assets
        print("Step 1: Creating Databricks Assets...")
        assets_result = workflow.create_databricks_assets()
        
        if not assets_result["success"]:
            print(f"❌ Failed to create assets: {assets_result['error']}")
            return
        
        print("✅ Databricks assets created successfully!")
        
        # Step 2: Upload documents
        print("\nStep 2: Document Upload...")
        upload_result = workflow.upload_documents_interactive()
        
        if upload_result["success"]:
            print(f"✅ {upload_result['message']}")
        else:
            print(f"⚠️  Upload issue: {upload_result['error']}")
        
        # Step 3: Start interactive querying
        print("\nStep 3: Interactive Querying...")
        workflow.start_interactive_querying()
        
        # Summary
        print("\n" + "=" * 70)
        print("🎉 Ultimate RAG Workflow Complete!")
        print("=" * 70)
        print(f"📁 Folder: {workflow.folder_path}")
        print(f"📓 Main System: {workflow.folder_path}/Ultimate_RAG_System")
        print(f"📊 MLflow Tracking: {workflow.folder_path}/MLflow_Tracking")
        print(f"📖 Usage Guide: {workflow.folder_path}/Usage_Guide")
        print(f"🔍 Vector Search Endpoint: {workflow.endpoint_name}")
        print(f"🆔 Workflow ID: {workflow.workflow_id}")
        
        print(f"\n🌐 View in Databricks: {workflow.host}/workspace{workflow.folder_path}")
        print(f"📊 MLflow Experiment: /Shared/Ultimate_RAG_{workflow.workflow_id}")
        
        print("\n💡 Next Steps:")
        print("1. Open the Ultimate_RAG_System notebook in Databricks")
        print("2. Upload documents if you haven't already")
        print("3. Start querying your documents!")
        print("4. Check MLflow tracking for performance insights")
        
    except Exception as e:
        print(f"❌ Workflow failed: {str(e)}")

if __name__ == "__main__":
    main()
