import azure.functions as func
import logging
import json
import os
import subprocess
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
import uuid
from typing import Dict, Any, List, Optional
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# In-memory storage for workflow status (in production, use Azure Table Storage or Cosmos DB)
workflow_status = {}

def main(req: func.HttpRequest) -> func.HttpResponse:
    """Main Azure Function entry point"""
    try:
        # Parse the request
        method = req.method
        route = req.route_params.get('route', '')
        
        # Route to appropriate handler
        if method == 'GET':
            if route == 'health':
                return health_check()
            elif route == 'workflows':
                return list_workflows()
            elif 'workflow_id' in req.route_params:
                return get_workflow_status(req.route_params['workflow_id'])
        elif method == 'POST':
            if route == 'workflows':
                return create_workflow(req)
            elif 'workflow_id' in req.route_params and 'query' in route:
                return query_workflow(req)
            elif 'workflow_id' in req.route_params and 'upload' in route:
                return upload_documents(req)
        elif method == 'DELETE':
            if 'workflow_id' in req.route_params:
                return delete_workflow(req.route_params['workflow_id'])
        
        # Default response for unhandled routes
        return func.HttpResponse(
            json.dumps({"error": "Route not found"}),
            status_code=404,
            mimetype="application/json"
        )
    
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            status_code=500,
            mimetype="application/json"
        )

def health_check() -> func.HttpResponse:
    """Health check endpoint"""
    return func.HttpResponse(
        json.dumps({
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "service": "azure-function-rag-workflow"
        }),
        status_code=200,
        mimetype="application/json"
    )

def create_workflow(req: func.HttpRequest) -> func.HttpResponse:
    """Create a new RAG workflow"""
    try:
        # Parse request body
        body = req.get_json()
        workflow_name = body.get('workflow_name')
        create_assets = body.get('create_assets', True)
        upload_documents = body.get('upload_documents', False)
        documents = body.get('documents', [])
        
        workflow_id = str(uuid.uuid4())
        
        # Initialize workflow status
        workflow_status[workflow_id] = {
            "status": "initializing",
            "progress": "Starting workflow creation...",
            "error": None,
            "created_at": datetime.now().isoformat(),
            "workflow_name": workflow_name
        }
        
        # Start workflow in background (Azure Functions don't support true background tasks)
        # In production, use Azure Durable Functions or Logic Apps
        try:
            run_workflow_sync(workflow_id, workflow_name, create_assets, documents)
        except Exception as e:
            workflow_status[workflow_id]["status"] = "failed"
            workflow_status[workflow_id]["error"] = str(e)
        
        return func.HttpResponse(
            json.dumps({
                "workflow_id": workflow_id,
                "status": "initializing",
                "message": "Workflow creation started",
                "created_at": datetime.now().isoformat()
            }),
            status_code=202,
            mimetype="application/json"
        )
    
    except Exception as e:
        logger.error(f"Error creating workflow: {str(e)}")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            status_code=500,
            mimetype="application/json"
        )

def get_workflow_status(workflow_id: str) -> func.HttpResponse:
    """Get the status of a workflow"""
    if workflow_id not in workflow_status:
        return func.HttpResponse(
            json.dumps({"error": "Workflow not found"}),
            status_code=404,
            mimetype="application/json"
        )
    
    status = workflow_status[workflow_id]
    return func.HttpResponse(
        json.dumps({
            "workflow_id": workflow_id,
            "status": status["status"],
            "progress": status.get("progress"),
            "error": status.get("error"),
            "created_at": status.get("created_at")
        }),
        status_code=200,
        mimetype="application/json"
    )

def query_workflow(req: func.HttpRequest) -> func.HttpResponse:
    """Query a RAG workflow"""
    try:
        workflow_id = req.route_params['workflow_id']
        body = req.get_json()
        query = body.get('query')
        
        if not query:
            return func.HttpResponse(
                json.dumps({"error": "Query is required"}),
                status_code=400,
                mimetype="application/json"
            )
        
        if workflow_id not in workflow_status:
            return func.HttpResponse(
                json.dumps({"error": "Workflow not found"}),
                status_code=404,
                mimetype="application/json"
            )
        
        if workflow_status[workflow_id]["status"] != "completed":
            return func.HttpResponse(
                json.dumps({"error": "Workflow not ready for querying"}),
                status_code=400,
                mimetype="application/json"
            )
        
        # Run the query
        result = run_rag_query_sync(workflow_id, query)
        
        return func.HttpResponse(
            json.dumps(result),
            status_code=200,
            mimetype="application/json"
        )
    
    except Exception as e:
        logger.error(f"Error querying workflow: {str(e)}")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            status_code=500,
            mimetype="application/json"
        )

def upload_documents(req: func.HttpRequest) -> func.HttpResponse:
    """Upload documents to a workflow"""
    try:
        workflow_id = req.route_params['workflow_id']
        
        if workflow_id not in workflow_status:
            return func.HttpResponse(
                json.dumps({"error": "Workflow not found"}),
                status_code=404,
                mimetype="application/json"
            )
        
        # Handle file uploads (Azure Functions handle this differently)
        # For now, we'll accept document content in the request body
        body = req.get_json()
        documents = body.get('documents', [])
        
        if not documents:
            return func.HttpResponse(
                json.dumps({"error": "No documents provided"}),
                status_code=400,
                mimetype="application/json"
            )
        
        # Process documents
        process_documents_sync(workflow_id, documents)
        
        return func.HttpResponse(
            json.dumps({
                "workflow_id": workflow_id,
                "message": f"Successfully uploaded {len(documents)} documents"
            }),
            status_code=200,
            mimetype="application/json"
        )
    
    except Exception as e:
        logger.error(f"Error uploading documents: {str(e)}")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            status_code=500,
            mimetype="application/json"
        )

def list_workflows() -> func.HttpResponse:
    """List all workflows"""
    workflows = []
    for workflow_id, status in workflow_status.items():
        workflows.append({
            "workflow_id": workflow_id,
            "status": status["status"],
            "created_at": status["created_at"],
            "progress": status.get("progress"),
            "workflow_name": status.get("workflow_name")
        })
    
    return func.HttpResponse(
        json.dumps({"workflows": workflows}),
        status_code=200,
        mimetype="application/json"
    )

def delete_workflow(workflow_id: str) -> func.HttpResponse:
    """Delete a workflow"""
    if workflow_id not in workflow_status:
        return func.HttpResponse(
            json.dumps({"error": "Workflow not found"}),
            status_code=404,
            mimetype="application/json"
        )
    
    del workflow_status[workflow_id]
    
    return func.HttpResponse(
        json.dumps({"message": f"Workflow {workflow_id} deleted successfully"}),
        status_code=200,
        mimetype="application/json"
    )

def run_workflow_sync(workflow_id: str, workflow_name: str, create_assets: bool, documents: List[Dict[str, str]]):
    """Synchronous workflow execution"""
    try:
        workflow_status[workflow_id]["status"] = "running"
        workflow_status[workflow_id]["progress"] = "Creating Databricks assets..."
        
        # Process documents if provided
        if documents:
            workflow_status[workflow_id]["progress"] = "Processing documents..."
            process_documents_sync(workflow_id, documents)
        
        # Run the ultimate RAG workflow
        cmd = ["python3", "ultimate_rag_workflow.py", "--workflow-id", workflow_id]
        
        if workflow_name:
            cmd.extend(["--workflow-name", workflow_name])
        
        if not create_assets:
            cmd.append("--skip-assets")
        
        # Run the workflow
        process = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if process.returncode == 0:
            workflow_status[workflow_id]["status"] = "completed"
            workflow_status[workflow_id]["progress"] = "Workflow completed successfully"
            workflow_status[workflow_id]["databricks_assets"] = parse_databricks_assets(process.stdout)
        else:
            workflow_status[workflow_id]["status"] = "failed"
            workflow_status[workflow_id]["error"] = process.stderr or "Unknown error occurred"
            logger.error(f"Workflow {workflow_id} failed: {process.stderr}")
    
    except subprocess.TimeoutExpired:
        workflow_status[workflow_id]["status"] = "failed"
        workflow_status[workflow_id]["error"] = "Workflow execution timed out"
    except Exception as e:
        workflow_status[workflow_id]["status"] = "failed"
        workflow_status[workflow_id]["error"] = str(e)
        logger.error(f"Error running workflow {workflow_id}: {str(e)}")

def run_rag_query_sync(workflow_id: str, query: str) -> Dict[str, Any]:
    """Synchronous RAG query execution"""
    try:
        # Create a temporary script to run the query
        query_script = f"""
import sys
sys.path.append('.')
from enhanced_rag_client import EnhancedRAGClient, EnhancedRAGConfig
import json

def run_query():
    config = EnhancedRAGConfig(
        openai_api_key="{os.getenv('OPENAI_API_KEY')}",
        databricks_host="{os.getenv('DATABRICKS_HOST')}",
        databricks_token="{os.getenv('DATABRICKS_TOKEN')}",
        workflow_id="{workflow_id}"
    )
    
    client = EnhancedRAGClient(config)
    result = client.query_real_documents("{query}")
    return result

if __name__ == "__main__":
    result = run_query()
    print(json.dumps(result))
"""
        
        # Write script to temporary file
        script_path = f"/tmp/query_{workflow_id}.py"
        with open(script_path, "w") as f:
            f.write(query_script)
        
        # Run the query
        process = subprocess.run(
            ["python3", script_path],
            capture_output=True,
            text=True,
            timeout=60  # 1 minute timeout
        )
        
        if process.returncode == 0:
            result = json.loads(process.stdout.strip())
            return result
        else:
            raise Exception(f"Query failed: {process.stderr}")
    
    except Exception as e:
        logger.error(f"Error running RAG query: {str(e)}")
        raise

def process_documents_sync(workflow_id: str, documents: List[Dict[str, str]]):
    """Synchronous document processing"""
    try:
        # Create documents directory
        docs_dir = Path(f"/tmp/workflow_{workflow_id}/documents")
        docs_dir.mkdir(parents=True, exist_ok=True)
        
        for i, doc in enumerate(documents):
            filename = f"document_{i+1}.txt"
            filepath = docs_dir / filename
            
            with open(filepath, "w") as f:
                f.write(doc.get("content", ""))
        
        workflow_status[workflow_id]["progress"] = f"Processed {len(documents)} documents"
    
    except Exception as e:
        logger.error(f"Error processing documents: {str(e)}")
        raise

def parse_databricks_assets(stdout: str) -> Dict[str, str]:
    """Parse Databricks assets from workflow output"""
    assets = {}
    lines = stdout.split('\n')
    
    for line in lines:
        if "📁 Folder:" in line:
            assets["folder"] = line.split("📁 Folder:")[1].strip()
        elif "🔍 Endpoint:" in line:
            assets["endpoint"] = line.split("🔍 Endpoint:")[1].strip()
        elif "📓 Notebook:" in line:
            assets["notebook"] = line.split("📓 Notebook:")[1].strip()
    
    return assets
