from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import json
import asyncio
import subprocess
import tempfile
import shutil
from pathlib import Path
import logging
from datetime import datetime
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Databricks RAG Workflow API",
    description="API for creating and managing Databricks RAG workflows",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class WorkflowRequest(BaseModel):
    workflow_name: Optional[str] = None
    create_assets: bool = True
    upload_documents: bool = False
    document_description: Optional[str] = None
    documents: Optional[List[Dict[str, str]]] = None

class WorkflowResponse(BaseModel):
    workflow_id: str
    status: str
    message: str
    databricks_assets: Optional[Dict[str, str]] = None
    created_at: str

class QueryRequest(BaseModel):
    query: str
    workflow_id: str

class QueryResponse(BaseModel):
    query: str
    response: str
    context: Optional[List[str]] = None
    similarity_score: Optional[float] = None
    response_time: Optional[float] = None

class StatusResponse(BaseModel):
    workflow_id: str
    status: str
    progress: Optional[str] = None
    error: Optional[str] = None

# In-memory storage for workflow status (in production, use Redis or database)
workflow_status = {}

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Databricks RAG Workflow API is running", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "api": "running",
            "databricks_connection": "check_required"
        }
    }

@app.post("/workflows", response_model=WorkflowResponse)
async def create_workflow(request: WorkflowRequest, background_tasks: BackgroundTasks):
    """Create a new RAG workflow"""
    workflow_id = str(uuid.uuid4())
    
    # Initialize workflow status
    workflow_status[workflow_id] = {
        "status": "initializing",
        "progress": "Starting workflow creation...",
        "error": None,
        "created_at": datetime.now().isoformat()
    }
    
    # Add background task to run the workflow
    background_tasks.add_task(run_workflow, workflow_id, request)
    
    return WorkflowResponse(
        workflow_id=workflow_id,
        status="initializing",
        message="Workflow creation started",
        created_at=datetime.now().isoformat()
    )

@app.get("/workflows/{workflow_id}/status", response_model=StatusResponse)
async def get_workflow_status(workflow_id: str):
    """Get the status of a workflow"""
    if workflow_id not in workflow_status:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    status = workflow_status[workflow_id]
    return StatusResponse(
        workflow_id=workflow_id,
        status=status["status"],
        progress=status.get("progress"),
        error=status.get("error")
    )

@app.post("/workflows/{workflow_id}/query", response_model=QueryResponse)
async def query_workflow(workflow_id: str, request: QueryRequest):
    """Query a RAG workflow"""
    if workflow_id not in workflow_status:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    if workflow_status[workflow_id]["status"] != "completed":
        raise HTTPException(status_code=400, detail="Workflow not ready for querying")
    
    try:
        # Run the enhanced RAG client for querying
        result = await run_rag_query(workflow_id, request.query)
        return QueryResponse(**result)
    except Exception as e:
        logger.error(f"Error querying workflow {workflow_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@app.post("/workflows/{workflow_id}/upload")
async def upload_documents(
    workflow_id: str,
    files: List[UploadFile] = File(...),
    description: Optional[str] = Form(None)
):
    """Upload documents to a workflow"""
    if workflow_id not in workflow_status:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    try:
        # Create temporary directory for uploaded files
        temp_dir = Path(f"/tmp/workflow_{workflow_id}")
        temp_dir.mkdir(exist_ok=True)
        
        uploaded_files = []
        for file in files:
            file_path = temp_dir / file.filename
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            uploaded_files.append(str(file_path))
        
        # Update workflow status
        workflow_status[workflow_id]["progress"] = f"Uploaded {len(files)} documents"
        
        return {
            "workflow_id": workflow_id,
            "uploaded_files": uploaded_files,
            "message": f"Successfully uploaded {len(files)} documents"
        }
    except Exception as e:
        logger.error(f"Error uploading documents for workflow {workflow_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.get("/workflows")
async def list_workflows():
    """List all workflows"""
    workflows = []
    for workflow_id, status in workflow_status.items():
        workflows.append({
            "workflow_id": workflow_id,
            "status": status["status"],
            "created_at": status["created_at"],
            "progress": status.get("progress")
        })
    return {"workflows": workflows}

@app.delete("/workflows/{workflow_id}")
async def delete_workflow(workflow_id: str):
    """Delete a workflow"""
    if workflow_id not in workflow_status:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    del workflow_status[workflow_id]
    return {"message": f"Workflow {workflow_id} deleted successfully"}

async def run_workflow(workflow_id: str, request: WorkflowRequest):
    """Background task to run the workflow"""
    try:
        workflow_status[workflow_id]["status"] = "running"
        workflow_status[workflow_id]["progress"] = "Creating Databricks assets..."
        
        # Run the ultimate RAG workflow
        cmd = ["python3", "ultimate_rag_workflow.py", "--workflow-id", workflow_id]
        
        if request.workflow_name:
            cmd.extend(["--workflow-name", request.workflow_name])
        
        if not request.create_assets:
            cmd.append("--skip-assets")
        
        if request.upload_documents and request.documents:
            # Handle document upload
            workflow_status[workflow_id]["progress"] = "Processing documents..."
            await process_documents(workflow_id, request.documents)
        
        # Run the workflow
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        stdout, stderr = process.communicate()
        
        if process.returncode == 0:
            workflow_status[workflow_id]["status"] = "completed"
            workflow_status[workflow_id]["progress"] = "Workflow completed successfully"
            workflow_status[workflow_id]["databricks_assets"] = parse_databricks_assets(stdout)
        else:
            workflow_status[workflow_id]["status"] = "failed"
            workflow_status[workflow_id]["error"] = stderr or "Unknown error occurred"
            logger.error(f"Workflow {workflow_id} failed: {stderr}")
    
    except Exception as e:
        workflow_status[workflow_id]["status"] = "failed"
        workflow_status[workflow_id]["error"] = str(e)
        logger.error(f"Error running workflow {workflow_id}: {str(e)}")

async def run_rag_query(workflow_id: str, query: str) -> Dict[str, Any]:
    """Run a RAG query"""
    try:
        # Create a temporary script to run the query
        query_script = f"""
import sys
sys.path.append('.')
from enhanced_rag_client import EnhancedRAGClient, EnhancedRAGConfig
import asyncio

async def run_query():
    config = EnhancedRAGConfig(
        openai_api_key="{os.getenv('OPENAI_API_KEY')}",
        databricks_host="{os.getenv('DATABRICKS_HOST')}",
        databricks_token="{os.getenv('DATABRICKS_TOKEN')}",
        workflow_id="{workflow_id}"
    )
    
    client = EnhancedRAGClient(config)
    result = await client.query_real_documents("{query}")
    return result

if __name__ == "__main__":
    result = asyncio.run(run_query())
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
            text=True
        )
        
        if process.returncode == 0:
            result = json.loads(process.stdout.strip())
            return result
        else:
            raise Exception(f"Query failed: {process.stderr}")
    
    except Exception as e:
        logger.error(f"Error running RAG query: {str(e)}")
        raise

async def process_documents(workflow_id: str, documents: List[Dict[str, str]]):
    """Process documents for a workflow"""
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
