"""
Natural Language Document Uploader for Databricks RAG System
Upload documents to Databricks using plain English descriptions
"""

import os
import json
import re
import glob
from typing import List, Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv
import mlflow
import time
from datetime import datetime
import requests
import base64

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

class NaturalLanguageDocumentUploader:
    """Upload documents to Databricks using natural language descriptions"""
    
    def __init__(self):
        self.host = os.getenv("DATABRICKS_HOST")
        self.token = os.getenv("DATABRICKS_TOKEN")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        # Initialize LLM for natural language processing
        self.llm = ChatOpenAI(
            model="gpt-4-turbo-preview",
            temperature=0.1,
            openai_api_key=self.openai_api_key
        )
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=self.openai_api_key
        )
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        # Databricks API headers
        self.api_headers = {
            'Authorization': f'Bearer {self.token}',
            'Content-Type': 'application/json'
        }
        
        # Setup MLflow
        self.experiment_id = self._setup_mlflow()
        
        print("🚀 Natural Language Document Uploader initialized")
        if self.experiment_id:
            print(f"📊 MLflow tracking enabled - Experiment ID: {self.experiment_id}")
    
    def _setup_mlflow(self) -> Optional[str]:
        """Setup MLflow experiment for tracking"""
        try:
            mlflow.set_tracking_uri("databricks")
            experiment_name = f"/Shared/Natural_Language_Upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(experiment_name)
                print(f"✅ Created new MLflow experiment: {experiment_name}")
            else:
                experiment_id = experiment.experiment_id
                print(f"✅ Using existing MLflow experiment: {experiment_name}")
            
            mlflow.set_experiment(experiment_name)
            return experiment_id
        except Exception as e:
            print(f"⚠️  MLflow setup failed: {str(e)}")
            return None
    
    def understand_document_request(self, description: str) -> Dict[str, Any]:
        """Use LLM to understand what documents the user wants"""
        
        prompt = f"""
        Analyze this document request and extract the key information:
        
        User Request: "{description}"
        
        Please provide a JSON response with the following structure:
        {{
            "document_types": ["list of document types mentioned"],
            "file_extensions": ["list of file extensions to look for"],
            "keywords": ["list of keywords to search for"],
            "locations": ["list of possible file locations"],
            "content_themes": ["list of content themes"],
            "search_strategy": "description of how to find these documents"
        }}
        
        Examples:
        - "Upload my company policies" → {{"document_types": ["policy", "handbook"], "file_extensions": [".pdf", ".doc", ".txt"], "keywords": ["policy", "handbook", "employee"], "locations": ["current directory", "documents folder"], "content_themes": ["company policies", "employee guidelines"]}}
        - "Find technical documentation" → {{"document_types": ["documentation", "manual"], "file_extensions": [".md", ".txt", ".pdf"], "keywords": ["technical", "documentation", "api", "guide"], "locations": ["current directory", "docs folder"], "content_themes": ["technical documentation", "API guides"]}}
        
        JSON Response:
        """
        
        try:
            response = self.llm.invoke(prompt)
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return self._fallback_analysis(description)
        except Exception as e:
            print(f"⚠️  LLM analysis failed: {str(e)}")
            return self._fallback_analysis(description)
    
    def _fallback_analysis(self, description: str) -> Dict[str, Any]:
        """Fallback analysis if LLM fails"""
        description_lower = description.lower()
        
        # Simple keyword-based analysis
        document_types = []
        file_extensions = [".txt", ".md", ".pdf", ".doc", ".docx"]
        keywords = []
        locations = ["."]
        content_themes = []
        
        # Check for PDF-specific requests
        if any(word in description_lower for word in ["pdf", "pdfs", "pdf files"]):
            document_types.extend(["pdf", "document"])
            file_extensions = [".pdf"]  # Focus only on PDFs
            keywords = []  # No keyword filtering for PDFs
            content_themes.append("PDF documents")
        
        # Check for specific file types
        elif any(word in description_lower for word in ["txt", "text", "text files"]):
            document_types.extend(["text", "document"])
            file_extensions = [".txt"]
            keywords = []
            content_themes.append("Text documents")
        
        elif any(word in description_lower for word in ["md", "markdown", "markdown files"]):
            document_types.extend(["markdown", "documentation"])
            file_extensions = [".md"]
            keywords = []
            content_themes.append("Markdown documentation")
        
        # Check for content-based requests
        elif any(word in description_lower for word in ["policy", "policies", "handbook"]):
            document_types.extend(["policy", "handbook"])
            keywords.extend(["policy", "handbook", "employee", "company"])
            content_themes.append("company policies")
        
        elif any(word in description_lower for word in ["technical", "documentation", "docs", "api"]):
            document_types.extend(["documentation", "technical"])
            keywords.extend(["technical", "documentation", "api", "guide"])
            content_themes.append("technical documentation")
        
        elif any(word in description_lower for word in ["manual", "guide", "tutorial"]):
            document_types.extend(["manual", "guide"])
            keywords.extend(["manual", "guide", "tutorial", "how-to"])
            content_themes.append("user guides")
        
        # Check for location-specific requests
        if any(word in description_lower for word in ["folder", "directory", "subdirectory", "subfolder"]):
            locations = ["."]  # Search current directory and all subdirectories
        
        return {
            "document_types": document_types,
            "file_extensions": file_extensions,
            "keywords": keywords,
            "locations": locations,
            "content_themes": content_themes,
            "search_strategy": "Search current directory and subdirectories for relevant files"
        }
    
    def find_documents(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find documents based on the analysis"""
        found_documents = []
        
        print(f"🔍 Searching for documents...")
        print(f"   Types: {', '.join(analysis['document_types'])}")
        print(f"   Extensions: {', '.join(analysis['file_extensions'])}")
        print(f"   Keywords: {', '.join(analysis['keywords'])}")
        
        # Search in specified locations
        for location in analysis['locations']:
            if location == ".":
                search_path = Path.cwd()
            else:
                search_path = Path(location)
            
            if not search_path.exists():
                print(f"⚠️  Location does not exist: {location}")
                continue
            
            print(f"🔍 Searching in: {search_path}")
            
            # Search for files with specified extensions
            for extension in analysis['file_extensions']:
                pattern = f"**/*{extension}"
                files = list(search_path.glob(pattern))
                
                print(f"   Looking for {extension} files...")
                
                for file_path in files:
                    try:
                        # Skip files in venv directory
                        if 'venv' in str(file_path):
                            continue
                            
                        # Read file content
                        if file_path.suffix.lower() in ['.txt', '.md', '.py', '.js', '.html', '.css', '.json', '.xml', '.csv']:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                        elif file_path.suffix.lower() == '.pdf':
                            # For PDF files, create a placeholder content for now
                            # In a production system, you'd use PyPDF2 or similar to extract text
                            content = f"PDF Document: {file_path.name}\n\nThis is a PDF file that would be processed with a PDF text extractor in a production system.\n\nFile path: {file_path}\nFile size: {file_path.stat().st_size} bytes"
                        else:
                            # For other binary files, skip for now
                            continue
                        
                        # Check if content matches keywords
                        content_lower = content.lower()
                        keyword_matches = sum(1 for keyword in analysis['keywords'] if keyword.lower() in content_lower)
                        
                        # For PDFs or when no keywords specified, include all files
                        if keyword_matches > 0 or not analysis['keywords'] or file_path.suffix.lower() == '.pdf':
                            found_documents.append({
                                "title": file_path.name,
                                "source": str(file_path.relative_to(Path.cwd())),
                                "content": content,
                                "file_path": str(file_path),
                                "keyword_matches": keyword_matches,
                                "file_size": len(content)
                            })
                            print(f"✅ Found: {file_path.name} ({keyword_matches} keyword matches)")
                    
                    except Exception as e:
                        print(f"❌ Error reading {file_path.name}: {str(e)}")
        
        # Sort by keyword matches
        found_documents.sort(key=lambda x: x['keyword_matches'], reverse=True)
        
        print(f"📚 Found {len(found_documents)} relevant documents")
        return found_documents
    
    def upload_to_databricks(self, documents: List[Dict[str, Any]], description: str) -> Dict[str, Any]:
        """Upload documents to Databricks with MLflow tracking"""
        
        if not self.experiment_id:
            return {"success": False, "error": "MLflow tracking not available"}
        
        with mlflow.start_run(experiment_id=self.experiment_id) as run:
            start_time = time.time()
            
            try:
                # Log parameters
                mlflow.log_param("upload_description", description)
                mlflow.log_param("num_documents", len(documents))
                mlflow.log_param("total_content_size", sum(doc['file_size'] for doc in documents))
                
                # Generate unique folder name
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                folder_name = f"Natural_Language_Upload_{timestamp}"
                folder_path = f"/Shared/{folder_name}"
                
                # Create folder in Databricks
                print(f"📁 Creating folder: {folder_path}")
                folder_result = self._create_databricks_folder(folder_path)
                
                if not folder_result["success"]:
                    return {"success": False, "error": f"Failed to create folder: {folder_result['error']}"}
                
                # Process documents and create embeddings
                print("🔄 Processing documents...")
                processed_docs = []
                embeddings_data = []
                
                for i, doc in enumerate(documents):
                    # Split into chunks
                    chunks = self.text_splitter.split_text(doc['content'])
                    
                    for j, chunk in enumerate(chunks):
                        chunk_id = f"{doc['title']}_{j}"
                        embedding = self.embeddings.embed_query(chunk)
                        
                        processed_docs.append({
                            'id': chunk_id,
                            'content': chunk,
                            'title': doc['title'],
                            'source': doc['source'],
                            'chunk_index': j,
                            'document_index': i
                        })
                        
                        embeddings_data.append({
                            'id': chunk_id,
                            'embedding': embedding,
                            'content': chunk,
                            'title': doc['title'],
                            'source': doc['source']
                        })
                
                # Create RAG notebook with the uploaded documents
                notebook_path = f"{folder_path}/RAG_System"
                notebook_content = self._generate_rag_notebook_content(
                    timestamp, 
                    processed_docs, 
                    description
                )
                
                print(f"📓 Creating RAG notebook: {notebook_path}")
                notebook_result = self._create_databricks_notebook(notebook_path, notebook_content)
                
                if not notebook_result["success"]:
                    return {"success": False, "error": f"Failed to create notebook: {notebook_result['error']}"}
                
                # Create document index notebook
                index_notebook_path = f"{folder_path}/Document_Index"
                index_notebook_content = self._generate_index_notebook_content(
                    timestamp, 
                    documents, 
                    processed_docs
                )
                
                print(f"📊 Creating document index notebook: {index_notebook_path}")
                index_notebook_result = self._create_databricks_notebook(index_notebook_path, index_notebook_content)
                
                # Calculate metrics
                end_time = time.time()
                upload_time = end_time - start_time
                
                # Log metrics
                mlflow.log_metric("upload_time_seconds", upload_time)
                mlflow.log_metric("num_processed_chunks", len(processed_docs))
                mlflow.log_metric("total_documents", len(documents))
                mlflow.log_metric("total_content_size", sum(doc['file_size'] for doc in documents))
                
                # Log artifacts
                mlflow.log_text(description, "upload_description.txt")
                mlflow.log_text(json.dumps([doc['title'] for doc in documents], indent=2), "uploaded_documents.json")
                
                # Log run info
                mlflow.set_tag("upload_type", "natural_language")
                mlflow.set_tag("folder_path", folder_path)
                mlflow.set_tag("generation_timestamp", datetime.now().isoformat())
                
                return {
                    "success": True,
                    "run_id": run.info.run_id,
                    "experiment_id": self.experiment_id,
                    "folder_path": folder_path,
                    "notebook_path": notebook_path,
                    "index_notebook_path": index_notebook_path,
                    "metrics": {
                        "upload_time_seconds": upload_time,
                        "num_documents": len(documents),
                        "num_processed_chunks": len(processed_docs),
                        "total_content_size": sum(doc['file_size'] for doc in documents)
                    },
                    "documents": [doc['title'] for doc in documents]
                }
                
            except Exception as e:
                mlflow.log_param("error", str(e))
                mlflow.set_tag("status", "failed")
                
                return {
                    "success": False,
                    "error": str(e),
                    "run_id": run.info.run_id,
                    "experiment_id": self.experiment_id
                }
    
    def _create_databricks_folder(self, path: str) -> Dict[str, Any]:
        """Create a folder in Databricks workspace"""
        url = f"{self.host}/api/2.0/workspace/mkdirs"
        data = {"path": path}
        
        response = requests.post(url, headers=self.api_headers, json=data)
        if response.status_code == 200:
            return {"success": True, "data": response.json()}
        else:
            return {"success": False, "error": response.text, "status": response.status_code}
    
    def _create_databricks_notebook(self, path: str, content: str) -> Dict[str, Any]:
        """Create a notebook in Databricks workspace"""
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
        
        response = requests.post(url, headers=self.api_headers, json=data)
        if response.status_code == 200:
            return {"success": True, "data": response.json()}
        else:
            return {"success": False, "error": response.text, "status": response.status_code}
    
    def _generate_rag_notebook_content(self, timestamp: str, processed_docs: List[Dict], description: str) -> str:
        """Generate RAG notebook content"""
        
        return f'''# Natural Language Upload RAG System
# Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
# Upload Description: {description}

# COMMAND ----------

# Install required packages
# MAGIC %pip install langchain langchain-openai mlflow

# COMMAND ----------

import mlflow
import json
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
import time
from datetime import datetime

# COMMAND ----------

# Configuration
UPLOAD_TIMESTAMP = "{timestamp}"
UPLOAD_DESCRIPTION = "{description}"
NUM_DOCUMENTS = {len(processed_docs)}
EXPERIMENT_NAME = f"/Shared/Natural_Language_Upload_{{UPLOAD_TIMESTAMP}}"

# COMMAND ----------

# Setup MLflow
mlflow.set_tracking_uri("databricks")
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
if experiment is None:
    experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
else:
    experiment_id = experiment.experiment_id

mlflow.set_experiment(EXPERIMENT_NAME)

# COMMAND ----------

# Initialize components
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0.1)

# COMMAND ----------

# Load processed documents
processed_documents = {json.dumps(processed_docs, indent=2)}

# COMMAND ----------

def query_documents(query: str):
    """Query the uploaded documents"""
    with mlflow.start_run(experiment_id=experiment_id) as run:
        start_time = time.time()
        
        try:
            # Log parameters
            mlflow.log_param("query", query)
            mlflow.log_param("upload_description", UPLOAD_DESCRIPTION)
            mlflow.log_param("num_documents", NUM_DOCUMENTS)
            
            # Create query embedding
            query_embedding = embeddings.embed_query(query)
            
            # Vector similarity search
            similarities = []
            for doc in processed_documents:
                doc_embedding = embeddings.embed_query(doc['content'])
                similarity = sum(a * b for a, b in zip(query_embedding, doc_embedding))
                similarities.append((similarity, doc))
            
            # Get top 5 most similar
            similarities.sort(key=lambda x: x[0], reverse=True)
            top_docs = [doc for _, doc in similarities[:5]]
            
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
            mlflow.set_tag("upload_timestamp", UPLOAD_TIMESTAMP)
            mlflow.set_tag("upload_description", UPLOAD_DESCRIPTION)
            mlflow.set_tag("query_type", "natural_language_upload")
            
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

# Example usage
print("🚀 Natural Language Upload RAG System Ready!")
print(f"📚 Upload Description: {{UPLOAD_DESCRIPTION}}")
print(f"📄 Number of Documents: {{NUM_DOCUMENTS}}")
print(f"📊 MLflow Experiment: {{EXPERIMENT_NAME}}")

# Test query
test_query = "What documents were uploaded?"
result = query_documents(test_query)

if result["success"]:
    print(f"\\n✅ Test Query Result: {{result['response']}}")
    print(f"⏱️  Response Time: {{result['response_time']:.2f}}s")
else:
    print(f"❌ Test Query Failed: {{result['error']}}")

print("\\n💡 Use query_documents('your question') to ask questions about your uploaded documents!")
'''
    
    def _generate_index_notebook_content(self, timestamp: str, documents: List[Dict], processed_docs: List[Dict]) -> str:
        """Generate document index notebook content"""
        
        return f'''# Document Index - Natural Language Upload
# Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

# COMMAND ----------

import json
import pandas as pd

# COMMAND ----------

# Document Index Information
UPLOAD_TIMESTAMP = "{timestamp}"
NUM_ORIGINAL_DOCUMENTS = {len(documents)}
NUM_PROCESSED_CHUNKS = {len(processed_docs)}

# COMMAND ----------

# Original Documents
original_documents = {json.dumps([{{"title": doc["title"], "source": doc["source"], "file_size": doc["file_size"]}} for doc in documents], indent=2)}

print("📚 Original Documents Uploaded:")
for i, doc in enumerate(original_documents, 1):
    print(f"  {{i}}. {{doc['title']}} ({{doc['source']}}) - {{doc['file_size']}} characters")

# COMMAND ----------

# Processed Chunks
processed_chunks = {json.dumps([{{"id": doc["id"], "title": doc["title"], "chunk_index": doc["chunk_index"]}} for doc in processed_docs], indent=2)}

print(f"\\n📄 Document Processing Summary:")
print(f"  Original Documents: {{NUM_ORIGINAL_DOCUMENTS}}")
print(f"  Processed Chunks: {{NUM_PROCESSED_CHUNKS}}")
print(f"  Average Chunks per Document: {{NUM_PROCESSED_CHUNKS / NUM_ORIGINAL_DOCUMENTS:.1f}}")

# COMMAND ----------

# Create summary DataFrame
summary_data = []
for doc in original_documents:
    doc_chunks = [chunk for chunk in processed_chunks if chunk['title'] == doc['title']]
    summary_data.append({{
        'Title': doc['title'],
        'Source': doc['source'],
        'File Size (chars)': doc['file_size'],
        'Number of Chunks': len(doc_chunks)
    }})

summary_df = pd.DataFrame(summary_data)
display(summary_df)

# COMMAND ----------

print("\\n✅ Document index created successfully!")
print("\\n💡 This notebook shows the structure of your uploaded documents.")
print("   Use the RAG_System notebook to query your documents.")
'''

def main():
    """Main function for natural language document upload"""
    
    print("🚀 Natural Language Document Uploader")
    print("=" * 50)
    print("Describe what documents you want to upload in plain English!")
    print("Examples:")
    print("- 'Upload my company policies and employee handbook'")
    print("- 'Find all technical documentation and API guides'")
    print("- 'Upload my project documentation and README files'")
    print("- 'Find all PDF files in my documents folder'")
    print()
    
    # Initialize uploader
    uploader = NaturalLanguageDocumentUploader()
    
    while True:
        try:
            # Get user description
            description = input("📝 Describe the documents you want to upload: ").strip()
            
            if description.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not description:
                continue
            
            print(f"\n🔍 Understanding your request: '{description}'")
            
            # Analyze the request
            analysis = uploader.understand_document_request(description)
            print(f"✅ Analysis complete: {analysis['search_strategy']}")
            
            # Find documents
            documents = uploader.find_documents(analysis)
            
            if not documents:
                print("❌ No relevant documents found. Try a different description.")
                continue
            
            # Show found documents
            print(f"\n📚 Found {len(documents)} documents:")
            for i, doc in enumerate(documents[:5], 1):  # Show first 5
                print(f"  {i}. {doc['title']} ({doc['keyword_matches']} keyword matches)")
            
            if len(documents) > 5:
                print(f"  ... and {len(documents) - 5} more")
            
            # Confirm upload
            confirm = input(f"\n❓ Upload {len(documents)} documents to Databricks? (y/n): ").strip().lower()
            
            if confirm == 'y':
                print("\n🚀 Uploading to Databricks...")
                result = uploader.upload_to_databricks(documents, description)
                
                if result["success"]:
                    print(f"\n✅ Upload successful!")
                    print(f"📁 Folder: {result['folder_path']}")
                    print(f"📓 RAG Notebook: {result['notebook_path']}")
                    print(f"📊 Index Notebook: {result['index_notebook_path']}")
                    print(f"⏱️  Upload Time: {result['metrics']['upload_time_seconds']:.2f}s")
                    print(f"📄 Documents: {result['metrics']['num_documents']}")
                    print(f"🔢 Chunks: {result['metrics']['num_processed_chunks']}")
                    
                    if result.get('run_id'):
                        print(f"🔗 MLflow Run ID: {result['run_id']}")
                        print(f"📊 View in MLflow: {uploader.host}/ml/experiments/{result['experiment_id']}")
                    
                    print(f"\n🌐 View in Databricks: {uploader.host}/workspace{result['folder_path']}")
                    
                else:
                    print(f"❌ Upload failed: {result['error']}")
            
            print("\n" + "="*50)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Unexpected error: {str(e)}")

if __name__ == "__main__":
    main()
