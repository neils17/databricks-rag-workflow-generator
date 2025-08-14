"""
Enhanced RAG Client for Real Documents and Vector Index in Databricks
This version supports real documents, vector indexing, and interactive queries
"""

import os
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from dotenv import load_dotenv
import mlflow
import time
from datetime import datetime
import requests
import base64

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Load environment variables
load_dotenv()

@dataclass
class EnhancedRAGConfig:
    """Configuration for enhanced RAG system with real documents"""
    workflow_id: str = "20250813_192821"
    vector_search_endpoint: str = "rag_endpoint_20250813_192821"
    embedding_model: str = "text-embedding-3-small"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 5
    enable_mlflow: bool = True
    experiment_name: str = "RAG_Workflow_20250813_192821"
    index_name: str = "document_index"
    catalog_name: str = "main"
    schema_name: str = "default"

class EnhancedDatabricksRAGClient:
    """Enhanced client for real documents with Databricks Vector Search"""
    
    def __init__(self, config: EnhancedRAGConfig):
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
        
        # Initialize Databricks API client
        self.api_headers = {
            'Authorization': f'Bearer {self.token}',
            'Content-Type': 'application/json'
        }
        
        print(f"Enhanced RAG Client initialized for workflow: {config.workflow_id}")
        if self.experiment_id:
            print(f"📊 MLflow tracking enabled - Experiment ID: {self.experiment_id}")
    
    def _setup_mlflow(self) -> Optional[str]:
        """Setup MLflow experiment for tracking"""
        try:
            # Set tracking URI to Databricks
            mlflow.set_tracking_uri("databricks")
            
            # Create experiment name with proper Databricks path format
            experiment_name = f"/Shared/RAG_Workflow_{self.config.workflow_id}"
            
            # Create or get experiment
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(experiment_name)
                print(f"✅ Created new MLflow experiment: {experiment_name}")
            else:
                experiment_id = experiment.experiment_id
                print(f"✅ Using existing MLflow experiment: {experiment_name}")
            
            # Set the experiment
            mlflow.set_experiment(experiment_name)
            
            return experiment_id
        except Exception as e:
            print(f"⚠️  MLflow setup failed: {str(e)}")
            return None
    
    def create_vector_index(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a vector index in Databricks with real documents"""
        try:
            print(f"🔍 Creating vector index: {self.config.index_name}")
            
            # Process documents into chunks
            processed_docs = []
            for i, doc in enumerate(documents):
                doc_text = doc.get('content', '')
                doc_title = doc.get('title', f'document_{i}')
                doc_source = doc.get('source', 'unknown')
                
                # Split into chunks
                chunks = self.text_splitter.split_text(doc_text)
                for j, chunk in enumerate(chunks):
                    processed_docs.append({
                        'id': f"{doc_title}_{j}",
                        'content': chunk,
                        'title': doc_title,
                        'source': doc_source,
                        'chunk_index': j,
                        'document_index': i
                    })
            
            print(f"✅ Processed {len(processed_docs)} chunks from {len(documents)} documents")
            
            # Create embeddings for all chunks
            print("🔄 Creating embeddings...")
            embeddings_data = []
            for doc in processed_docs:
                embedding = self.embeddings.embed_query(doc['content'])
                embeddings_data.append({
                    'id': doc['id'],
                    'embedding': embedding,
                    'content': doc['content'],
                    'title': doc['title'],
                    'source': doc['source'],
                    'chunk_index': doc['chunk_index'],
                    'document_index': doc['document_index']
                })
            
            print(f"✅ Created embeddings for {len(embeddings_data)} chunks")
            
            # Store in Databricks (simplified - in production you'd use Vector Search API)
            # For now, we'll store the processed data for local search
            self.processed_documents = embeddings_data
            
            return {
                "success": True,
                "index_name": self.config.index_name,
                "num_documents": len(documents),
                "num_chunks": len(processed_docs),
                "message": f"Vector index created with {len(processed_docs)} chunks"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def query_real_documents(self, query: str) -> Dict[str, Any]:
        """Query real documents using vector similarity"""
        if not hasattr(self, 'processed_documents'):
            return {
                "success": False,
                "error": "No documents indexed. Please create vector index first."
            }
        
        if not self.experiment_id:
            print("⚠️  MLflow tracking not available")
            return self._query_without_tracking(query)
        
        # Start MLflow run
        with mlflow.start_run(experiment_id=self.experiment_id) as run:
            start_time = time.time()
            
            try:
                # Log parameters
                mlflow.log_param("query", query)
                mlflow.log_param("num_indexed_chunks", len(self.processed_documents))
                mlflow.log_param("top_k", self.config.top_k)
                mlflow.log_param("workflow_id", self.config.workflow_id)
                mlflow.log_param("vector_search_endpoint", self.config.vector_search_endpoint)
                mlflow.log_param("embedding_model", self.config.embedding_model)
                mlflow.log_param("llm_model", "gpt-4-turbo-preview")
                mlflow.log_param("chunk_size", self.config.chunk_size)
                mlflow.log_param("chunk_overlap", self.config.chunk_overlap)
                
                # Create embedding for query
                query_embedding = self.embeddings.embed_query(query)
                mlflow.log_metric("num_indexed_chunks", len(self.processed_documents))
                
                # Vector similarity search
                similarities = []
                for doc in self.processed_documents:
                    # Calculate cosine similarity
                    similarity = sum(a * b for a, b in zip(query_embedding, doc['embedding']))
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
                context_parts = []
                for i, doc in enumerate(top_docs):
                    context_parts.append(f"Document {i+1} ({doc['title']}):\n{doc['content']}")
                
                context = "\n\n".join(context_parts)
                mlflow.log_metric("context_length", len(context))
                
                # Create prompt for LLM
                prompt = f"""Based on the following context from your documents, answer the question. If the context doesn't contain enough information to answer the question, say so.

Context:
{context}

Question: {query}

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
                mlflow.log_metric("context_utilization_ratio", len(context) / sum(len(doc['content']) for doc in self.processed_documents))
                
                # Log artifacts
                mlflow.log_text(prompt, "prompt.txt")
                mlflow.log_text(response.content, "response.txt")
                mlflow.log_text(context, "retrieved_context.txt")
                
                # Log run info
                mlflow.set_tag("workflow_id", self.config.workflow_id)
                mlflow.set_tag("vector_search_endpoint", self.config.vector_search_endpoint)
                mlflow.set_tag("query_type", "real_document_query")
                mlflow.set_tag("generation_timestamp", datetime.now().isoformat())
                
                return {
                    "success": True,
                    "response": response.content,
                    "run_id": run.info.run_id,
                    "experiment_id": self.experiment_id,
                    "metrics": {
                        "response_time_seconds": response_time,
                        "response_length": response_length,
                        "num_indexed_chunks": len(self.processed_documents),
                        "context_length": len(context)
                    },
                    "retrieved_documents": [
                        {
                            "title": doc['title'],
                            "source": doc['source'],
                            "content_preview": doc['content'][:200] + "...",
                            "similarity_score": similarities[i][0]
                        }
                        for i, doc in enumerate(top_docs)
                    ]
                }
                
            except Exception as e:
                # Log error
                mlflow.log_param("error", str(e))
                mlflow.set_tag("status", "failed")
                
                return {
                    "success": False,
                    "error": str(e),
                    "run_id": run.info.run_id,
                    "experiment_id": self.experiment_id
                }
    
    def _query_without_tracking(self, query: str) -> Dict[str, Any]:
        """Query without MLflow tracking"""
        try:
            # Create embedding for query
            query_embedding = self.embeddings.embed_query(query)
            
            # Vector similarity search
            similarities = []
            for doc in self.processed_documents:
                similarity = sum(a * b for a, b in zip(query_embedding, doc['embedding']))
                similarities.append((similarity, doc))
            
            # Sort by similarity and get top k
            similarities.sort(key=lambda x: x[0], reverse=True)
            top_docs = [doc for _, doc in similarities[:self.config.top_k]]
            
            # Prepare context
            context_parts = []
            for i, doc in enumerate(top_docs):
                context_parts.append(f"Document {i+1} ({doc['title']}):\n{doc['content']}")
            
            context = "\n\n".join(context_parts)
            
            # Create prompt for LLM
            prompt = f"""Based on the following context from your documents, answer the question. If the context doesn't contain enough information to answer the question, say so.

Context:
{context}

Question: {query}

Answer:"""
            
            # Generate response using LLM
            response = self.llm.invoke(prompt)
            
            return {
                "success": True,
                "response": response.content,
                "retrieved_documents": [
                    {
                        "title": doc['title'],
                        "source": doc['source'],
                        "content_preview": doc['content'][:200] + "...",
                        "similarity_score": similarities[i][0]
                    }
                    for i, doc in enumerate(top_docs)
                ]
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_index_status(self) -> Dict[str, Any]:
        """Get status of the vector index"""
        try:
            status = {
                "workflow_id": self.config.workflow_id,
                "index_name": self.config.index_name,
                "mlflow_experiment_id": self.experiment_id,
                "mlflow_experiment_name": self.config.experiment_name,
                "has_indexed_documents": hasattr(self, 'processed_documents'),
                "num_indexed_chunks": len(self.processed_documents) if hasattr(self, 'processed_documents') else 0
            }
            return status
        except Exception as e:
            return {"error": str(e)}

# Example usage with real documents
def main():
    """Example usage with real documents"""
    
    print("🚀 Enhanced RAG System with Real Documents")
    print("=" * 60)
    
    # Initialize configuration
    config = EnhancedRAGConfig()
    client = EnhancedDatabricksRAGClient(config)
    
    # Check status
    print("\n📊 Checking system status...")
    status = client.get_index_status()
    print("Status:", status)
    
    # Example real documents (you can replace these with your own)
    real_documents = [
        {
            "title": "Company Policy Document",
            "source": "HR Department",
            "content": """
            Employee Benefits Policy
            
            All full-time employees are entitled to the following benefits:
            1. Health insurance coverage for employee and dependents
            2. 401(k) retirement plan with company matching up to 6%
            3. Paid time off: 20 days per year
            4. Professional development budget of $2,000 annually
            5. Flexible work arrangements including remote work options
            
            Vacation Policy:
            - Employees must submit vacation requests 2 weeks in advance
            - Maximum 10 consecutive days without special approval
            - Unused vacation days roll over to next year (max 5 days)
            
            Sick Leave:
            - 10 sick days per year
            - Doctor's note required for absences longer than 3 days
            - Sick days do not roll over to next year
            """
        },
        {
            "title": "Technical Documentation",
            "source": "Engineering Team",
            "content": """
            API Integration Guide
            
            Our REST API supports the following endpoints:
            
            Authentication:
            - POST /auth/login - Obtain access token
            - POST /auth/refresh - Refresh expired token
            
            User Management:
            - GET /users - List all users
            - POST /users - Create new user
            - PUT /users/{id} - Update user
            - DELETE /users/{id} - Delete user
            
            Data Operations:
            - GET /data - Retrieve data with pagination
            - POST /data - Create new data record
            - PUT /data/{id} - Update data record
            - DELETE /data/{id} - Delete data record
            
            Rate Limiting:
            - 1000 requests per hour per API key
            - 100 requests per minute per endpoint
            - Exceeding limits returns 429 status code
            
            Error Handling:
            - 400: Bad Request - Invalid parameters
            - 401: Unauthorized - Invalid or missing token
            - 403: Forbidden - Insufficient permissions
            - 404: Not Found - Resource doesn't exist
            - 500: Internal Server Error - Server issue
            """
        },
        {
            "title": "Product Manual",
            "source": "Product Team",
            "content": """
            Product Features and Specifications
            
            Core Features:
            1. Real-time data processing
            2. Advanced analytics dashboard
            3. Custom reporting tools
            4. Multi-user collaboration
            5. Mobile app support
            
            System Requirements:
            - Operating System: Windows 10+, macOS 10.15+, Linux Ubuntu 18.04+
            - RAM: Minimum 8GB, Recommended 16GB
            - Storage: 50GB available space
            - Network: Broadband internet connection
            
            Installation Process:
            1. Download installer from official website
            2. Run installer as administrator
            3. Follow setup wizard instructions
            4. Configure initial settings
            5. Import existing data (optional)
            
            Troubleshooting:
            - If installation fails, check system requirements
            - Clear browser cache if web interface doesn't load
            - Restart application if performance issues occur
            - Contact support for database connection errors
            """
        }
    ]
    
    # Create vector index with real documents
    print("\n📚 Creating vector index with real documents...")
    index_result = client.create_vector_index(real_documents)
    
    if index_result["success"]:
        print(f"✅ {index_result['message']}")
        
        # Interactive query loop
        print("\n🔍 Interactive Query Mode")
        print("Type your questions about the documents (type 'quit' to exit):")
        print("-" * 50)
        
        while True:
            try:
                query = input("\n❓ Your question: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not query:
                    continue
                
                print("🔍 Searching documents...")
                result = client.query_real_documents(query)
                
                if result["success"]:
                    print(f"\n✅ Answer: {result['response']}")
                    
                    if "retrieved_documents" in result:
                        print(f"\n📄 Retrieved Documents:")
                        for i, doc in enumerate(result["retrieved_documents"], 1):
                            print(f"  {i}. {doc['title']} (Score: {doc['similarity_score']:.3f})")
                            print(f"     Source: {doc['source']}")
                            print(f"     Preview: {doc['content_preview']}")
                    
                    if "metrics" in result:
                        print(f"\n📊 Metrics:")
                        print(f"  Response Time: {result['metrics']['response_time_seconds']:.2f}s")
                        print(f"  Response Length: {result['metrics']['response_length']} characters")
                        print(f"  Context Length: {result['metrics']['context_length']} characters")
                    
                    if "run_id" in result:
                        print(f"\n🔗 MLflow Run ID: {result['run_id']}")
                        
                else:
                    print(f"❌ Error: {result['error']}")
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Unexpected error: {str(e)}")
    
    else:
        print(f"❌ Failed to create vector index: {index_result['error']}")

if __name__ == "__main__":
    main()
