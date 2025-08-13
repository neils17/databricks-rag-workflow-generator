"""
Working Example Usage of Databricks RAG System
"""

from working_rag_client import WorkingDatabricksRAGClient, WorkingRAGConfig
from working_rag_config import working_databricks_config, working_rag_config
import os

def main():
    """Working example usage of the RAG system"""
    
    print("🚀 Starting Working RAG System")
    print("=" * 50)
    
    # Initialize configuration
    config = WorkingRAGConfig(
        workflow_id=working_rag_config.workflow_id,
        vector_search_endpoint=working_databricks_config.vector_search_endpoint,
        embedding_model=working_rag_config.embedding_model,
        chunk_size=working_rag_config.chunk_size,
        chunk_overlap=working_rag_config.chunk_overlap,
        top_k=working_rag_config.top_k
    )
    
    # Create RAG client
    client = WorkingDatabricksRAGClient(config)
    
    # Check asset status
    print("\n📊 Checking Databricks assets...")
    status = client.get_asset_status()
    print("Asset Status:", status)
    
    # Example: Process documents and perform query
    print("\n📄 Processing sample documents...")
    sample_docs = [
        "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions without being explicitly programmed. It uses algorithms and statistical models to analyze and draw inferences from patterns in data.",
        "Data science combines statistics, programming, and domain expertise to extract insights from data. It involves collecting, cleaning, analyzing, and interpreting data to support decision-making processes.",
        "Natural language processing (NLP) is a field of AI that focuses on the interaction between computers and human language. It enables machines to understand, interpret, and generate human language in a meaningful way."
    ]
    
    processed_docs = client.process_documents(sample_docs)
    print(f"✅ Processed {len(processed_docs)} document chunks")
    
    # Example: Perform RAG query with full implementation
    print("\n🔍 Performing RAG query with full implementation...")
    query = "What is machine learning and how does it relate to AI?"
    result = client.query(query, sample_docs)
    print(f"Query: {query}")
    print(f"Response: {result}")
    
    print("\n🎉 Working production system ready!")
    print("\n💡 Next steps:")
    print("1. Visit your Databricks workspace to see created assets")
    print("2. Upload your own documents to the system")
    print("3. Use Vector Search endpoint when it's ready")

if __name__ == "__main__":
    main()
