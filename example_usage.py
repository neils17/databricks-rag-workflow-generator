"""
Example Usage of Databricks RAG System (Production Version)
"""

from rag_client import DatabricksRAGClient, RAGConfig
from rag_config import databricks_config, rag_config
import os

def main():
    """Example usage of the RAG system"""
    
    print("🚀 Starting RAG System (Production Version)")
    print("=" * 50)
    
    # Initialize configuration
    config = RAGConfig(
        workflow_id=rag_config.workflow_id,
        embedding_model=rag_config.embedding_model,
        chunk_size=rag_config.chunk_size,
        chunk_overlap=rag_config.chunk_overlap,
        top_k=rag_config.top_k
    )
    
    # Create RAG client
    client = DatabricksRAGClient(config)
    
    # Check asset status
    print("\n📊 Checking Databricks assets...")
    status = client.get_asset_status()
    print("Asset Status:", status)
    
    # Example: Process documents
    print("\n📄 Processing sample documents...")
    sample_docs = [
        "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions without being explicitly programmed. It uses algorithms and statistical models to analyze and draw inferences from patterns in data.",
        "Data science combines statistics, programming, and domain expertise to extract insights from data. It involves collecting, cleaning, analyzing, and interpreting data to support decision-making processes.",
        "Natural language processing (NLP) is a field of AI that focuses on the interaction between computers and human language. It enables machines to understand, interpret, and generate human language in a meaningful way."
    ]
    
    processed_docs = client.process_documents(sample_docs)
    print(f"✅ Processed {len(processed_docs)} document chunks")
    
    # Example: Perform RAG query
    print("\n🔍 Performing RAG query...")
    query = "What is machine learning and how does it relate to AI?"
    result = client.query(query)
    print(result)
    
    print("\n🎉 Production system ready!")
    print("\n💡 Next steps:")
    print("1. Visit your Databricks workspace to see the created assets")
    print("2. Set up Vector Search endpoint in Databricks")
    print("3. Upload documents and implement full RAG pipeline")

if __name__ == "__main__":
    main()
