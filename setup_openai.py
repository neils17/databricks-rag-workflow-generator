#!/usr/bin/env python3
"""
OpenAI API Key Setup Script
This script helps you set up your OpenAI API key to test the full RAG system.
"""

import os
import sys
from pathlib import Path

def setup_openai_api_key():
    """Set up OpenAI API key for testing"""
    
    print("🔑 OpenAI API Key Setup for RAG System Testing")
    print("=" * 50)
    
    # Check if API key is already set
    current_key = os.getenv("OPENAI_API_KEY")
    if current_key and current_key != "your_openai_api_key_here":
        print(f"✅ OpenAI API key is already set: {current_key[:10]}...")
        return True
    
    print("\n📋 To get your OpenAI API key:")
    print("1. Go to https://platform.openai.com/api-keys")
    print("2. Sign in to your OpenAI account")
    print("3. Click 'Create new secret key'")
    print("4. Copy the generated key (it starts with 'sk-')")
    print("5. Keep it secure - you won't be able to see it again!")
    
    print("\n⚠️  Important:")
    print("- Never share your API key publicly")
    print("- The key starts with 'sk-'")
    print("- You can set it temporarily for testing")
    
    # Get API key from user
    api_key = input("\n🔑 Enter your OpenAI API key (or press Enter to skip): ").strip()
    
    if not api_key:
        print("❌ No API key provided. You can set it later using:")
        print("   export OPENAI_API_KEY='your_api_key_here'")
        return False
    
    if not api_key.startswith("sk-"):
        print("❌ Invalid API key format. OpenAI API keys start with 'sk-'")
        return False
    
    # Set the API key for current session
    os.environ["OPENAI_API_KEY"] = api_key
    print(f"✅ OpenAI API key set successfully: {api_key[:10]}...")
    
    # Test the API key
    print("\n🧪 Testing OpenAI API key...")
    try:
        from langchain_openai import OpenAIEmbeddings
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        test_embedding = embeddings.embed_query("test")
        print("✅ OpenAI API key is working!")
        return True
    except Exception as e:
        print(f"❌ Error testing API key: {str(e)}")
        print("Please check your API key and try again.")
        return False

def test_rag_system():
    """Test the full RAG system with the API key"""
    
    print("\n🚀 Testing Full RAG System...")
    print("=" * 40)
    
    try:
        from working_rag_client import WorkingDatabricksRAGClient, WorkingRAGConfig
        from working_rag_config import working_databricks_config, working_rag_config
        
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
        print("📊 Checking Databricks assets...")
        status = client.get_asset_status()
        print("Asset Status:", status)
        
        # Test document processing
        print("\n📄 Testing document processing...")
        sample_docs = [
            "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions without being explicitly programmed. It uses algorithms and statistical models to analyze and draw inferences from patterns in data.",
            "Data science combines statistics, programming, and domain expertise to extract insights from data. It involves collecting, cleaning, analyzing, and interpreting data to support decision-making processes.",
            "Natural language processing (NLP) is a field of AI that focuses on the interaction between computers and human language. It enables machines to understand, interpret, and generate human language in a meaningful way."
        ]
        
        processed_docs = client.process_documents(sample_docs)
        print(f"✅ Processed {len(processed_docs)} document chunks")
        
        # Test RAG query
        print("\n🔍 Testing RAG query with full implementation...")
        query = "What is machine learning and how does it relate to AI?"
        result = client.query(query, sample_docs)
        print(f"Query: {query}")
        print(f"Response: {result}")
        
        print("\n🎉 Full RAG system test completed successfully!")
        print("\n💡 Your RAG system is now fully functional!")
        print("   - Document processing: ✅ Working")
        print("   - Embeddings: ✅ Working")
        print("   - LLM responses: ✅ Working")
        print("   - Databricks integration: ✅ Working")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing RAG system: {str(e)}")
        return False

def main():
    """Main function"""
    
    print("🤖 OpenAI API Key Setup for RAG System")
    print("=" * 50)
    
    # Set up API key
    if not setup_openai_api_key():
        print("\n❌ Setup incomplete. Please set your OpenAI API key and try again.")
        return
    
    # Test the RAG system
    if test_rag_system():
        print("\n🎉 Setup complete! Your RAG system is ready to use.")
        print("\n📖 Next steps:")
        print("1. Your OpenAI API key is set for this session")
        print("2. The RAG system is fully functional")
        print("3. You can now use it with your own documents")
        print("4. To make it permanent, add to your shell profile:")
        print("   echo 'export OPENAI_API_KEY=\"your_key_here\"' >> ~/.zshrc")
    else:
        print("\n❌ Setup failed. Please check your configuration and try again.")

if __name__ == "__main__":
    main()
