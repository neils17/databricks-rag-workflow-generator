"""
Demo Script for LangGraph Multi-Agent RAG Workflow
"""

from simple_demo import simulate_workflow

def main():
    """Run the complete workflow demo"""
    
    print("🤖 LangGraph Multi-Agent RAG Workflow Demo")
    print("=" * 60)
    
    # Example user request
    user_request = "Create a RAG workflow for document analysis"
    
    print(f"User Request: {user_request}")
    print("\n🚀 Starting multi-agent workflow...")
    print("\nAgents will execute in sequence:")
    print("1. User Interaction Agent - Understanding the request")
    print("2. Databricks Asset Agent - Creating Databricks assets")
    print("3. RAG Configuration Agent - Setting up RAG parameters")
    print("4. Local File Generation Agent - Creating Python files")
    
    # Run the workflow
    result = simulate_workflow(user_request)
    
    if result:
        print("\n✅ Workflow completed successfully!")
        print("\n📁 Generated files:")
        for file in result.get("generated_files", []):
            print(f"  - {file}")
        
        print("\n🏗️  Databricks assets (demo mode):")
        assets = result.get("databricks_assets", {})
        for asset_type, asset_info in assets.items():
            print(f"  - {asset_type}: {asset_info.get('name', asset_info.get('path', 'N/A'))}")
        
        print("\n🔧 RAG Configuration:")
        config = result.get("rag_config", {})
        for key, value in config.items():
            print(f"  - {key}: {value}")
        
        print("\n📖 Next steps:")
        print("1. Run 'python example_usage.py' to test the generated system")
        print("2. Set up real Databricks credentials for production use")
        print("3. Configure Vector Search endpoint in Databricks")
        print("4. Upload and process your documents")

if __name__ == "__main__":
    main()
