"""
Production Setup for Databricks RAG Workflow
This script validates credentials and tests connections before running the full workflow
"""

import os
import sys
from dotenv import load_dotenv

def validate_environment():
    """Validate that all required environment variables are set"""
    print("🔍 Validating Environment Variables...")
    
    required_vars = {
        "OPENAI_API_KEY": "OpenAI API Key",
        "DATABRICKS_HOST": "Databricks Workspace URL",
        "DATABRICKS_TOKEN": "Databricks Personal Access Token"
    }
    
    missing_vars = []
    
    for var, description in required_vars.items():
        value = os.getenv(var)
        if not value or value.startswith("your_") or value == "":
            missing_vars.append(f"  - {var}: {description}")
        else:
            print(f"  ✅ {var}: {description} - Set")
    
    if missing_vars:
        print("\n❌ Missing or invalid environment variables:")
        for var in missing_vars:
            print(var)
        print("\n📝 Please create a .env file with the following format:")
        print("""
# OpenAI API Configuration
OPENAI_API_KEY=sk-your-actual-openai-api-key

# Databricks Configuration
DATABRICKS_HOST=https://your-workspace.cloud.databricks.com
DATABRICKS_TOKEN=dapi-your-actual-databricks-token

# Optional: Databricks Catalog and Schema
DATABRICKS_CATALOG=your_catalog_name
DATABRICKS_SCHEMA=your_schema_name
        """)
        return False
    
    print("✅ All environment variables are set correctly!")
    return True

def test_databricks_connection():
    """Test the Databricks connection"""
    print("\n🔗 Testing Databricks Connection...")
    
    try:
        from databricks_sdk import WorkspaceClient
        
        host = os.getenv("DATABRICKS_HOST")
        token = os.getenv("DATABRICKS_TOKEN")
        
        print(f"  Connecting to: {host}")
        
        client = WorkspaceClient(host=host, token=token)
        
        # Test connection by getting current user
        current_user = client.current_user.me()
        print(f"  ✅ Connected successfully!")
        print(f"  👤 Current user: {current_user.user_name}")
        
        # Test Unity Catalog access
        try:
            catalogs = client.catalogs.list()
            print(f"  📚 Unity Catalog access: ✅ (Found {len(catalogs)} catalogs)")
        except Exception as e:
            print(f"  ⚠️  Unity Catalog access: Limited - {str(e)}")
            print("     You may need Unity Catalog admin privileges")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Databricks connection failed: {str(e)}")
        print("     Please check your DATABRICKS_HOST and DATABRICKS_TOKEN")
        return False

def test_openai_connection():
    """Test the OpenAI connection"""
    print("\n🤖 Testing OpenAI Connection...")
    
    try:
        from langchain_openai import ChatOpenAI
        
        api_key = os.getenv("OPENAI_API_KEY")
        
        llm = ChatOpenAI(
            model="gpt-4-turbo-preview",
            temperature=0.1,
            api_key=api_key
        )
        
        # Test with a simple message
        response = llm.invoke("Hello, this is a test message.")
        print("  ✅ OpenAI connection successful!")
        print(f"  📝 Test response: {response.content[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"  ❌ OpenAI connection failed: {str(e)}")
        print("     Please check your OPENAI_API_KEY")
        return False

def install_dependencies():
    """Check and install required dependencies"""
    print("\n📦 Checking Dependencies...")
    
    required_packages = [
        "langgraph",
        "langchain",
        "langchain-openai", 
        "databricks-sdk",
        "python-dotenv",
        "pydantic"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"  ✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"  ❌ {package} - Missing")
    
    if missing_packages:
        print(f"\n📥 Installing missing packages: {', '.join(missing_packages)}")
        try:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_packages)
            print("✅ Dependencies installed successfully!")
        except Exception as e:
            print(f"❌ Failed to install dependencies: {str(e)}")
            print("Please run: pip install -r requirements.txt")
            return False
    
    return True

def run_production_workflow():
    """Run the production workflow"""
    print("\n🚀 Starting Production Workflow...")
    
    try:
        from databricks_rag_flow import run_rag_workflow
        import asyncio
        
        # Get user input
        user_request = input("\nWhat would you like to do? (e.g., 'Create a RAG workflow'): ")
        
        if not user_request:
            print("No request provided. Exiting.")
            return
        
        print(f"\n🎯 Processing request: {user_request}")
        print("This will create REAL Databricks assets in your workspace...")
        
        # Confirm with user
        confirm = input("\n⚠️  This will create real Databricks assets. Continue? (y/N): ")
        if confirm.lower() != 'y':
            print("Operation cancelled.")
            return
        
        # Run the workflow
        result = asyncio.run(run_rag_workflow(user_request))
        
        if "error" in result:
            print(f"\n❌ Error: {result['error']}")
        else:
            print("\n✅ Production workflow completed successfully!")
            print("\n📁 Generated files:")
            for file in result.get("generated_files", []):
                print(f"  - {file}")
            
            print("\n🏗️  REAL Databricks assets created:")
            assets = result.get("databricks_assets", {})
            for asset_type, asset_info in assets.items():
                print(f"  - {asset_type}: {asset_info.get('name', asset_info.get('path', 'N/A'))}")
            
            print("\n🌐 View your assets in Databricks:")
            host = os.getenv("DATABRICKS_HOST")
            print(f"  Workspace: {host}")
            print("  - Go to Data > Unity Catalog to see your catalog and schema")
            print("  - Go to Workspace > Shared to see your notebook")
            print("  - Go to Data > Volumes to see your document volume")
            
            print("\n📖 Next steps:")
            print("1. Check your Databricks workspace for the created assets")
            print("2. Set up Vector Search endpoint in Databricks")
            print("3. Configure embedding model endpoint")
            print("4. Upload documents to the created volume")
            print("5. Run the generated example_usage.py to test the system")
        
    except Exception as e:
        print(f"\n❌ Error running production workflow: {str(e)}")
        print("Please check the error message above and try again.")

def main():
    """Main setup function"""
    print("🤖 Databricks RAG Workflow - Production Setup")
    print("=" * 60)
    
    # Load environment variables
    load_dotenv()
    
    # Step 1: Validate environment
    if not validate_environment():
        return
    
    # Step 2: Install dependencies
    if not install_dependencies():
        return
    
    # Step 3: Test connections
    databricks_ok = test_databricks_connection()
    openai_ok = test_openai_connection()
    
    if not databricks_ok or not openai_ok:
        print("\n❌ Connection tests failed. Please fix the issues above and try again.")
        return
    
    print("\n✅ All tests passed! Ready for production.")
    
    # Step 4: Run production workflow
    run_production_workflow()

if __name__ == "__main__":
    main()
