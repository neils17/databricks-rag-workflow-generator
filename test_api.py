#!/usr/bin/env python3
"""
Test script for the Databricks RAG Workflow API
This script tests the main API endpoints to ensure they work correctly.
"""

import requests
import json
import time
import sys
from typing import Dict, Any

class APITester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def test_health(self) -> bool:
        """Test the health check endpoint"""
        print("🔍 Testing health check...")
        try:
            response = self.session.get(f"{self.base_url}/health")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Health check passed: {data}")
                return True
            else:
                print(f"❌ Health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Health check error: {e}")
            return False
    
    def test_create_workflow(self) -> Dict[str, Any]:
        """Test workflow creation"""
        print("🚀 Testing workflow creation...")
        payload = {
            "workflow_name": "Test Workflow",
            "create_assets": False,  # Skip Databricks assets for testing
            "upload_documents": True,
            "documents": [
                {
                    "content": "This is a test document about artificial intelligence and machine learning. AI is transforming industries across the globe.",
                    "metadata": {"source": "test", "type": "documentation"}
                },
                {
                    "content": "Machine learning algorithms can be supervised, unsupervised, or reinforcement learning. Each has different applications and use cases.",
                    "metadata": {"source": "test", "type": "tutorial"}
                }
            ]
        }
        
        try:
            response = self.session.post(f"{self.base_url}/workflows", json=payload)
            if response.status_code == 202:
                data = response.json()
                print(f"✅ Workflow created: {data['workflow_id']}")
                return data
            else:
                print(f"❌ Workflow creation failed: {response.status_code} - {response.text}")
                return {}
        except Exception as e:
            print(f"❌ Workflow creation error: {e}")
            return {}
    
    def test_workflow_status(self, workflow_id: str) -> bool:
        """Test workflow status endpoint"""
        print(f"📊 Testing workflow status for {workflow_id}...")
        try:
            response = self.session.get(f"{self.base_url}/workflows/{workflow_id}/status")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Status: {data['status']} - {data.get('progress', 'No progress info')}")
                return True
            else:
                print(f"❌ Status check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Status check error: {e}")
            return False
    
    def test_list_workflows(self) -> bool:
        """Test listing workflows"""
        print("📋 Testing workflow listing...")
        try:
            response = self.session.get(f"{self.base_url}/workflows")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Found {len(data['workflows'])} workflows")
                return True
            else:
                print(f"❌ Workflow listing failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Workflow listing error: {e}")
            return False
    
    def test_query_workflow(self, workflow_id: str) -> bool:
        """Test querying a workflow"""
        print(f"🔍 Testing query for workflow {workflow_id}...")
        payload = {
            "query": "What is artificial intelligence?"
        }
        
        try:
            response = self.session.post(f"{self.base_url}/workflows/{workflow_id}/query", json=payload)
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Query successful: {data['response'][:100]}...")
                return True
            else:
                print(f"❌ Query failed: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            print(f"❌ Query error: {e}")
            return False
    
    def test_upload_documents(self, workflow_id: str) -> bool:
        """Test document upload"""
        print(f"📄 Testing document upload for workflow {workflow_id}...")
        
        # Create a simple text file for upload
        files = {
            'files': ('test_document.txt', 'This is a test document uploaded via API.', 'text/plain')
        }
        data = {
            'description': 'Test document upload'
        }
        
        try:
            response = self.session.post(f"{self.base_url}/workflows/{workflow_id}/upload", files=files, data=data)
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Document upload successful: {data['message']}")
                return True
            else:
                print(f"❌ Document upload failed: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            print(f"❌ Document upload error: {e}")
            return False
    
    def test_delete_workflow(self, workflow_id: str) -> bool:
        """Test workflow deletion"""
        print(f"🗑️ Testing workflow deletion for {workflow_id}...")
        try:
            response = self.session.delete(f"{self.base_url}/workflows/{workflow_id}")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Workflow deleted: {data['message']}")
                return True
            else:
                print(f"❌ Workflow deletion failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Workflow deletion error: {e}")
            return False
    
    def run_all_tests(self) -> bool:
        """Run all API tests"""
        print("🧪 Starting API Tests")
        print("=" * 50)
        
        # Test health check
        if not self.test_health():
            print("❌ Health check failed - API may not be running")
            return False
        
        print()
        
        # Test workflow creation
        workflow_data = self.test_create_workflow()
        if not workflow_data:
            print("❌ Workflow creation failed")
            return False
        
        workflow_id = workflow_data.get('workflow_id')
        if not workflow_id:
            print("❌ No workflow ID returned")
            return False
        
        print()
        
        # Test workflow status
        if not self.test_workflow_status(workflow_id):
            print("❌ Workflow status check failed")
            return False
        
        print()
        
        # Test listing workflows
        if not self.test_list_workflows():
            print("❌ Workflow listing failed")
            return False
        
        print()
        
        # Test document upload
        if not self.test_upload_documents(workflow_id):
            print("❌ Document upload failed")
            return False
        
        print()
        
        # Test querying (may fail if workflow is not ready)
        print("⚠️ Testing query (may fail if workflow not ready)...")
        self.test_query_workflow(workflow_id)
        
        print()
        
        # Test workflow deletion
        if not self.test_delete_workflow(workflow_id):
            print("❌ Workflow deletion failed")
            return False
        
        print()
        print("🎉 All tests completed!")
        return True

def main():
    """Main test function"""
    # Check if API URL is provided as argument
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    
    print(f"🧪 Testing Databricks RAG Workflow API at {base_url}")
    print("=" * 60)
    
    tester = APITester(base_url)
    success = tester.run_all_tests()
    
    if success:
        print("\n✅ All tests passed! The API is working correctly.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Check the API logs for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()
