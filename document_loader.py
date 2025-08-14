"""
Document Loader Utility for Enhanced RAG System
Load your own documents from various sources
"""

import os
import json
from typing import List, Dict, Any
from pathlib import Path

def load_documents_from_text_files(directory: str) -> List[Dict[str, Any]]:
    """Load documents from text files in a directory"""
    documents = []
    directory_path = Path(directory)
    
    if not directory_path.exists():
        print(f"❌ Directory not found: {directory}")
        return documents
    
    # Supported text file extensions
    text_extensions = {'.txt', '.md', '.rst', '.py', '.js', '.html', '.css', '.json', '.xml', '.csv'}
    
    for file_path in directory_path.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in text_extensions:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                documents.append({
                    "title": file_path.name,
                    "source": str(file_path.relative_to(directory_path)),
                    "content": content
                })
                print(f"✅ Loaded: {file_path.name}")
                
            except Exception as e:
                print(f"❌ Error loading {file_path.name}: {str(e)}")
    
    print(f"📚 Loaded {len(documents)} documents from {directory}")
    return documents

def load_documents_from_json(json_file: str) -> List[Dict[str, Any]]:
    """Load documents from a JSON file"""
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle different JSON formats
        if isinstance(data, list):
            documents = data
        elif isinstance(data, dict) and 'documents' in data:
            documents = data['documents']
        else:
            print(f"❌ Unexpected JSON format in {json_file}")
            return []
        
        print(f"📚 Loaded {len(documents)} documents from {json_file}")
        return documents
        
    except Exception as e:
        print(f"❌ Error loading {json_file}: {str(e)}")
        return []

def create_sample_documents() -> List[Dict[str, Any]]:
    """Create sample documents for testing"""
    return [
        {
            "title": "Sample Company Policy",
            "source": "HR Department",
            "content": """
            Employee Handbook
            
            Welcome to our company! This handbook outlines our policies and procedures.
            
            Working Hours:
            - Standard work hours: 9:00 AM - 5:00 PM
            - Flexible work arrangements available
            - Remote work options for eligible positions
            
            Benefits:
            - Health insurance
            - Dental and vision coverage
            - 401(k) with company match
            - Paid time off
            - Professional development
            
            Code of Conduct:
            - Treat all colleagues with respect
            - Maintain confidentiality
            - Follow company policies
            - Report violations to HR
            """
        },
        {
            "title": "Technical Documentation",
            "source": "Engineering Team",
            "content": """
            System Architecture Overview
            
            Our system is built using modern microservices architecture.
            
            Components:
            - Frontend: React.js with TypeScript
            - Backend: Python FastAPI
            - Database: PostgreSQL
            - Cache: Redis
            - Message Queue: RabbitMQ
            
            API Endpoints:
            - GET /api/users - List users
            - POST /api/users - Create user
            - PUT /api/users/{id} - Update user
            - DELETE /api/users/{id} - Delete user
            
            Authentication:
            - JWT tokens
            - OAuth 2.0 support
            - Role-based access control
            
            Deployment:
            - Docker containers
            - Kubernetes orchestration
            - CI/CD pipeline with GitHub Actions
            """
        },
        {
            "title": "Product Guide",
            "source": "Product Team",
            "content": """
            Product User Guide
            
            Getting Started:
            1. Download the application
            2. Install following the setup wizard
            3. Create your account
            4. Complete the onboarding process
            
            Key Features:
            - Dashboard with real-time analytics
            - Custom reporting tools
            - Data import/export capabilities
            - User management system
            - Integration with third-party services
            
            Troubleshooting:
            - Check system requirements
            - Verify internet connection
            - Clear browser cache if needed
            - Contact support for technical issues
            
            Best Practices:
            - Regular data backups
            - Keep software updated
            - Use strong passwords
            - Review security settings
            """
        }
    ]

def load_documents_interactive() -> List[Dict[str, Any]]:
    """Interactive document loading"""
    print("📚 Document Loading Options:")
    print("1. Load from text files directory")
    print("2. Load from JSON file")
    print("3. Use sample documents")
    print("4. Enter documents manually")
    
    choice = input("\nSelect option (1-4): ").strip()
    
    if choice == "1":
        directory = input("Enter directory path: ").strip()
        return load_documents_from_text_files(directory)
    
    elif choice == "2":
        json_file = input("Enter JSON file path: ").strip()
        return load_documents_from_json(json_file)
    
    elif choice == "3":
        print("Using sample documents...")
        return create_sample_documents()
    
    elif choice == "4":
        return enter_documents_manually()
    
    else:
        print("Invalid choice. Using sample documents...")
        return create_sample_documents()

def enter_documents_manually() -> List[Dict[str, Any]]:
    """Enter documents manually"""
    documents = []
    
    print("\n📝 Manual Document Entry")
    print("Enter your documents (type 'done' when finished):")
    
    while True:
        print(f"\n--- Document {len(documents) + 1} ---")
        
        title = input("Document title: ").strip()
        if title.lower() == 'done':
            break
        
        source = input("Document source: ").strip()
        print("Document content (type 'END' on a new line when finished):")
        
        content_lines = []
        while True:
            line = input()
            if line.strip() == 'END':
                break
            content_lines.append(line)
        
        content = '\n'.join(content_lines)
        
        if title and content:
            documents.append({
                "title": title,
                "source": source,
                "content": content
            })
            print(f"✅ Added document: {title}")
        else:
            print("❌ Title and content are required")
    
    print(f"📚 Added {len(documents)} documents")
    return documents

def save_documents_to_json(documents: List[Dict[str, Any]], filename: str = "documents.json"):
    """Save documents to a JSON file"""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(documents, f, indent=2, ensure_ascii=False)
        print(f"✅ Documents saved to {filename}")
    except Exception as e:
        print(f"❌ Error saving documents: {str(e)}")

if __name__ == "__main__":
    print("📚 Document Loader Utility")
    print("=" * 40)
    
    documents = load_documents_interactive()
    
    if documents:
        print(f"\n📊 Summary: {len(documents)} documents loaded")
        for i, doc in enumerate(documents, 1):
            print(f"  {i}. {doc['title']} ({doc['source']})")
        
        save_choice = input("\nSave documents to JSON file? (y/n): ").strip().lower()
        if save_choice == 'y':
            filename = input("Enter filename (default: documents.json): ").strip()
            if not filename:
                filename = "documents.json"
            save_documents_to_json(documents, filename)
    else:
        print("❌ No documents loaded")
