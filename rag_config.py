"""
Configuration for Databricks RAG System (Production Version)
"""

import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class DatabricksConfig:
    """Databricks connection configuration"""
    host: str = os.getenv("DATABRICKS_HOST", "")
    token: str = os.getenv("DATABRICKS_TOKEN", "")
    workflow_id: str = "20250813_131016"

@dataclass
class RAGConfig:
    """RAG system configuration"""
    workflow_id: str = "20250813_131016"
    embedding_model: str = "text-embedding-3-small"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 5
    vector_search_endpoint: Optional[str] = "your_vector_search_endpoint"

@dataclass
class OpenAIConfig:
    """OpenAI configuration"""
    api_key: str = os.getenv("OPENAI_API_KEY", "")
    model: str = "gpt-4-turbo-preview"
    temperature: float = 0.1

# Configuration instances
databricks_config = DatabricksConfig()
rag_config = RAGConfig()
openai_config = OpenAIConfig()
