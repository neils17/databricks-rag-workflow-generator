"""
Working Configuration for Databricks RAG System
"""

import os
from dataclasses import dataclass

@dataclass
class WorkingDatabricksConfig:
    """Working Databricks connection configuration"""
    host: str = os.getenv("DATABRICKS_HOST", "")
    token: str = os.getenv("DATABRICKS_TOKEN", "")
    workflow_id: str = "20250813_132627"
    vector_search_endpoint: str = "rag_endpoint_20250813_132627"

@dataclass
class WorkingRAGConfig:
    """Working RAG system configuration"""
    workflow_id: str = "20250813_132627"
    embedding_model: str = "text-embedding-3-small"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 5
    vector_search_endpoint: str = "rag_endpoint_20250813_132627"

@dataclass
class OpenAIConfig:
    """OpenAI configuration"""
    api_key: str = os.getenv("OPENAI_API_KEY", "")
    model: str = "gpt-4-turbo-preview"
    temperature: float = 0.1

# Configuration instances
working_databricks_config = WorkingDatabricksConfig()
working_rag_config = WorkingRAGConfig()
openai_config = OpenAIConfig()
