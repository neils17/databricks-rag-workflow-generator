#!/usr/bin/env python3

import sys
print("Python version:", sys.version)
print("Python path:", sys.path)

try:
    import databricks_sdk
    print("✅ databricks_sdk imported successfully")
    print("Version:", databricks_sdk.__version__)
except ImportError as e:
    print("❌ databricks_sdk import failed:", e)

try:
    from databricks_sdk import WorkspaceClient
    print("✅ WorkspaceClient imported successfully")
except ImportError as e:
    print("❌ WorkspaceClient import failed:", e)

try:
    import langchain_openai
    print("✅ langchain_openai imported successfully")
except ImportError as e:
    print("❌ langchain_openai import failed:", e)

try:
    import langgraph
    print("✅ langgraph imported successfully")
except ImportError as e:
    print("❌ langgraph import failed:", e)
