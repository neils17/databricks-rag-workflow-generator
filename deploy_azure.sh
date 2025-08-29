#!/bin/bash

# Azure Functions Deployment Script
# This script deploys the RAG workflow to Azure Functions

set -e

echo "🚀 Azure Functions Deployment"
echo "============================="

# Check if Azure CLI is installed
if ! command -v az &> /dev/null; then
    echo "❌ Error: Azure CLI is not installed"
    echo "Please install Azure CLI: https://docs.microsoft.com/en-us/cli/azure/install-azure-cli"
    exit 1
fi

# Check if .env file exists
if [ ! -f .env ]; then
    echo "❌ Error: .env file not found!"
    echo "Please create a .env file with your configuration:"
    echo "cp config.env.example .env"
    echo "Then edit .env with your actual values."
    exit 1
fi

# Load environment variables
source .env

# Check required environment variables
required_vars=("OPENAI_API_KEY" "DATABRICKS_HOST" "DATABRICKS_TOKEN")
for var in "${required_vars[@]}"; do
    if [ -z "${!var}" ]; then
        echo "❌ Error: $var is not set in .env file"
        exit 1
    fi
done

echo "✅ Environment variables loaded"

# Check if logged in to Azure
if ! az account show &> /dev/null; then
    echo "🔐 Please log in to Azure..."
    az login
fi

# Get current subscription
SUBSCRIPTION_ID=$(az account show --query id -o tsv)
echo "📋 Using subscription: $SUBSCRIPTION_ID"

# Prompt for resource group and function app name
read -p "Enter resource group name (or press Enter for 'rag-workflow-rg'): " RESOURCE_GROUP
RESOURCE_GROUP=${RESOURCE_GROUP:-rag-workflow-rg}

read -p "Enter function app name (or press Enter for 'rag-workflow-$(date +%s)'): " FUNCTION_APP_NAME
FUNCTION_APP_NAME=${FUNCTION_APP_NAME:-rag-workflow-$(date +%s)}

read -p "Enter location (or press Enter for 'eastus'): " LOCATION
LOCATION=${LOCATION:-eastus}

read -p "Enter storage account name (or press Enter for 'ragstorage$(date +%s)'): " STORAGE_ACCOUNT
STORAGE_ACCOUNT=${STORAGE_ACCOUNT:-ragstorage$(date +%s)}

echo ""
echo "🔧 Configuration:"
echo "  Resource Group: $RESOURCE_GROUP"
echo "  Function App: $FUNCTION_APP_NAME"
echo "  Location: $LOCATION"
echo "  Storage Account: $STORAGE_ACCOUNT"
echo ""

read -p "Continue with deployment? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Deployment cancelled"
    exit 1
fi

# Create resource group
echo "📦 Creating resource group..."
az group create --name $RESOURCE_GROUP --location $LOCATION

# Create storage account
echo "💾 Creating storage account..."
az storage account create \
    --name $STORAGE_ACCOUNT \
    --resource-group $RESOURCE_GROUP \
    --location $LOCATION \
    --sku Standard_LRS

# Create function app
echo "🚀 Creating function app..."
az functionapp create \
    --name $FUNCTION_APP_NAME \
    --storage-account $STORAGE_ACCOUNT \
    --resource-group $RESOURCE_GROUP \
    --consumption-plan-location $LOCATION \
    --runtime python \
    --runtime-version 3.11 \
    --functions-version 4 \
    --os-type Linux

# Configure app settings
echo "⚙️ Configuring app settings..."
az functionapp config appsettings set \
    --name $FUNCTION_APP_NAME \
    --resource-group $RESOURCE_GROUP \
    --settings \
    "OPENAI_API_KEY=$OPENAI_API_KEY" \
    "DATABRICKS_HOST=$DATABRICKS_HOST" \
    "DATABRICKS_TOKEN=$DATABRICKS_TOKEN" \
    "DATABRICKS_CATALOG=$DATABRICKS_CATALOG" \
    "DATABRICKS_SCHEMA=$DATABRICKS_SCHEMA"

# Deploy function app
echo "📤 Deploying function app..."
cd azure_function
func azure functionapp publish $FUNCTION_APP_NAME --python

if [ $? -eq 0 ]; then
    echo "✅ Deployment successful!"
    echo ""
    echo "🌐 Function App URL: https://$FUNCTION_APP_NAME.azurewebsites.net"
    echo "📚 API Documentation: https://$FUNCTION_APP_NAME.azurewebsites.net/api/docs"
    echo "🔍 Health Check: https://$FUNCTION_APP_NAME.azurewebsites.net/api/health"
    echo ""
    echo "📋 Available endpoints:"
    echo "  POST   /api/workflows                    - Create new workflow"
    echo "  GET    /api/workflows                    - List all workflows"
    echo "  GET    /api/workflows/{id}/status        - Get workflow status"
    echo "  POST   /api/workflows/{id}/query         - Query workflow"
    echo "  POST   /api/workflows/{id}/upload        - Upload documents"
    echo "  DELETE /api/workflows/{id}               - Delete workflow"
    echo ""
    echo "🔧 To view logs: az functionapp logs tail --name $FUNCTION_APP_NAME --resource-group $RESOURCE_GROUP"
else
    echo "❌ Deployment failed"
    exit 1
fi
