#!/bin/bash
# OpenAI API Key Setup Script

echo "🔑 OpenAI API Key Setup"
echo "======================="

echo ""
echo "📋 To get your OpenAI API key:"
echo "1. Go to https://platform.openai.com/api-keys"
echo "2. Sign in to your OpenAI account"
echo "3. Click 'Create new secret key'"
echo "4. Copy the generated key (it starts with 'sk-')"
echo "5. Keep it secure - you won't be able to see it again!"
echo ""

read -p "🔑 Enter your OpenAI API key: " api_key

if [[ -z "$api_key" ]]; then
    echo "❌ No API key provided."
    exit 1
fi

if [[ ! "$api_key" =~ ^sk- ]]; then
    echo "❌ Invalid API key format. OpenAI API keys start with 'sk-'"
    exit 1
fi

# Set the environment variable
export OPENAI_API_KEY="$api_key"

echo "✅ OpenAI API key set successfully!"
echo "🔍 Testing the key..."

# Test the key by running a simple Python script
python3 -c "
import os
from langchain_openai import OpenAIEmbeddings

try:
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv('OPENAI_API_KEY'))
    test_embedding = embeddings.embed_query('test')
    print('✅ OpenAI API key is working!')
    print('🚀 You can now test the full RAG system with:')
    print('   python working_example_usage.py')
except Exception as e:
    print(f'❌ Error testing API key: {e}')
"

echo ""
echo "💡 To make this permanent, add to your shell profile:"
echo "   echo 'export OPENAI_API_KEY=\"$api_key\"' >> ~/.zshrc"
