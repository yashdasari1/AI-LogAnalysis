#!/bin/bash

# Setup script for Bob - LangChain + Ollama Integration
# This script sets up the environment for your persistent AI assistant "Bob"

echo "🤖 Setting up Bob - Your Persistent AI Assistant"
echo "================================================"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "❌ Ollama is required but not installed."
    echo "💡 Please install Ollama from: https://ollama.ai/"
    exit 1
fi

# Check if llama3.1:8b model is available
echo "🔍 Checking for llama3.1:8b model..."
if ! ollama list | grep -q "llama3.1:8b"; then
    echo "⬇️  Downloading llama3.1:8b model..."
    ollama pull llama3.1:8b
else
    echo "✅ llama3.1:8b model found!"
fi

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip3 install -r requirements.txt

# Make the script executable
chmod +x langchain_ollama_bob.py

echo ""
echo "🎉 Setup complete! Here's how to use Bob:"
echo ""
echo "🔸 Interactive Mode (recommended):"
echo "   python3 langchain_ollama_bob.py"
echo ""
echo "🔸 Example Mode:"
echo "   python3 langchain_ollama_bob.py example"
echo ""
echo "🔸 Direct CLI Integration:"
echo "   You can also create an alias for easy access:"
echo "   echo 'alias bob=\"python3 /workspace/langchain_ollama_bob.py\"' >> ~/.bashrc"
echo "   source ~/.bashrc"
echo "   bob"
echo ""
echo "💾 Bob will remember all conversations in: /workspace/.bob_memory.json"
echo "🔄 You can access the same Bob from Ollama CLI by referencing this memory file"
echo ""
echo "🚀 Ready to chat with Bob!"