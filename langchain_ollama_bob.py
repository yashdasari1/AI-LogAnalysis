#!/usr/bin/env python3
"""
LangChain + Ollama Integration with Persistent Memory
Model: llama3.1:8b (named "bob")
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any

from langchain_community.llms import Ollama
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


class PersistentOllamaChat:
    def __init__(self, model_name: str = "llama3.1:8b", session_name: str = "bob"):
        """
        Initialize the Ollama chat with persistent memory
        
        Args:
            model_name: The Ollama model to use (default: llama3.1:8b)
            session_name: Name for the persistent session (default: bob)
        """
        self.model_name = model_name
        self.session_name = session_name
        self.memory_file = Path(f"/workspace/.{session_name}_memory.json")
        
        # Initialize Ollama with streaming
        self.llm = Ollama(
            model=model_name,
            callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
            temperature=0.7,
        )
        
        # Initialize memory with conversation buffer
        self.memory = ConversationBufferWindowMemory(
            k=10,  # Keep last 10 exchanges
            return_messages=True
        )
        
        # Load existing conversation history
        self.load_memory()
        
        print(f"🤖 Bob (powered by {model_name}) is ready!")
        print(f"💾 Memory will be saved to: {self.memory_file}")
    
    def load_memory(self):
        """Load conversation history from file"""
        if self.memory_file.exists():
            try:
                with open(self.memory_file, 'r') as f:
                    data = json.load(f)
                    
                # Restore conversation history
                for exchange in data.get('conversations', []):
                    self.memory.chat_memory.add_user_message(exchange['human'])
                    self.memory.chat_memory.add_ai_message(exchange['ai'])
                
                print(f"📚 Loaded {len(data.get('conversations', []))} previous conversations")
            except Exception as e:
                print(f"⚠️  Could not load memory: {e}")
    
    def save_memory(self):
        """Save conversation history to file"""
        try:
            # Extract conversations from memory
            conversations = []
            messages = self.memory.chat_memory.messages
            
            for i in range(0, len(messages), 2):
                if i + 1 < len(messages):
                    human_msg = messages[i]
                    ai_msg = messages[i + 1]
                    conversations.append({
                        'human': human_msg.content,
                        'ai': ai_msg.content
                    })
            
            data = {
                'session_name': self.session_name,
                'model_name': self.model_name,
                'conversations': conversations
            }
            
            with open(self.memory_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            print(f"⚠️  Could not save memory: {e}")
    
    def chat(self, message: str) -> str:
        """
        Send a message to Bob and get a response
        
        Args:
            message: The user's message
            
        Returns:
            Bob's response
        """
        try:
            # Add user message to memory
            self.memory.chat_memory.add_user_message(message)
            
            # Get conversation context
            context = self.memory.chat_memory.messages
            
            # Prepare the prompt with context
            if len(context) > 1:
                conversation_history = "\n".join([
                    f"Human: {msg.content}" if isinstance(msg, HumanMessage) 
                    else f"Bob: {msg.content}" 
                    for msg in context[:-1]  # Exclude the current message
                ])
                
                prompt = f"""Previous conversation:
{conversation_history}

Current message:
Human: {message}

Bob: """
            else:
                prompt = f"Human: {message}\n\nBob: "
            
            # Get response from Ollama
            print(f"\n🧠 Bob is thinking...")
            response = self.llm(prompt)
            
            # Add AI response to memory
            self.memory.chat_memory.add_ai_message(response)
            
            # Save memory after each interaction
            self.save_memory()
            
            return response
            
        except Exception as e:
            error_msg = f"Sorry, I encountered an error: {e}"
            print(f"❌ {error_msg}")
            return error_msg
    
    def clear_memory(self):
        """Clear all conversation history"""
        self.memory.clear()
        if self.memory_file.exists():
            self.memory_file.unlink()
        print("🧹 Memory cleared!")
    
    def show_memory_stats(self):
        """Display memory statistics"""
        messages = self.memory.chat_memory.messages
        conversations = len(messages) // 2
        print(f"📊 Memory Stats:")
        print(f"   • Total conversations: {conversations}")
        print(f"   • Memory file: {self.memory_file}")
        print(f"   • Model: {self.model_name}")
        print(f"   • Session: {self.session_name}")


def interactive_mode():
    """Run Bob in interactive mode"""
    bob = PersistentOllamaChat(session_name="bob")
    
    print("\n" + "="*50)
    print("🤖 Welcome to Bob - Your Persistent AI Assistant!")
    print("💡 Commands:")
    print("   • Type 'quit' or 'exit' to leave")
    print("   • Type 'clear' to clear memory")
    print("   • Type 'stats' to see memory statistics")
    print("="*50 + "\n")
    
    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit']:
                print("👋 Goodbye! Bob will remember our conversation for next time.")
                break
            elif user_input.lower() == 'clear':
                bob.clear_memory()
                continue
            elif user_input.lower() == 'stats':
                bob.show_memory_stats()
                continue
            elif not user_input:
                continue
            
            print(f"\n🤖 Bob: ", end="")
            response = bob.chat(user_input)
            print()  # Add newline after streaming response
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye! Bob will remember our conversation for next time.")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def example_usage():
    """Example of how to use Bob programmatically"""
    print("🔧 Example Usage:")
    
    # Create Bob instance
    bob = PersistentOllamaChat(session_name="bob")
    
    # Have a conversation
    questions = [
        "Hi Bob! What's your name?",
        "What can you help me with?",
        "Remember that my favorite color is blue.",
        "What's my favorite color?"
    ]
    
    for question in questions:
        print(f"\n👤 Human: {question}")
        print(f"🤖 Bob: ", end="")
        response = bob.chat(question)
        print()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "example":
            example_usage()
        elif sys.argv[1] == "interactive":
            interactive_mode()
        else:
            print("Usage: python langchain_ollama_bob.py [interactive|example]")
    else:
        interactive_mode()