#!/usr/bin/env python3
"""
Bob CLI Integration Helper
This script allows you to use Bob's persistent memory with direct Ollama CLI commands
"""

import json
import sys
from pathlib import Path
from typing import List, Dict


class BobMemoryHelper:
    def __init__(self, session_name: str = "bob"):
        self.session_name = session_name
        self.memory_file = Path(f"/workspace/.{session_name}_memory.json")
    
    def get_conversation_context(self, max_exchanges: int = 5) -> str:
        """Get recent conversation context for Ollama CLI usage"""
        if not self.memory_file.exists():
            return ""
        
        try:
            with open(self.memory_file, 'r') as f:
                data = json.load(f)
            
            conversations = data.get('conversations', [])
            recent_conversations = conversations[-max_exchanges:] if conversations else []
            
            if not recent_conversations:
                return ""
            
            context_lines = ["Previous conversation with Bob:"]
            for conv in recent_conversations:
                context_lines.append(f"Human: {conv['human']}")
                context_lines.append(f"Bob: {conv['ai']}")
            
            context_lines.append("\nCurrent message:")
            return "\n".join(context_lines)
            
        except Exception as e:
            print(f"Warning: Could not load Bob's memory: {e}", file=sys.stderr)
            return ""
    
    def add_exchange(self, human_message: str, ai_response: str):
        """Add a new exchange to Bob's memory"""
        try:
            # Load existing data
            data = {'session_name': self.session_name, 'model_name': 'llama3.1:8b', 'conversations': []}
            if self.memory_file.exists():
                with open(self.memory_file, 'r') as f:
                    data = json.load(f)
            
            # Add new conversation
            data['conversations'].append({
                'human': human_message,
                'ai': ai_response
            })
            
            # Keep only last 50 conversations to prevent file from growing too large
            if len(data['conversations']) > 50:
                data['conversations'] = data['conversations'][-50:]
            
            # Save updated data
            with open(self.memory_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            print(f"Warning: Could not save to Bob's memory: {e}", file=sys.stderr)


def generate_ollama_prompt(user_message: str, include_context: bool = True) -> str:
    """Generate a prompt for direct Ollama CLI usage with Bob's context"""
    helper = BobMemoryHelper()
    
    if include_context:
        context = helper.get_conversation_context()
        if context:
            return f"{context}\nHuman: {user_message}\n\nBob:"
    
    return f"Human: {user_message}\n\nBob:"


def main():
    if len(sys.argv) < 2:
        print("Bob CLI Integration Helper")
        print("")
        print("Usage:")
        print("  python3 bob_cli_integration.py prompt 'Your message'")
        print("    - Generates a prompt with Bob's memory context")
        print("")
        print("  python3 bob_cli_integration.py save 'Human message' 'AI response'")
        print("    - Saves an exchange to Bob's memory")
        print("")
        print("  python3 bob_cli_integration.py context")
        print("    - Shows recent conversation context")
        print("")
        print("Example Ollama CLI usage:")
        print("  MESSAGE='Hello Bob, how are you?'")
        print("  PROMPT=$(python3 bob_cli_integration.py prompt \"$MESSAGE\")")
        print("  RESPONSE=$(echo \"$PROMPT\" | ollama run llama3.1:8b)")
        print("  python3 bob_cli_integration.py save \"$MESSAGE\" \"$RESPONSE\"")
        return
    
    command = sys.argv[1]
    helper = BobMemoryHelper()
    
    if command == "prompt":
        if len(sys.argv) < 3:
            print("Error: Please provide a message", file=sys.stderr)
            sys.exit(1)
        
        user_message = sys.argv[2]
        prompt = generate_ollama_prompt(user_message)
        print(prompt)
    
    elif command == "save":
        if len(sys.argv) < 4:
            print("Error: Please provide both human message and AI response", file=sys.stderr)
            sys.exit(1)
        
        human_message = sys.argv[2]
        ai_response = sys.argv[3]
        helper.add_exchange(human_message, ai_response)
        print("Conversation saved to Bob's memory!")
    
    elif command == "context":
        context = helper.get_conversation_context()
        if context:
            print(context)
        else:
            print("No conversation history found.")
    
    else:
        print(f"Error: Unknown command '{command}'", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()