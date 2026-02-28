#!/usr/bin/env python3
"""Quick test of the Apriel model setup"""

from src.config import Config
from src.clients.together_client import TogetherClient

def main():
    # Load config from environment
    config = Config.from_env()
    config.base_model = "ServiceNow-AI/Apriel-1.5-15b-Thinker"
    
    # Initialize client
    client = TogetherClient(api_key=config.together_api_key)
    
    # Simple test message
    messages = [
        {"role": "user", "content": "What is 2+2? Answer briefly."}
    ]
    
    print(f"Testing model: {config.base_model}")
    print(f"Sending test message...")
    
    try:
        response = client.chat(
            model=config.base_model,
            messages=messages,
            max_tokens=256,
            temperature=0.7
        )
        print(f"\n✓ Success!")
        print(f"Response: {response}")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        raise

if __name__ == "__main__":
    main()
