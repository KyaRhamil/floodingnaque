#!/usr/bin/env python
"""Quick test script to verify the /api/models endpoint works."""

import requests
import json

def test_models_endpoint():
    """Test the /api/models endpoint."""
    url = "http://localhost:5000/api/models"
    
    try:
        print(f"Testing: {url}")
        response = requests.get(url)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("\n✓ Endpoint works!")
            print(f"\nResponse:")
            print(json.dumps(data, indent=2))
        else:
            print(f"\n✗ Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("✗ Error: Could not connect to server.")
        print("Make sure Flask app is running: python app.py")
    except Exception as e:
        print(f"✗ Error: {str(e)}")

if __name__ == '__main__':
    test_models_endpoint()

