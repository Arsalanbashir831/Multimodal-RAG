import requests
import json
import os
import time

# Configuration
API_URL = "http://localhost:8000"
TEST_PDF = "test.pdf"  # Replace with a real PDF path if you have one

def test_api_endpoints():
    print("\n===== Testing Multimodal RAG API =====\n")
    
    # Check if API is running
    try:
        response = requests.get(f"{API_URL}/docs")
        if response.status_code == 200:
            print("✅ API is running")
        else:
            print("❌ API is not responding correctly")
            return
    except requests.exceptions.ConnectionError:
        print("❌ API is not running. Please start the server with 'python app.py'")
        return
    
    # Test PDF upload if a test file exists
    session_id = None
    if os.path.exists(TEST_PDF):
        print(f"\n📄 Uploading test PDF: {TEST_PDF}")
        with open(TEST_PDF, "rb") as f:
            files = {"file": (os.path.basename(TEST_PDF), f, "application/pdf")}
            response = requests.post(f"{API_URL}/upload-pdf/", files=files)
        
        if response.status_code == 200:
            result = response.json()
            session_id = result["session_id"]
            print(f"✅ PDF uploaded successfully. Session ID: {session_id}")
        else:
            print(f"❌ Error uploading PDF: {response.text}")
            print("⚠️ Skipping query tests that require a session ID")
    else:
        print(f"⚠️ Test PDF file '{TEST_PDF}' not found. Skipping upload test.")
        print("⚠️ Skipping query tests that require a session ID")
    
    # Test query endpoint if we have a session ID
    if session_id:
        print("\n🔍 Testing query endpoint")
        test_query = "What is this document about?"
        print(f"Query: '{test_query}'")
        
        payload = {"query": test_query, "session_id": session_id}
        response = requests.post(f"{API_URL}/query/", json=payload)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Query successful")
            print("\n🧠 Answer:")
            print(result["answer"])
            print("\n📚 Sources:")
            for source in result["sources"][:2]:  # Show only first 2 sources
                print(f"- {source['metadata']} {source['content']}")
            if len(result["sources"]) > 2:
                print(f"  ... and {len(result["sources"]) - 2} more sources")
        else:
            print(f"❌ Error querying API: {response.text}")
    
        # Test get session endpoint
        print("\n📋 Testing get session endpoint")
        response = requests.get(f"{API_URL}/sessions/{session_id}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Session retrieved successfully")
            print(f"Session ID: {result['session_id']}")
            print(f"Message count: {len(result['messages'])}")
        else:
            print(f"❌ Error getting session: {response.text}")
    
        # Test delete session endpoint
        print("\n🗑️ Testing delete session endpoint")
        response = requests.delete(f"{API_URL}/sessions/{session_id}")
        
        if response.status_code == 200:
            print(f"✅ Session deleted successfully")
        else:
            print(f"❌ Error deleting session: {response.text}")
    
    print("\n===== API Test Complete =====\n")

if __name__ == "__main__":
    test_api_endpoints()