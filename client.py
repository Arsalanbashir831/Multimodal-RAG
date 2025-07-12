import requests
import json
import os
import argparse


def main():
    parser = argparse.ArgumentParser(description="Client for Multimodal RAG API")
    parser.add_argument("--pdf", type=str, help="Path to PDF file to upload")
    parser.add_argument("--query", type=str, help="Query to send to the API")
    parser.add_argument("--session", type=str, help="Session ID for continuing a conversation")
    parser.add_argument("--list-session", type=str, help="List chat history for a session ID")
    parser.add_argument("--delete-session", type=str, help="Delete a session by ID")
    parser.add_argument("--url", type=str, default="http://localhost:8000", help="API URL")
    
    args = parser.parse_args()
    base_url = args.url
    
    # Upload PDF
    if args.pdf:
        if not os.path.exists(args.pdf):
            print(f"Error: PDF file '{args.pdf}' not found")
            return
            
        print(f"Uploading PDF: {args.pdf}")
        with open(args.pdf, "rb") as f:
            files = {"file": (os.path.basename(args.pdf), f, "application/pdf")}
            response = requests.post(f"{base_url}/upload-pdf/", files=files)
        
        if response.status_code == 200:
            result = response.json()
            print(f"PDF uploaded successfully. Session ID: {result['session_id']}")
            print(f"Save this session ID for future queries: {result['session_id']}")
            
            # If query is also provided, use this session ID
            if args.query:
                args.session = result['session_id']
        else:
            print(f"Error uploading PDF: {response.text}")
            return
    
    # Query the API
    if args.query:
        if not args.session:
            print("Error: Session ID is required for queries. Upload a PDF first or provide --session")
            return
            
        print(f"Sending query: '{args.query}' to session {args.session}")
        payload = {"query": args.query, "session_id": args.session}
        response = requests.post(f"{base_url}/query/", json=payload)
        
        if response.status_code == 200:
            result = response.json()
            print("\n🧠 Answer:")
            print(result["answer"])
            print("\n📚 Sources:")
            for source in result["sources"]:
                print(f"- {source['metadata']} {source['content']}")
        else:
            print(f"Error querying API: {response.text}")
    
    # List session chat history
    if args.list_session:
        print(f"Getting chat history for session: {args.list_session}")
        response = requests.get(f"{base_url}/sessions/{args.list_session}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Session ID: {result['session_id']}")
            print("Chat History:")
            for i, msg in enumerate(result['messages'], 1):
                print(f"{i}. {msg['role']}: {msg['content']}")
        else:
            print(f"Error getting session: {response.text}")
    
    # Delete session
    if args.delete_session:
        print(f"Deleting session: {args.delete_session}")
        response = requests.delete(f"{base_url}/sessions/{args.delete_session}")
        
        if response.status_code == 200:
            print(f"Session {args.delete_session} deleted successfully")
        else:
            print(f"Error deleting session: {response.text}")


if __name__ == "__main__":
    main()