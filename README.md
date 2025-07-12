# Multimodal RAG API

This project implements a FastAPI application for a Retrieval-Augmented Generation (RAG) pipeline that works with PDFs containing both text and images. The system maintains chat context across multiple queries.

## Features

- PDF processing with text extraction and image captioning
- Multimodal RAG pipeline that handles both text and images
- Chat context maintenance across multiple queries
- Session management for multiple users/documents
- RESTful API with FastAPI

## Requirements

- Python 3.8+
- OpenAI API key
- GPU recommended for faster image processing (but CPU mode is supported)

## Installation

1. Clone this repository or download the files

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Set your OpenAI API key as an environment variable:

```bash
# On Windows
set OPENAI_API_KEY=your-api-key-here

# On Linux/Mac
export OPENAI_API_KEY=your-api-key-here
```

Alternatively, you can edit the `app.py` file and set your API key directly:

```python
os.environ["OPENAI_API_KEY"] = "your-api-key-here"
```

## Usage

1. Start the FastAPI server:

```bash
python app.py
```

Or using uvicorn directly:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

2. Access the API documentation at http://localhost:8000/docs

## API Endpoints

### Upload PDF

```
POST /upload-pdf/
```

Upload a PDF file for processing. Returns a session ID for subsequent queries.

### Query

```
POST /query/
```

Send a query to the RAG system. Include the session ID to maintain chat context.

Example request body:
```json
{
  "query": "What are the key points in this document?",
  "session_id": "your-session-id"
}
```

### Get Session

```
GET /sessions/{session_id}
```

Retrieve the chat history and details for a specific session.

### Delete Session

```
DELETE /sessions/{session_id}
```

Delete a session and clean up its resources.

## How It Works

1. **PDF Processing**: When a PDF is uploaded, the system extracts text and images from each page.

2. **Image Captioning**: Images are processed using the BLIP model to generate descriptive captions.

3. **Embedding Generation**: Text and image captions are embedded using OpenAI's text-embedding-3-small model.

4. **Vector Storage**: Embeddings are stored in a FAISS vector database for efficient similarity search.

5. **Query Processing**: When a query is received, the system retrieves relevant documents and generates a response using a language model.

6. **Chat Context**: The system maintains the conversation history to provide context-aware responses.

## Customization

- **Language Model**: You can change the language model by modifying the `hybrid_rag_chain` function.
- **Embedding Model**: You can use different embedding models by changing the initialization in the app.
- **Number of Retrieved Documents**: Adjust the `k` parameter in the `hybrid_rag_chain` function.

## License

This project is licensed under the MIT License - see the LICENSE file for details.