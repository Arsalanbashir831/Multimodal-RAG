import os
import fitz  # PyMuPDF
from PIL import Image
import numpy as np
from io import BytesIO
from typing import List, Dict, Optional, Any
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Body, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
import tempfile
import shutil

# Import RAG components
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from sentence_transformers import SentenceTransformer

# Import image captioning model
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

# Initialize FastAPI app
app = FastAPI(title="Multimodal RAG API", description="API for RAG on PDFs with text and images")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Mount static files directory
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "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")

# Initialize embedding models
text_emb = OpenAIEmbeddings(model="text-embedding-3-small")
image_emb = SentenceTransformer("clip-ViT-B-32")

# Initialize BLIP for image captioning
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").eval()
blip_model.to(device)

# Store chat sessions
chat_sessions = {}

# Pydantic models
class Query(BaseModel):
    query: str
    session_id: Optional[str] = None

class ChatMessage(BaseModel):
    role: str  # 'user' or 'assistant'
    content: str

class ChatSession(BaseModel):
    session_id: str
    messages: List[ChatMessage]
    vectorstore_path: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    session_id: str
    sources: List[Dict[str, Any]]

# Helper functions
def describe_image(image_path):
    """Generate a caption for an image using BLIP model"""
    raw_image = Image.open(image_path).convert('RGB')
    inputs = processor(raw_image, return_tensors="pt").to(device)
    out = blip_model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

def extract_pdf_pages(pdf_path, image_dir):
    """Extract text and images from PDF"""
    os.makedirs(image_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    pages = []

    for i, page in enumerate(doc, 1):
        txt = page.get_text()
        imgs = []

        for img in page.get_images(full=True):
            xref = img[0]
            try:
                pix = fitz.Pixmap(doc, xref)
                if pix.colorspace.n not in [1, 3]:  # Not grayscale or RGB
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                buf = pix.tobytes("png")
                path = f"{image_dir}/pg{i}_img{xref}.png"
                Image.open(BytesIO(buf)).save(path)
                imgs.append(path)
            except Exception as e:
                print(f"Skipping image {xref} on page {i}: {e}")

        pages.append({"page": i, "text": txt, "images": imgs})
    return pages

def build_embeddings(pages, text_batch_size=50):
    """Build embeddings for text and image captions"""
    texts, metadatas = [], []

    print("Processing pages...")
    for p in pages:
        if p["text"].strip():
            texts.append(p["text"])
            metadatas.append({"type": "text", "page": p["page"]})

        for im in p["images"]:
            caption = describe_image(im)
            texts.append(caption)
            metadatas.append({
                "type": "image_caption",
                "page": p["page"],
                "path": im,
                "caption": caption
            })

    text_vecs = []
    for i in range(0, len(texts), text_batch_size):
        batch = texts[i : i + text_batch_size]
        text_vecs.extend(text_emb.embed_documents(batch))
    text_vecs = np.array(text_vecs)

    return text_vecs, texts, metadatas

def create_vectorstore(pages, index_path):
    """Create FAISS vectorstore from embeddings"""
    idx, docs, metas = build_embeddings(pages)
    text_embedding_pairs = list(zip(docs, idx))

    vs = FAISS.from_embeddings(
        text_embedding_pairs,
        embedding=text_emb,
        metadatas=metas
    )
    vs.save_local(index_path)
    return vs

def hybrid_rag_chain(vs, k=5):
    """Create RAG chain with retriever"""
    retriever = vs.as_retriever(search_kwargs={"k": k})
    llm = ChatOpenAI(model_name="gpt-4.1-mini", temperature=0.2)
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

def query_rag(rag, user_q, chat_history=None):
    """Query the RAG system with chat history context"""
    # If chat history exists, include it in the query context
    if chat_history and len(chat_history) > 0:
        context = "\n\n".join([f"{msg.role}: {msg.content}" for msg in chat_history])
        enhanced_query = f"Chat history:\n{context}\n\nCurrent question: {user_q}\n\nPlease answer the current question based on the retrieved documents and considering the chat history if relevant."
    else:
        enhanced_query = user_q
        
    result = rag({"query": enhanced_query})
    return result["result"], result["source_documents"]

# API endpoints
@app.post("/upload-pdf/", response_model=Dict[str, str])
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a PDF file and process it for RAG"""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    
    # Create a unique session ID
    session_id = str(uuid.uuid4())
    
    # Create temporary directories for this session
    temp_dir = tempfile.mkdtemp()
    pdf_path = os.path.join(temp_dir, file.filename)
    image_dir = os.path.join(temp_dir, "images")
    index_path = os.path.join(temp_dir, "faiss_index")
    
    # Save the uploaded PDF
    with open(pdf_path, "wb") as pdf_file:
        shutil.copyfileobj(file.file, pdf_file)
    
    try:
        # Process the PDF
        pages = extract_pdf_pages(pdf_path, image_dir)
        vs = create_vectorstore(pages, index_path)
        
        # Initialize chat session
        chat_sessions[session_id] = ChatSession(
            session_id=session_id,
            messages=[],
            vectorstore_path=index_path
        )
        
        return {"session_id": session_id, "message": "PDF processed successfully"}
    except Exception as e:
        # Clean up on error
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

@app.post("/query/", response_model=ChatResponse)
async def query(query_data: Query):
    """Query the RAG system with a question"""
    session_id = query_data.session_id
    
    # Create a new session if none exists
    if not session_id or session_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="Session not found. Please upload a PDF first.")
    
    session = chat_sessions[session_id]
    
    # Load the vectorstore
    if not session.vectorstore_path or not os.path.exists(os.path.join(session.vectorstore_path, "index.faiss")):
        raise HTTPException(status_code=500, detail="Vector store not found for this session")
    
    vs = FAISS.load_local(session.vectorstore_path, text_emb, allow_dangerous_deserialization=True)
    rag = hybrid_rag_chain(vs)
    
    # Add user message to chat history
    session.messages.append(ChatMessage(role="user", content=query_data.query))
    
    # Query the RAG system with chat history context
    answer, sources = query_rag(rag, query_data.query, session.messages)
    
    # Add assistant response to chat history
    session.messages.append(ChatMessage(role="assistant", content=answer))
    
    # Format sources for response
    formatted_sources = []
    for doc in sources:
        source = {
            "metadata": doc.metadata,
            "content": doc.page_content[:100] + ("..." if len(doc.page_content) > 100 else "")
        }
        formatted_sources.append(source)
    
    return ChatResponse(
        answer=answer,
        session_id=session_id,
        sources=formatted_sources
    )

@app.get("/sessions/{session_id}", response_model=ChatSession)
async def get_session(session_id: str):
    """Get chat session details"""
    if session_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return chat_sessions[session_id]

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a chat session and its resources"""
    if session_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Get the session and clean up resources
    session = chat_sessions[session_id]
    if session.vectorstore_path and os.path.exists(session.vectorstore_path):
        shutil.rmtree(os.path.dirname(session.vectorstore_path), ignore_errors=True)
    
    # Remove the session
    del chat_sessions[session_id]
    
    return {"message": f"Session {session_id} deleted successfully"}

@app.get("/", response_class=HTMLResponse)
async def get_html():
    """Serve the HTML interface"""
    with open(os.path.join(static_dir, "index.html"), "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

# Run the application
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)