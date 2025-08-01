from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import UserRegistrationSerializer
from rest_framework.permissions import IsAuthenticated
from rest_framework.parsers import MultiPartParser, FormParser
from .models import File, User
from .serializers import FileSerializer
import chromadb
from .models import Chat, Message
from .serializers import ChatSerializer, MessageSerializer
from rest_framework import generics
import fitz  # PyMuPDF
from PIL import Image
import numpy as np
from io import BytesIO
from docx import Document
from langchain_openai import OpenAIEmbeddings as OE
from sentence_transformers import SentenceTransformer
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
import os
from tqdm import tqdm
import openai
from uuid import uuid4
import re
from pathlib import Path
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from django.shortcuts import get_object_or_404

load_dotenv()

from django.conf import settings
from django.db import transaction, models
from django.utils import timezone

# Initialize ChromaDB clients (simple local setup)
chroma_client = chromadb.Client(chromadb.config.Settings(
    persist_directory="chroma_data"
))

# Load BLIP for image captioning
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast=True)
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
blip_model.to(device)

text_emb = OE(model="text-embedding-3-small")
#image_emb = SentenceTransformer("clip-ViT-B-32")

# --- PDF and Image Processing ---
def describe_image(image_path):
    raw_image = Image.open(image_path).convert('RGB')
    try:
        inputs = processor([raw_image], padding=True, return_tensors="pt").to(device)
        out = blip_model.generate(**inputs)
        # Check if output tensor has at least one sequence
        if out is None or out.shape[0] == 0:
            return "No caption generated"
        caption = processor.decode(out[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        return "No caption generated"

def extract_pdf_pages(pdf_path, image_dir="imgs"):
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
                if pix.colorspace.n not in [1, 3]:
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                buf = pix.tobytes("png")
                path = f"{image_dir}/pg{i}_img{xref}.png"
                Image.open(BytesIO(buf)).save(path)
                imgs.append(path)
            except Exception:
                pass
        pages.append({"page": i, "text": txt, "images": imgs})
    return pages

def extract_docx_text(docx_path):
    """Extract text from DOCX file"""
    try:
        doc = Document(docx_path)
        full_text = []
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                full_text.append(paragraph.text)
        
        # Also extract text from tables
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    if cell.text.strip():
                        full_text.append(cell.text)
        
        return {"page": 1, "text": "\n".join(full_text), "images": []}
    except Exception as e:
        raise ValueError(f"Failed to extract text from DOCX file: {str(e)}")

def build_embeddings(pages, text_batch_size=50):
    texts, metadatas = [], []
    for p in pages:
        if p["text"].strip():
            texts.append(p["text"])
            metadatas.append({"type": "text", "page": p["page"]})
        for im in p["images"]:
            caption = describe_image(im)
            if not caption or caption.startswith("Image captioning failed"):
                caption = "No caption generated"
            texts.append(caption)
            metadatas.append({
                "type": "image_caption",
                "page": p["page"],
                "path": im,
                "caption": caption
            })
    text_vecs = []
    if texts:
        for i in range(0, len(texts), text_batch_size):
            batch = texts[i : i + text_batch_size]
            text_vecs.extend(text_emb.embed_documents(batch))  # Only use OpenAIEmbeddings
    return text_vecs, texts, metadatas

# --- Per-user Chroma directory helper ---
def get_user_chroma_dir(user_or_id):
    if isinstance(user_or_id, int) or isinstance(user_or_id, str):
        user_id = user_or_id
    else:
        user_id = user_or_id.id
    return os.path.join("chroma_data", f"user_{user_id}")

# --- File upload: extract, caption, embed, and store in per-user Chroma ---
def process_and_store_file(user, file_path, collection_name, file_id=None):
    try:
        print(f"Processing file: {file_path}")
        
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Determine file type and extract content accordingly
        file_extension = file_path.lower().split('.')[-1]
        print(f"File extension: {file_extension}")
        
        if file_extension == 'pdf':
            print("Extracting PDF pages...")
            pages = extract_pdf_pages(file_path)
        elif file_extension == 'docx':
            print("Extracting DOCX text...")
            # For DOCX, we get a single page with all text
            pages = [extract_docx_text(file_path)]
        else:
            raise ValueError(f"Unsupported file type: {file_extension}. Only PDF and DOCX are supported.")
        
        print(f"Extracted {len(pages)} pages")
        
        texts, metas, ids = [], [], []
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunk_idx = 0
        
        for p in pages:
            if p["text"].strip():
                chunks = splitter.split_text(p["text"])
                for chunk in chunks:
                    texts.append(chunk)
                    metas.append({"type": "text", "page": p["page"], "file_id": file_id})
                    ids.append(f"{file_id}_{chunk_idx}")
                    chunk_idx += 1
            for im in p["images"]:
                caption = describe_image(im)
                texts.append(caption)
                metas.append({
                    "type": "image_caption",
                    "page": p["page"],
                    "file_id": file_id,
                    "path": im,
                    "caption": caption
                })
                ids.append(f"{file_id}_img_{chunk_idx}")
                chunk_idx += 1

        print(f"Created {len(texts)} text chunks")
        
        if not texts:
            raise ValueError(f"{file_extension.upper()} file contained no text or images.")
        
        user_chroma_dir = get_user_chroma_dir(user)
        print(f"Chroma directory: {user_chroma_dir}")
        
        # Ensure directory exists
        os.makedirs(user_chroma_dir, exist_ok=True)
        
        vs = Chroma(
            collection_name=collection_name,
            embedding_function=text_emb,
            persist_directory=user_chroma_dir,
        )
        
        print(f"Adding {len(texts)} texts to ChromaDB...")
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_metas = metas[i:i+batch_size]
            batch_ids = ids[i:i+batch_size]
            vs.add_texts(
                texts=batch_texts,
                metadatas=batch_metas,
                ids=batch_ids
            )
            print(f"Added batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
        
        # The newer version of langchain-chroma doesn't have persist() method
        # The data is automatically persisted when using persist_directory
        print(f"Successfully stored {len(texts)} documents for user {user.id} in {user_chroma_dir}")
        
        # Clean up extracted images after processing
        for p in pages:
            for im in p["images"]:
                try:
                    os.remove(im)
                except Exception:
                    pass  # Ignore errors during cleanup
                    
    except Exception as e:
        print(f"Error in process_and_store_file: {str(e)}")
        import traceback
        traceback.print_exc()
        raise e

# --- RAG query: load per-user Chroma and run RetrievalQA ---
def run_rag_query(user, query, collection_name, llm=None, k=5):
    user_chroma_dir = get_user_chroma_dir(user)
    if not Path(user_chroma_dir).exists():
        # For general questions, provide a helpful response without mentioning documents
        if any(word in query.lower() for word in ['hello', 'hi', 'hey', 'greetings', 'how are you', 'good morning', 'good afternoon', 'good evening']):
            return "Hello! I'm here to help you with your questions. Feel free to ask me anything!", []
        return "I don't have any documents to reference, but I'm happy to help with general questions!", []
    
    vs = Chroma(
        persist_directory=user_chroma_dir,
        embedding_function=text_emb,
        collection_name=collection_name,
    )
    prompt_template = (
        "You are a helpful assistant. Use the provided context and conversation history to answer the question. "
        "If the question refers to previous messages in the conversation, make sure to consider that context. "
        "When you find relevant information in the context, provide specific details and facts from the documents. "
        "If you cannot find specific information in the context, respond with a helpful message like 'I don't have enough information to answer this question' or 'I cannot find specific information about that in the available documents.' "
        "Be honest about what information is available and what is not. Provide detailed answers when you have specific information from the documents.\n\n"
        "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    if llm is None:
        llm = ChatOpenAI(model_name="gpt-4.1-mini", openai_api_key=settings.OPENAI_API_KEY, temperature=0.2)
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vs.as_retriever(search_kwargs={"k": k}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    try:
        result = rag_chain.invoke({"query": query})
        answer = result["result"]
        sources = result.get("source_documents", [])
        
        # Check if sources are actually relevant (have reasonable similarity scores)
        # This is a heuristic - if the answer is very generic, the sources might not be relevant
        print(f"Retrieved {len(sources)} source documents")
        
        # Simple logic: only include sources if documents were actually retrieved
        if not sources:
            print(f"No documents retrieved - returning empty sources")
            return answer, []
        
        # If documents were retrieved, include them as sources
        print(f"Documents retrieved - including sources")
        return answer, sources
    except Exception as e:
        print(f"RAG query failed: {str(e)}")
        return "I apologize, but I encountered an error while processing your request. Please try again or contact support if the issue persists.", []


def run_rag_pipeline(user, query, chat_id=None):
    # First, check if this is a question that doesn't need document retrieval
    query_lower = query.lower().strip()
    
    # Basic greetings and casual conversation
    basic_question_keywords = [
        'hello', 'hi', 'hey', 'greetings', 'good morning', 'good afternoon', 'good evening',
        'how are you', 'how are you doing', 'what\'s up', 'sup', 'yo',
        'thanks', 'thank you', 'bye', 'goodbye', 'see you', 'good night',
        'nice to meet you', 'pleasure to meet you', 'good to see you'
    ]
    
    # Conversation context questions (about previous messages)
    context_keywords = [
        'what did we discuss', 'what did we talk about', 'what was our conversation',
        'what did i ask', 'what did you say', 'what was your answer', 'what did you tell me',
        'what was mentioned', 'what did we cover', 'what was the topic', 'what were we discussing',
        'what did we just talk about', 'what was our last conversation', 'what did you just say',
        'can you repeat', 'can you remind me', 'what was that again', 'what did you just tell me',
        'what was the last thing', 'what did we just discuss', 'what was our previous topic',
        'what did you mention', 'what did you explain', 'what did you describe',
        'what was the subject', 'what was the theme', 'what were we covering'
    ]
    
    # General knowledge questions that don't need documents
    general_knowledge_keywords = [
        'what is the weather', 'what time is it', 'what day is it', 'what date is it',
        'how are you', 'what can you do', 'what are your capabilities', 'what are your features',
        'help', 'what do you do', 'what is your purpose', 'what is your function','what is my name',    ]
    
    # Document-specific question indicators
    document_keywords = [
        'what does the document', 'what does the report', 'what does the file', 'what does the pdf',
        'what does the docx', 'what does the file say', 'what does the report say', 'what does the document say',
        'what is in the document', 'what is in the report', 'what is in the file', 'what is in the pdf',
        'what is in the docx', 'what does it say in the document', 'what does it say in the report',
        'what does it say in the file', 'what does it say in the pdf', 'what does it say in the docx',
        'what is mentioned in the document', 'what is mentioned in the report', 'what is mentioned in the file',
        'what is mentioned in the pdf', 'what is mentioned in the docx', 'what is stated in the document',
        'what is stated in the report', 'what is stated in the file', 'what is stated in the pdf',
        'what is stated in the docx', 'what is written in the document', 'what is written in the report',
        'what is written in the file', 'what is written in the pdf', 'what is written in the docx',
        'what are the contents of', 'what are the details in', 'what are the facts in',
        'what are the figures in', 'what are the numbers in', 'what are the statistics in',
        'what are the data in', 'what are the results in', 'what are the findings in',
        'what are the conclusions in', 'what are the recommendations in', 'what are the suggestions in',
        'what does the', 'what is in the', 'what is mentioned in the', 'what is stated in the',
        'what is written in the', 'what are the contents of the', 'what are the details in the',
        'what are the facts in the', 'what are the figures in the', 'what are the numbers in the',
        'what are the statistics in the', 'what are the data in the', 'what are the results in the',
        'what are the findings in the', 'what are the conclusions in the', 'what are the recommendations in the',
        'what are the suggestions in the', 'what does it say about', 'what is mentioned about',
        'what is stated about', 'what is written about', 'what are the details about',
        'what are the facts about', 'what are the figures about', 'what are the numbers about',
        'what are the statistics about', 'what are the data about', 'what are the results about',
        'what are the findings about', 'what are the conclusions about', 'what are the recommendations about'
    ]
    
    # Check if it's any type of question that doesn't need document retrieval
    is_basic_question = any(keyword in query_lower for keyword in basic_question_keywords)
    is_context_question = any(keyword in query_lower for keyword in context_keywords)
    is_general_knowledge = any(keyword in query_lower for keyword in general_knowledge_keywords)
    is_document_question = any(keyword in query_lower for keyword in document_keywords)
    
    # For questions that don't need document retrieval, provide appropriate responses
    if is_basic_question or is_general_knowledge:
        if is_basic_question:
            if any(greeting in query_lower for greeting in ['hello', 'hi', 'hey', 'greetings', 'good morning', 'good afternoon', 'good evening']):
                response = "Hello! I'm here to help you with your questions. Feel free to ask me anything about your documents or any general questions!"
            elif any(thanks in query_lower for thanks in ['thanks', 'thank you']):
                response = "You're welcome! I'm happy to help. Is there anything else you'd like to know?"
            elif any(bye in query_lower for bye in ['bye', 'goodbye', 'see you', 'good night']):
                response = "Goodbye! Feel free to come back if you have more questions."
            else:
                response = "Hello! How can I help you today?"
        elif is_general_knowledge:
            response = "I'm an AI assistant designed to help you with questions about your uploaded documents. I can analyze PDFs, DOCX files, and answer questions based on their content."
        
        print(f"Non-document question detected: '{query}' -> Direct response without RAG")
        return {"answer": response, "sources": []}
    
    # For context questions, use conversation context but don't show sources
    if is_context_question:
        print(f"Context question detected: '{query}' -> Will use conversation context without document retrieval")
        # Set a flag to indicate this is a context-only question
        use_context_only = True
    else:
        use_context_only = False

    collection_name = get_chroma_collection_name(user)

    # Retrieve user selected LLM model or default
    default_model = "openai"
    selected_model = getattr(user, "preferred_llm", default_model)

    # Instantiate corresponding llm object with error handling
    try:
        if selected_model == "gemini":
            llm_instance = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", google_api_key=settings.GEMINI_API_KEY, temperature=0.2)
        else:
            llm_instance = ChatOpenAI(model_name="gpt-4.1-mini", openai_api_key=settings.OPENAI_API_KEY, temperature=0.2)
    except Exception as e:
        # If Gemini fails, fall back to OpenAI
        print(f"Failed to initialize {selected_model} LLM: {str(e)}")
        llm_instance = ChatOpenAI(model_name="gpt-4.1-mini", openai_api_key=settings.OPENAI_API_KEY, temperature=0.2)

    # Get conversation context if chat_id is provided
    conversation_context = ""
    if chat_id:
        try:
            # Get recent messages from the chat (last 10 messages for context)
            # Exclude the current message being processed
            recent_messages = Message.objects.filter(
                chat_id=chat_id,
                chat__user_id=user.id
            ).order_by('-timestamp')[:10]
            
            # Build conversation context from recent messages
            if recent_messages:
                conversation_parts = []
                for msg in reversed(recent_messages):  # Reverse to get chronological order
                    if msg.sender_type == "user":
                        conversation_parts.append(f"User: {msg.content}")
                    elif msg.sender_type == "ai":
                        # Include sources in AI responses for better context only if sources exist
                        ai_response = msg.content
                        if hasattr(msg, 'sources') and msg.sources and len(msg.sources) > 0:
                            ai_response += f"\n[Source: 1 document referenced]"
                        conversation_parts.append(f"Assistant: {ai_response}")
                
                conversation_context = "\n".join(conversation_parts)
                print(f"Conversation context length: {len(conversation_context)} characters")
                print(f"Conversation context preview: {conversation_context[:200]}...")
        except Exception as e:
            print(f"Error getting conversation context: {str(e)}")

    # Enhance the query with conversation context
    enhanced_query = query
    if conversation_context:
        enhanced_query = f"Based on the following conversation history and the current question, provide a comprehensive answer. Consider the context from previous messages when responding:\n\nConversation History:\n{conversation_context}\n\nCurrent Question: {query}"
        print(f"Enhanced query with context length: {len(enhanced_query)} characters")

    # Handle different types of questions
    if use_context_only:
        # For context questions, use conversation context but don't retrieve documents
        print(f"Processing context-only question with conversation history")
        # Create a simple prompt for context questions
        context_prompt = f"Based on the following conversation history, answer the question naturally:\n\nConversation History:\n{conversation_context}\n\nQuestion: {query}\nAnswer:"
        
        # Use the LLM directly without RAG
        try:
            if selected_model == "gemini":
                llm_instance = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", google_api_key=settings.GEMINI_API_KEY, temperature=0.2)
            else:
                llm_instance = ChatOpenAI(model_name="gpt-4.1-mini", openai_api_key=settings.OPENAI_API_KEY, temperature=0.2)
            
            response = llm_instance.invoke(context_prompt)
            answer = response.content if hasattr(response, 'content') else str(response)
            print(f"Context question answered without document retrieval")
            return {"answer": answer, "sources": []}
        except Exception as e:
            print(f"Error processing context question: {str(e)}")
            return {"answer": "I'm sorry, I encountered an error processing your question. Please try again.", "sources": []}
    else:
        # For document questions or general questions, use RAG
        answer, sources = run_rag_query(user, enhanced_query, collection_name, llm=llm_instance)
        
        # Only include the highest matching document as source
        enhanced_sources = []
        if sources and len(sources) > 0:
            # Take only the first (highest matching) document
            best_doc = sources[0]
            print(f"Retrieved {len(sources)} documents, using highest match as source")
            
            if hasattr(best_doc, 'metadata') and best_doc.metadata:
                source_info = {
                    "content": best_doc.page_content,
                    "page": best_doc.metadata.get("page", "Unknown"),
                    "file_id": best_doc.metadata.get("file_id", "Unknown"),
                    "type": best_doc.metadata.get("type", "text")
                }
                
                # Try to get the filename from the file_id
                try:
                    if source_info["file_id"] != "Unknown":
                        file_obj = File.objects.get(id=source_info["file_id"])
                        source_info["filename"] = file_obj.filename
                    else:
                        source_info["filename"] = "Unknown"
                except File.DoesNotExist:
                    source_info["filename"] = "Unknown"
                    
                enhanced_sources.append(source_info)
            else:
                # Fallback for documents without metadata
                enhanced_sources.append({
                    "content": best_doc.page_content,
                    "page": "Unknown",
                    "file_id": "Unknown",
                    "type": "text",
                    "filename": "Unknown"
                })
        
        return {"answer": answer, "sources": enhanced_sources}


from rest_framework.permissions import IsAuthenticated
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

class UserLLMModelView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        user = request.user
        selected = getattr(user, "preferred_llm", "openai")
        
        # Get additional information about the model
        model_info = {
            "preferred_llm": selected,
            "available_models": ["openai", "gemini"],
            "current_model_details": {
                "name": selected,
                "provider": "OpenAI" if selected == "openai" else "Google",
                "model_id": "gpt-4.1-mini" if selected == "openai" else "gemini-2.5-flash-lite",
                "status": "active"
            }
        }
        
        return Response(model_info, status=status.HTTP_200_OK)

    def post(self, request):
        user = request.user
        new_model = request.data.get("preferred_llm")
        if new_model not in ("openai", "gemini"):
            return Response({"error": "Invalid LLM model. Choose 'openai' or 'gemini'."}, status=status.HTTP_400_BAD_REQUEST)
        setattr(user, "preferred_llm", new_model)
        user.save()
        return Response({"message": f"Preferred LLM model set to '{new_model}'"}, status=status.HTTP_200_OK)

def get_chroma_collection_name(user):
    username = re.sub(r'[^a-zA-Z0-9._-]', '_', user.username)
    if 3 <= len(username) <= 512 and username[0].isalnum() and username[-1].isalnum():
        return username
    return f"user_{user.id}"

# Create your views here.

from supabase import create_client, Client
from supabase.lib.client_options import ClientOptions
from gotrue.errors import AuthApiError
import os

from datetime import datetime


# SUPABASE_URL = os.getenv('SUPABASE_URL')
# SUPABASE_KEY = os.getenv('SUPABASE_KEY')
supabase: Client = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
class RegisterView(APIView):
    def post(self, request):
        data = request.data
        required = [
            'email', 'password',
            'first_name','last_name',
            'phone_number','gender','date_of_birth'
        ]
        missing = [f for f in required if not data.get(f)]
        if missing:
            return Response(
                {'error': f"Missing fields: {', '.join(missing)}"},
                status=status.HTTP_400_BAD_REQUEST
            )

        supabase = create_client(
            settings.SUPABASE_URL,
            settings.SUPABASE_SERVICE_ROLE_KEY
        )


        try:
            resp = supabase.auth.sign_up({
                'email':    data['email'],
                'password': data['password'],
                'options': {
                    'redirect_to': settings.BASE_URL_SIGNIN
                }
            })
        except AuthApiError as e:
            # If user already exists but is not verified, resend verification email
            if "User already registered" in str(e) or "already been registered" in str(e):
                try:
                    # Resend verification email for existing unverified user
                    resend_resp = supabase.auth.resend({
                        'type': 'signup',
                        'email': data['email'],
                        'options': {
                            'email_redirect_to': settings.BASE_URL_SIGNIN
                        }
                    })
                    return Response(
                        {'message': 'Verification email has been resent. Please check your email and verify your account.'},
                        status=status.HTTP_200_OK
                    )
                except Exception as resend_error:
                    return Response(
                        {'error': f'Failed to resend verification email: {str(resend_error)}'},
                        status=status.HTTP_500_INTERNAL_SERVER_ERROR
                    )
            else:
                return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)

        # At this point, resp.user is guaranteed to exist
        supa_user = resp.user  # <— has .id and .email :contentReference[oaicite:0]{index=0}

        # Push metadata into Supabase
        metadata = {
            'first_name':   data['first_name'],
            'last_name':    data['last_name'],
            'phone_number': data['phone_number'],
            'gender':       data['gender'],
            'date_of_birth': data['date_of_birth']
        }
        try:
            supabase.auth.admin.update_user_by_id(
                supa_user.id,
                {'user_metadata': metadata}
            )
        except AuthApiError as e:
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        # Mirror into Django
        try:
            dob = datetime.strptime(data['date_of_birth'], "%Y-%m-%d").date()
        except ValueError:
            dob = None

        User.objects.create_user(
            id=supa_user.id,
            email=supa_user.email,
            first_name=metadata['first_name'],
            last_name=metadata['last_name'],
            phone_number=metadata['phone_number'],
            gender=metadata['gender'],
            date_of_birth=dob
        )

        return Response(
            {'message': 'Registration successful. Please verify your email.'},
            status=status.HTTP_201_CREATED
        )


from rest_framework import generics
from rest_framework.permissions import IsAuthenticated
from .serializers import UserSerializer

class UserProfileView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, *args, **kwargs):
        supabase = create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_ROLE_KEY)
        user = request.user

        # Fetch fresh user data from Supabase using the user's UUID
        response = supabase.auth.admin.get_user_by_id(user.id)
        if response.get('error'):
            return Response({"error": response['error']['message']}, status=status.HTTP_400_BAD_REQUEST)

        raw_user_data = response.get('data')
        if not raw_user_data:
            return Response({"error": "User not found in Supabase."}, status=status.HTTP_404_NOT_FOUND)

        # Construct profile picture URL if available
        profile_picture_key = raw_user_data.get('user_metadata', {}).get('profile_picture')
        profile_picture_url = None
        bucket_name = "user-uploads"
        if profile_picture_key:
            if profile_picture_key.startswith("http"):
                profile_picture_url = profile_picture_key
            else:
                profile_picture_url = f"{settings.SUPABASE_URL}/storage/v1/object/public/{bucket_name}/{profile_picture_key}"

        # Compose all user data to return
        user_metadata = raw_user_data.get('user_metadata', {})
        bucket_name = "user-uploads"
        if profile_picture_key and not profile_picture_key.startswith("http"):
            profile_picture_url = f"{settings.SUPABASE_URL}/storage/v1/object/public/{bucket_name}/{profile_picture_key}"
            user_metadata['profile_picture'] = profile_picture_url
        elif profile_picture_key and profile_picture_key.startswith("http"):
            user_metadata['profile_picture'] = profile_picture_key
        else:
            user_metadata['profile_picture'] = None

        result = {
            "id": raw_user_data.get('id'),
            "email": raw_user_data.get('email'),
            "phone_confirmed_at": raw_user_data.get('phone_confirmed_at'),
            "email_confirmed_at": raw_user_data.get('email_confirmed_at'),
            "last_sign_in_at": raw_user_data.get('last_sign_in_at'),
            "app_metadata": raw_user_data.get('app_metadata'),
            "user_metadata": user_metadata,
            "created_at": raw_user_data.get('created_at'),
            "updated_at": raw_user_data.get('updated_at')
        }

        return Response(result, status=status.HTTP_200_OK)

    def put(self, request, *args, **kwargs):
        user = request.user
        data = request.data
        supabase = create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_ROLE_KEY)

        user_metadata = {
            "first_name": data.get('first_name', ''),
            "last_name": data.get('last_name', ''),
            "phone_number": data.get('phone_number', ''),
            "gender": data.get('gender', ''),
            "date_of_birth": data.get('date_of_birth', ''),
            "profile_picture": data.get('profile_picture', '')
        }

        updates = {
            "user_metadata": user_metadata
        }

        response = supabase.auth.admin.update_user_by_id(user.id, updates)
        if response.get('error'):
            return Response({"error": response['error']['message']}, status=status.HTTP_400_BAD_REQUEST)

        # Also update local Django user model if exists
        local_user = user
        local_user.first_name = user_metadata.get('first_name')
        local_user.last_name = user_metadata.get('last_name')
        local_user.phone_number = user_metadata.get('phone_number')
        local_user.gender = user_metadata.get('gender')
        if user_metadata.get('date_of_birth'):
            from datetime import datetime
            try:
                local_user.date_of_birth = datetime.strptime(user_metadata.get('date_of_birth'), "%Y-%m-%d").date()
            except Exception:
                local_user.date_of_birth = None
        local_user.profile_picture = user_metadata.get('profile_picture')
        local_user.save()

        return Response({"message": "User profile updated successfully."}, status=status.HTTP_200_OK)



from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework import status


class UserProfilePictureGetView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, *args, **kwargs):
        user = request.user
        profile_picture_key = user.profile_picture
        if not profile_picture_key:
            return Response({"error": "User has no profile picture."}, status=status.HTTP_404_NOT_FOUND)

        supabase_url = settings.SUPABASE_URL
        bucket_name = "user-uploads"

        # Construct public URL (assuming bucket is public or using signed URLs would require more implementation)
        image_url = f"{supabase_url}/storage/v1/object/public/{bucket_name}/{profile_picture_key}"

        return Response({"profile_picture_url": image_url}, status=status.HTTP_200_OK)


class UserProfilePictureUploadView(APIView):
    permission_classes = [IsAuthenticated]
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request, *args, **kwargs):
        user = request.user
        file_obj = request.FILES.get('profile_picture')
        if not file_obj:
            return Response({"error": "No profile_picture file provided."}, status=status.HTTP_400_BAD_REQUEST)

        folder = f"user_{user.id}"
        filename = file_obj.name
        storage_key = f"{folder}/{filename}"

        supabase = create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_ROLE_KEY)

        try:
            file_bytes = file_obj.read()

            # Upload to Supabase Storage
            res = supabase.storage.from_("user-uploads").upload(
                storage_key,
                file_bytes,
                {"content-type": file_obj.content_type}
            )
            if isinstance(res, dict) and res.get("error"):
                return Response({"error": res["error"]["message"]}, status=status.HTTP_400_BAD_REQUEST)

            # Set profile_picture field to the storage key or URL
            user.profile_picture = storage_key  # or construct a full URL if needed
            user.save(update_fields=['profile_picture'])

            # No local file to delete since file is read directly from upload

            serializer = UserSerializer(user)
            return Response(serializer.data, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class LoginView(APIView):
    def post(self, request):
        data = request.data
        email = data.get('email')
        password = data.get('password')

        if not email or not password:
            return Response({'error': 'Email and password are required.'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            response = supabase.auth.sign_in_with_password(
                {'email': email, 'password': password}
            )
        except Exception as e:
            return Response({'error': 'Failed to login. ' + str(e)}, status=status.HTTP_400_BAD_REQUEST)

        if hasattr(response, 'error') and response.error:
            return Response({'error': str(response.error)}, status=status.HTTP_400_BAD_REQUEST)

        user = response.user
        if not user:
            return Response({'error': 'Invalid login credentials.'}, status=status.HTTP_401_UNAUTHORIZED)

        return Response({
            'access_token': response.session.access_token,
            'refresh_token': response.session.refresh_token,
            'user': {'email': user.email, 'id': user.id}
        }, status=status.HTTP_200_OK)

# resetpass_url ='http://localhost:3000/reset-password'
class PasswordResetView(APIView):
  
    def post(self, request):
        email = request.data.get('email')
        if not email:
            return Response({'error': 'Email is required'}, status=status.HTTP_400_BAD_REQUEST)

        # Validate email format
        import re
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, email):
            return Response({'error': 'Please enter a valid email address.'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            # Trigger password reset email through Supabase
            # Supabase handles non-existent users gracefully and returns success for security
            response = supabase.auth.reset_password_for_email(
                email, 
                options={'redirect_to': settings.BASE_URL_RESET_PASSWORD}
            )
            
            # Always return success to prevent email enumeration attacks
            # Supabase will only send emails to registered users
            return Response({
                'message': 'If an account with this email exists, a password reset link has been sent.'
            }, status=status.HTTP_200_OK)
                
        except Exception as e:
            print(f"Password reset error: {str(e)}")
            return Response({
                'error': 'Failed to send password reset email. Please try again later.'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class PasswordResetConfirmView(APIView):
    def post(self, request):
        token = request.data.get('token')
        new_password = request.data.get('new_password')
        if not token or not new_password:
            return Response({'error': 'Token and new password are required.'}, status=status.HTTP_400_BAD_REQUEST)

        import requests
        from django.conf import settings

        project_url = getattr(settings, 'SUPABASE_URL', '').rstrip('/')
        api_key = getattr(settings, 'SUPABASE_SERVICE_ROLE_KEY', '') or getattr(settings, 'SUPABASE_ANON_KEY', '')
        if not project_url or not api_key:
            return Response({'error': 'Supabase configuration missing.'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        url = f'{project_url}/auth/v1/user'
        headers = {
            'Authorization': f'Bearer {token}',
            'apikey': api_key,
            'Content-Type': 'application/json'
        }
        payload = {'password': new_password}

        try:
            resp = requests.put(url, json=payload, headers=headers)
            if resp.status_code != 200:
                return Response({'error': resp.json().get('message', 'Failed to update password')}, status=resp.status_code)
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)

        return Response({'message': 'Password has been reset successfully.'}, status=status.HTTP_200_OK)

from gotrue.errors import AuthApiError

class VerifyOtpView(APIView):

    def post(self, request):
        access_token = request.data.get('access_token')
        if not access_token:
            return Response({'error': 'Access token is required'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            user_response = supabase.auth.get_user(access_token)
        except AuthApiError as e:
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)

        user = user_response.user
        if not user:
            return Response({'error': 'Invalid token or user not found'}, status=status.HTTP_400_BAD_REQUEST)

        if user.user_metadata.get('email_confirmed'):
            return Response({'message': 'Email verified'}, status=status.HTTP_200_OK)
        else:
            return Response({'message': 'Email not verified'}, status=status.HTTP_400_BAD_REQUEST)

import uuid
import os


class SupabaseOptions:
    def __init__(self, token):
        self.headers = {"Authorization": f"Bearer {token}"}
        self.auto_refresh_token = False
        self.persist_session = False
        self.detect_session_in_url = False
        self.storage = None
        self.flow_type = None
        self.httpx_client = None


class FileUploadView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request, *args, **kwargs):
        user = request.user
        folder = f"user_{user.id}"

        supabase = create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_ROLE_KEY)

        uploaded_file = request.FILES.get("file")
        if not uploaded_file:
            return Response({"error": "No file provided"}, status=status.HTTP_400_BAD_REQUEST)

        ext = uploaded_file.name.split(".")[-1].lower()
        
        # Validate file type
        if ext not in ['pdf', 'docx']:
            return Response(
                {"error": "Unsupported file type. Only PDF and DOCX files are supported."}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Use original filename for storage
        import re
        original_filename = uploaded_file.name
        # Sanitize filename to remove unwanted characters and avoid path traversal
        sanitized_filename = re.sub(r'[^a-zA-Z0-9_.-]', '_', original_filename)
        storage_key = f"{folder}/{sanitized_filename}"

        try:
            file_bytes = uploaded_file.read()

            # 1) Upload to Supabase
            res = supabase.storage.from_("user-uploads").upload(
                storage_key,
                file_bytes,
                {"content-type": uploaded_file.content_type}
            )
            if isinstance(res, dict) and res.get("error"):
                return Response({"error": res["error"]["message"]}, status=status.HTTP_400_BAD_REQUEST)

            # 2) Save DB row & run embedding pipeline
            with transaction.atomic():
                collection_name = get_chroma_collection_name(user)

                # Save local copy so process_and_store_file can read from disk
                file_obj = File.objects.create(
                    user=user,
                    file=uploaded_file,          # FileField writes to MEDIA_ROOT/uploads/...
                    filename=uploaded_file.name, # original name
                    chroma_collection=collection_name,
                    storage_key=storage_key
                )

            # Path on local disk (since we just saved FileField)
            file_path = file_obj.file.path

            # 3) Create embeddings (sync)
            try:
                print(f"Starting embedding process for file: {file_path}")
                process_and_store_file(user, file_path, collection_name, file_id=file_obj.id)
                print(f"Embedding process completed successfully for file: {file_path}")
            except Exception as e:
                print(f"Embedding process failed for file {file_path}: {str(e)}")
                # Delete the file from database if embedding failed
                file_obj.delete()
                raise e

            # Delete local file after processing
            import os
            if os.path.exists(file_path):
                os.remove(file_path)

            return Response(
                {
                    "message": "File uploaded & embedded",
                    "file_id": file_obj.id,
                    "storage_key": storage_key
                },
                status=status.HTTP_201_CREATED
            )

        except Exception as e:
            # Optional: clean up supabase object if DB/embedding failed
            try:
                supabase.storage.from_("user-uploads").remove([storage_key])
            except Exception:
                pass

            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

from django.conf import settings

from .serializers import UserSerializer
from rest_framework import generics
from rest_framework.permissions import IsAuthenticated

class UserProfileView(generics.RetrieveUpdateAPIView):
    serializer_class = UserSerializer
    permission_classes = [IsAuthenticated]

    def get_object(self):
        return self.request.user


class UserFilesView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, *args, **kwargs):
        supabase = create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_ROLE_KEY)

        folder_name = f"user_{request.user.id}"

        try:
            files = supabase.storage.from_("user-uploads").list(
                path=folder_name,
                options={"limit": 100, "offset": 0, "sortBy": {"column": "name", "order": "asc"}},
            )

            # helper to build path once
            def full_path(name: str) -> str:
                return f"{folder_name}/{name}"

            # ----- OPTION A: bucket is PUBLIC -----
            # get_public_url = supabase.storage.from_("user-uploads").get_public_url

            # ----- OPTION B: bucket is PRIVATE (recommended) -----
            create_signed_url = supabase.storage.from_("user-uploads").create_signed_url
            EXPIRES_IN = 3600  # seconds

            files_info = []
            for f in files:
                if not f["name"].lower().endswith((".pdf", ".docx")):
                    continue

                path = full_path(f["name"])

                # PUBLIC:
                # url = get_public_url(path)

                # PRIVATE (signed):
                signed = create_signed_url(path, EXPIRES_IN)
                url = signed.get("signedURL") or signed.get("signed_url")  # lib versions differ

                files_info.append({
                    "name": f["name"],
                    "updated_at": f.get("updated_at"),
                    "created_at": f.get("created_at"),
                    "id": f.get("id"),
                    "url": url,
                })

            return Response({"files": files_info}, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class UserFileDeleteView(APIView):
    permission_classes = [IsAuthenticated]

    # def delete(self, request, file_id):
    #     user = request.user
    #     supabase_token = request.headers.get("Authorization", "").replace("Bearer ", "")

    #     try:
    #         # Initialize supabase client with service role key
    #         supabase = create_client(
    #             settings.SUPABASE_URL,
    #             settings.SUPABASE_SERVICE_ROLE_KEY,
    #             options={
    #                 "auto_refresh_token": True,
    #                 "persist_session": True,
    #                 "detect_session_in_url": True,
    #                 "storage": None,
    #             },
    #         )

    #         # Validate file ownership by id
    #         file_obj = File.objects.filter(id=file_id, user=user).first()
    #         if not file_obj:
    #             return Response({"error": "File not found or you do not have permission."}, status=status.HTTP_404_NOT_FOUND)

    #         file_path = file_obj.storage_key

    #         # Remove file from Supabase storage
    #         removed = supabase.storage.from_("user-uploads").remove([file_path])
    #         if isinstance(removed, dict) and removed.get("error"):
    #             return Response({"error": removed["error"]["message"]}, status=status.HTTP_400_BAD_REQUEST)

    #         # Remove corresponding embeddings from ChromaDB
    #         try:
    #             collection = chroma_client.get_collection(name="user_files")  # Change to your collection name
    #             collection.delete(where={"file_id": str(file_id)})
    #         except Exception:
    #             pass  # Log error if needed

    #         # Remove DB record for the file
    #         file_obj.delete()

    #         return Response({"message": "File deleted."}, status=status.HTTP_200_OK)

    #     except Exception as e:
    #         return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    # permission_classes = [IsAuthenticated]

    def delete(self, request, file_name: str, *args, **kwargs):
        # Server-side key (or a valid Supabase user JWT in another header if you insist)
        supabase = create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_ROLE_KEY)

        user = request.user
        folder = f"user_{user.id}"
        file_path = f"{folder}/{file_name}"

        # (Optional) simple path traversal guard
        if "/" in file_name or ".." in file_name:
            return Response({"error": "Invalid file name."}, status=status.HTTP_400_BAD_REQUEST)

        try:
            # (Optional) verify ownership before delete
            # Cheap check using list; for large folders use your DB instead
            files = supabase.storage.from_("user-uploads").list(folder)
            if not any(f["name"] == file_name for f in files):
                return Response({"error": "File not found."}, status=status.HTTP_404_NOT_FOUND)

            # Remove from storage
            removed = supabase.storage.from_("user-uploads").remove([file_path])
            # v2: returns list of dicts; v1: dict with data/error
            if isinstance(removed, dict) and removed.get("error"):
                return Response({"error": removed["error"]["message"]}, status=status.HTTP_400_BAD_REQUEST)

            # Remove from DB (wrap to keep consistency)
            with transaction.atomic():
                from .models import File
                file_obj = File.objects.filter(user=user, storage_key=file_path).first()
                if file_obj:
                    if file_obj.chroma_collection:
                        try:
                            from langchain_chroma import Chroma
                        except ImportError:
                            from langchain_community.vectorstores import Chroma
                        import logging
                        user_chroma_dir = get_user_chroma_dir(user.id)
                        vs = Chroma(
                            collection_name=file_obj.chroma_collection,
                            embedding_function=text_emb,
                            persist_directory=user_chroma_dir,
                        )
                        try:
                            vs.delete(where={"file_id": file_obj.id})
                            #vs.persist()
                        except Exception as e:
                            logging.error(f"Failed to delete embeddings for file_id {file_obj.id}: {e}")
                            return Response({"error": "Failed to delete file embeddings."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
                    # Delete local file
                    file_obj.file.delete(save=False)
                    file_obj.delete()

            return Response({"message": "File deleted."}, status=status.HTTP_200_OK)
        except Exception as e:
            # log.exception("Supabase delete failed")
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

 

from rest_framework.generics import DestroyAPIView

class FileDeleteView(DestroyAPIView):
    permission_classes = [IsAuthenticated]
    queryset = File.objects.all()
    serializer_class = FileSerializer

    def perform_destroy(self, instance):
        # Delete file from Supabase Storage
        try:
            if getattr(instance, 'storage_key', None):
                supabase.storage.from_('user-uploads').remove([instance.storage_key])
        except Exception as e:
            pass  # Log error if needed
        # Delete from local storage and database
        instance.file.delete(save=False)
        instance.delete()


class FileListView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        user_id = self.request.user.id
        files = File.objects.filter(user_id=user_id)
        serializer = FileSerializer(files, many=True)
        return Response(serializer.data)

class FileDeleteView(APIView):
    permission_classes = [IsAuthenticated]

    def delete(self, request, pk):
        user_id = self.request.user.id
        try:
            file_obj = File.objects.get(pk=pk, user_id=user_id)
        except File.DoesNotExist:
            return Response({'error': 'File not found.'}, status=404)
        # Remove from ChromaDB using LangChain Chroma vector store
        if file_obj.chroma_collection:
            try:
                from langchain_chroma import Chroma
            except ImportError:
                from langchain_community.vectorstores import Chroma
            user_chroma_dir = get_user_chroma_dir(user_id)
            vs = Chroma(
                collection_name=file_obj.chroma_collection,
                embedding_function=text_emb,
                persist_directory=user_chroma_dir,
            )
            vs.delete(where={"file_id": file_obj.id})
            #vs.persist()
            #vs.persist()  # Add this line to persist deletion
        # Delete file from storage and DB
        file_obj.file.delete(save=False)
        file_obj.delete()
        return Response({'message': 'File deleted.'}, status=204)

class ChatListCreateView(generics.ListCreateAPIView):
    serializer_class = ChatSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        user = self.request.user  # this is a Django User instance now
        user_id = user.id
        # Sort chats by the latest updated timestamp (most recent first)
        return Chat.objects.filter(user_id=user_id).order_by('-updated_at', '-created_at')

    def perform_create(self, serializer):
        user = self.request.user  # this is a Django User instance now
        user_id = user.id
        serializer.save(user_id=user_id)


class ChatUpdateDeleteView(APIView):
    permission_classes = [IsAuthenticated]

    def get_object(self, chat_id, user):
        return get_object_or_404(Chat, id=chat_id, user_id=user.id)

    def patch(self, request, chat_id):
        chat = self.get_object(chat_id, request.user)
        title = request.data.get('title')
        if not title or not title.strip():
            return Response({'error': 'Title is required.'}, status=status.HTTP_400_BAD_REQUEST)
        
        # Update title and explicitly update the updated_at field
        chat.title = title.strip()
        chat.updated_at = timezone.now()
        chat.save()
        
        serializer = ChatSerializer(chat)
        return Response(serializer.data, status=status.HTTP_200_OK)

    def delete(self, request, chat_id):
        chat = self.get_object(chat_id, request.user)
        chat.delete()
        return Response({'message': 'Chat deleted.'}, status=status.HTTP_204_NO_CONTENT)


def serialize_message(m: Message):
    return {
        "id": m.id,
        "chat_id": str(m.chat_id),
        "sender_id": str(m.sender_id) if m.sender_id else None,
        "sender_type": m.sender_type,
        "content": m.content,
        "timestamp": m.timestamp.isoformat(),
    }
    
class MessageListCreateView(generics.ListCreateAPIView):
    serializer_class = MessageSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        chat_id = self.kwargs["chat_id"]
        return (Message.objects
                .filter(chat_id=chat_id, chat__user_id=self.request.user.id)
                .order_by("timestamp"))

    def list(self, request, *args, **kwargs):
        msgs = self.get_queryset()
        data = [{
            "id": m.id,
            "chat_id": str(m.chat_id),
            "user_message": m.content if m.sender_type == "user" else None,
            "ai_response":  m.content if m.sender_type == "ai"   else None,
            "rag_sources":  m.sources  if m.sender_type == "ai"   else None,
        } for m in msgs]
        return Response({"messages": data}, status=status.HTTP_200_OK)

    def create(self, request, *args, **kwargs):
        try:
            user    = request.user
            chat_id = self.kwargs["chat_id"]
            chat    = get_object_or_404(Chat, id=chat_id, user_id=user.id)

            # 1) Save user message
            ser = self.get_serializer(data=request.data)
            ser.is_valid(raise_exception=True)
            user_msg = ser.save(sender=user, sender_type="user", chat=chat)

            # 2) RAG -> AI message
            rag = run_rag_pipeline(user, user_msg.content, chat_id=chat_id)
            ai_msg = Message.objects.create(
                chat=chat,
                sender=None,
                sender_type="ai",
                content=rag["answer"],
                sources=rag.get("sources", [])
            )

            # 3) Update chat's updated_at field to reflect new activity
            chat.updated_at = timezone.now()
            chat.save()

            # 4) Flat response
            return Response(
                {
                    "id": ai_msg.id,                     # or user_msg.id / chat_id — your choice
                    "chat_id": str(chat.id),
                    "user_message": user_msg.content,
                    "ai_response":  ai_msg.content,
                    "rag_sources": rag.get("sources", [])  # optional - now includes filename, page, and content
                },
                status=status.HTTP_201_CREATED
            )
        except Exception as e:
            print(f"Error in message creation: {str(e)}")
            return Response(
                {"error": "Failed to process message. Please try again."}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from supabase import create_client
from django.conf import settings

class SendVerificationEmailView(APIView):
    """
    POST endpoint to send a verification email to a user if not verified in Supabase.
    Accepts 'email' in POST data (optional if authenticated user).
    """
    def post(self, request):
        email = request.data.get('email')
        if not email and request.user and hasattr(request.user, 'email'):
            email = request.user.email
        if not email:
            return Response({'error': 'Email is required.'}, status=status.HTTP_400_BAD_REQUEST)
        supabase = create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_ROLE_KEY)
        # Check user status in Supabase
        try:
            # list_users() does not accept email param; fetch all and filter
            users = supabase.auth.admin.list_users() or []
            user_obj = None
            for u in users:
                # Support both dict and object for user
                email_val = u['email'] if isinstance(u, dict) else getattr(u, 'email', None)
                if email_val and email_val.lower() == email.lower():
                    user_obj = u
                    break
            if not user_obj:
                return Response({'error': 'User not found.'}, status=status.HTTP_404_NOT_FOUND)
            # Optionally, get latest user details by id
            user_id = user_obj['id'] if isinstance(user_obj, dict) else getattr(user_obj, 'id', None)
            user_details = supabase.auth.admin.get_user_by_id(user_id)
            # user_details may be dict or object
            email_confirmed_at = None
            if isinstance(user_details, dict):
                data = user_details.get('data', {})
                email_confirmed_at = data.get('email_confirmed_at')
            else:
                email_confirmed_at = getattr(user_details, 'email_confirmed_at', None)
            if email_confirmed_at:
                return Response({'error': 'User already verified.'}, status=status.HTTP_400_BAD_REQUEST)
            # Resend verification email for existing unverified user
          
            try:
                # print("email",email)
                # print("BASE_URL_SIGNIN",settings.BASE_URL_SIGNIN)
                resend_resp = supabase.auth.resend({
                    'type': 'signup',
                    'email': email,
                    'options': {
                        'email_redirect_to': settings.BASE_URL_SIGNIN
                    }
                })
                if resend_resp is not None:
                    # Check for error or success in resend_resp
                    error = None
                    if isinstance(resend_resp, dict):
                        error = resend_resp.get('error')
                    else:
                        error = getattr(resend_resp, 'error', None)
                    if error:
                        return Response({'error': f'Verification email not sent: {error}'}, status=status.HTTP_400_BAD_REQUEST)
                    return Response(
                        {'message': 'Verification email has been resent. Please check your email and verify your account.'},
                        status=status.HTTP_200_OK
                    )
                else:
                    return Response({'error': 'Failed to resend verification email'}, status=status.HTTP_400_BAD_REQUEST)
            except Exception as resend_error:
                return Response(
                    {'error': f'Failed to resend verification email: {str(resend_error)}'},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )
        except Exception as e:
            return Response({'error': f'Failed to send verification email: {str(e)}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class TokenRefreshView(APIView):
    def post(self, request):
        refresh_token = request.data.get("refresh_token")
        if not refresh_token:
            return Response({"error": "refresh_token is required"}, status=status.HTTP_400_BAD_REQUEST)

        supabase_url = settings.SUPABASE_URL
        supabase_key = settings.SUPABASE_KEY
        supabase = create_client(supabase_url, supabase_key)

        try:
            data = supabase.auth.refresh_session(refresh_token)
            if data.session and data.session.access_token and data.session.refresh_token:
                return Response({
                    "access_token": data.session.access_token,
                    "refresh_token": data.session.refresh_token
                })
            else:
                return Response({"error": "Failed to refresh token"}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)