from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import UserRegistrationSerializer
from rest_framework.permissions import IsAuthenticated
from rest_framework.parsers import MultiPartParser, FormParser
from .models import File
from .serializers import FileSerializer
import chromadb
import uuid
from .models import Chat, Message
from .serializers import ChatSerializer, MessageSerializer
from rest_framework import generics
import fitz  # PyMuPDF
from PIL import Image
import numpy as np
from io import BytesIO
from langchain_openai import OpenAIEmbeddings as OE
from sentence_transformers import SentenceTransformer
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
import os
from tqdm import tqdm
import openai
import re
from pathlib import Path
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()


# Initialize ChromaDB client (simple local setup)
chroma_client = chromadb.Client(chromadb.config.Settings(
    persist_directory="chroma_data"
))

# Load BLIP for image captioning
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast=True)
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
blip_model.to(device)

text_emb = OE(model="text-embedding-3-small")
image_emb = SentenceTransformer("clip-ViT-B-32")

# --- PDF and Image Processing ---
def describe_image(image_path):
    raw_image = Image.open(image_path).convert('RGB')
    inputs = processor(raw_image, return_tensors="pt").to(device)
    out = blip_model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

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

def build_embeddings(pages, text_batch_size=50):
    texts, metadatas = [], []
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
    pages = extract_pdf_pages(file_path)
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
                "path": im,
                "caption": caption,
                "file_id": file_id
            })
            ids.append(f"{file_id}_{chunk_idx}")
            chunk_idx += 1
    if not texts:
        raise ValueError("PDF contained no text or images.")
    user_chroma_dir = get_user_chroma_dir(user)
    vs = Chroma(
        collection_name=collection_name,
        embedding_function=text_emb,
        persist_directory=user_chroma_dir,
    )
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        vs.add_texts(
            texts=texts[i:i+batch_size],
            metadatas=metas[i:i+batch_size],
            ids=ids[i:i+batch_size]
        )
    vs.persist()
    print(f"[DEBUG] Stored {len(texts)} documents for user {user.id} in {user_chroma_dir}")
    for i, t in enumerate(texts[:3]):
        print(f"[DEBUG] Example doc {i+1}: {t[:100]}")

# --- RAG query: load per-user Chroma and run RetrievalQA ---
def run_rag_query(user, query, collection_name, k=5):
    user_chroma_dir = get_user_chroma_dir(user)
    if not Path(user_chroma_dir).exists():
        return "No documents found for this user.", []
    vs = Chroma(
        persist_directory=user_chroma_dir,
        embedding_function=text_emb,
        collection_name=collection_name,
    )
    prompt_template = (
        "You are a helpful assistant. Make sense of the context and answer the question. "
        "If unsure, say 'I don't know at end.'\n\n"
        "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    rag_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model_name="gpt-4.1-mini", temperature=0.2),
        retriever=vs.as_retriever(search_kwargs={"k": k}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    result = rag_chain({"query": query})
    answer = result["result"]
    sources = result.get("source_documents", [])
    print(f"[DEBUG] Retrieved {len(sources)} source documents for user {user.id}")
    for i, doc in enumerate(sources[:3]):
        print(f"[DEBUG] Source {i+1}: {doc.page_content[:100]} (page: {doc.metadata.get('page')})")
    return answer, sources

# --- Update run_rag_pipeline to return both answer and sources ---
def run_rag_pipeline(user, query):
    collection_name = get_chroma_collection_name(user)
    answer, sources = run_rag_query(user, query, collection_name)
    # Return both answer and sources for debugging
    return {"answer": answer, "sources": [doc.page_content for doc in sources]}

def get_chroma_collection_name(user):
    username = re.sub(r'[^a-zA-Z0-9._-]', '_', user.username)
    if 3 <= len(username) <= 512 and username[0].isalnum() and username[-1].isalnum():
        return username
    return f"user_{user.id}"

# Create your views here.

from supabase import create_client, Client
import os

SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

class RegisterView(APIView):
    def post(self, request):
        data = request.data
        email = data.get('email')
        password = data.get('password')
        username = data.get('username')

        if not email or not password or not username:
            return Response({'error': 'Email, password, and username are required.'}, status=status.HTTP_400_BAD_REQUEST)

        # Check if user already exists by trying to sign in
        try:
            sign_in_response = supabase.auth.sign_in_with_password({'email': email, 'password': password})
            # If sign_in is successful, user exists, ask to login
            if sign_in_response.user:
                return Response({'error': 'User already registered. Please login instead.'}, status=status.HTTP_400_BAD_REQUEST)
        except Exception:
            # If login fails, likely user not registered yet, proceed to sign up
            pass
        
        # Check if user already exists by trying to sign in
        try:
            sign_in_response = supabase.auth.sign_in_with_password({
                'email': email,
                'password': password
            })
            if sign_in_response.user is not None:
                return Response({'error': 'User already registered, please login instead.'}, status=status.HTTP_400_BAD_REQUEST)
        except AuthApiError as e:
            # If sign in fails due to user not existing, proceed with registration
            if 'Invalid login credentials' not in str(e):
                return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)
        # Proceed with sign up
        try:
            response = supabase.auth.sign_up(
                {
                    'email': email,
                    'password': password,
                    'options': {'data': {'username': username}}
                }
            )
        except AuthApiError as e:
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)

        if hasattr(response, 'error') and response.error:
            return Response({'error': str(response.error)}, status=status.HTTP_400_BAD_REQUEST)

        return Response({'message': 'User registered successfully. Please verify your email.'}, status=status.HTTP_201_CREATED)


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


class PasswordResetView(APIView):
    def post(self, request):
        email = request.data.get('email')
        if not email:
            return Response({'error': 'Email is required'}, status=status.HTTP_400_BAD_REQUEST)

        # Trigger password reset email through Supabase
        response = supabase.auth.reset_password_for_email(email)

        if hasattr(response, 'error') and response.error:
            return Response({'error': str(response.error)}, status=status.HTTP_400_BAD_REQUEST)

        return Response({'message': 'Password reset email sent'}, status=status.HTTP_200_OK)


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

class FileUploadView(APIView):
    permission_classes = [IsAuthenticated]
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request):
        uploaded_file = request.FILES.get("file")
        if not uploaded_file:
            return Response({"error": "No file provided."}, status=400)

        user = request.user
        filename = uploaded_file.name

        # Save file locally first so PyMuPDF can read it
        file_obj = File.objects.create(user=user, file=uploaded_file, filename=filename)
        file_path = file_obj.file.path

        # Get user access token to initialize supabase client for this request
        access_token = None
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            access_token = auth_header[len("Bearer "):].strip()

        if access_token is None:
            return Response({"error": "Authorization token missing"}, status=401)

        from supabase import create_client

        supabase_user = create_client(SUPABASE_URL, access_token)

        # Upload file to Supabase Storage using user-specific folder with user's supabase client
        try:
            file_key = f"user_{user.id}/{uuid.uuid4()}_{filename}"
            with open(file_path, "rb") as f:
                supabase_user.storage.from_('user-uploads').upload(file_key, f)
            # Save the storage key to the model
            file_obj.storage_key = file_key
            file_obj.save()
        except Exception as e:
            return Response({"error": f"Failed to upload to Supabase Storage: {str(e)}"}, status=500)

        collection_name = get_chroma_collection_name(user)
        file_obj.chroma_collection = collection_name
        file_obj.save()

        process_and_store_file(user, file_path, collection_name, file_id=file_obj.id)

        return Response(FileSerializer(file_obj).data, status=201)


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
            from langchain_community.vectorstores import Chroma
            user_chroma_dir = get_user_chroma_dir(user_id)
            vs = Chroma(
                collection_name=file_obj.chroma_collection,
                embedding_function=text_emb,
                persist_directory=user_chroma_dir,
            )
            vs.delete(where={"file_id": file_obj.id})
            vs.persist()
            vs.persist()  # Add this line to persist deletion
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
        return Chat.objects.filter(user_id=user_id)

    def perform_create(self, serializer):
        user = self.request.user  # this is a Django User instance now
        user_id = user.id
        serializer.save(user_id=user_id)

class MessageListCreateView(generics.ListCreateAPIView):
    serializer_class = MessageSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        chat_id = self.kwargs['chat_id']
        user = self.request.user  # this is a Django User instance now
        user_id = user.id
        return Message.objects.filter(chat_id=chat_id, chat__user_id=user_id)

    def perform_create(self, serializer):
        chat_id = self.kwargs['chat_id']
        user = self.request.user  # this is a Django User instance now
        user_id = user.id
        chat = Chat.objects.get(id=chat_id, user_id=user_id)
        message = serializer.save(sender_id=user_id, chat_id=chat_id)
        # Trigger RAG pipeline on user message
        rag_result = run_rag_pipeline(user, message.content)
        rag_answer = rag_result["answer"]
        rag_sources = rag_result["sources"]
        # Save the RAG response as a system message
        Message.objects.create(chat_id=chat_id, sender_id=user_id, content=rag_answer)
        # Attach sources to the response for debugging
        self._rag_sources = rag_sources

    def create(self, request, *args, **kwargs):
        response = super().create(request, *args, **kwargs)
        # Add sources to the response if available
        if hasattr(self, '_rag_sources'):
            response.data["rag_sources"] = self._rag_sources
        return response
