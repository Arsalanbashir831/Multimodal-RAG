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
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# Initialize ChromaDB client (simple local setup)
chroma_client = chromadb.Client(chromadb.config.Settings(
    persist_directory="chroma_data"
))

# Load BLIP for image captioning
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
blip_model.to(device)

os.environ["OPENAI_API_KEY"] = "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"
text_emb = OE(model="text-embedding-3-small")
#text_emb = OE(model="text-embedding-3-small")
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
def get_user_chroma_dir(user):
    return os.path.join("chroma_data", f"user_{user.id}")

# --- File upload: extract, caption, embed, and store in per-user Chroma ---
def process_and_store_file(user, file_path, collection_name):
    pages = extract_pdf_pages(file_path)
    texts, metas = [], []
    for p in pages:
        if p["text"].strip():
            texts.append(p["text"])
            metas.append({"type": "text", "page": p["page"]})
        for im in p["images"]:
            caption = describe_image(im)
            texts.append(caption)
            metas.append({
                "type": "image_caption",
                "page": p["page"],
                "path": im,
                "caption": caption
            })
    if not texts:
        raise ValueError("PDF contained no text or images.")
    user_chroma_dir = get_user_chroma_dir(user)
    vs = Chroma.from_texts(
        texts=texts,
        embedding=text_emb,
        metadatas=metas,
        collection_name=collection_name,
        persist_directory=user_chroma_dir,
    )
    vs.persist()
    print(f"[DEBUG] Stored {len(texts)} documents for user {user.id} in {user_chroma_dir}")
    for i, t in enumerate(texts[:3]):
        print(f"[DEBUG] Example doc {i+1}: {t[:100]}")

# --- RAG query: load per-user Chroma and run RetrievalQA ---
def run_rag_query(user, query, collection_name, k=3):
    user_chroma_dir = get_user_chroma_dir(user)
    if not Path(user_chroma_dir).exists():
        return "No documents found for this user.", []
    vs = Chroma(
        persist_directory=user_chroma_dir,
        embedding_function=text_emb,
        collection_name=collection_name,
    )
    rag_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model_name="gpt-4o-mini", temperature=0.2),
        retriever=vs.as_retriever(search_kwargs={"k": k}),
        return_source_documents=True
    )
    result = rag_chain({"query": query})
    answer = result["result"]
    sources = result.get("source_documents", [])
    print(f"[DEBUG] Retrieved {len(sources)} source documents for user {user.id}")
    for i, doc in enumerate(sources[:3]):
        print(f"[DEBUG] Source {i+1}: {doc.page_content[:100]}")
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

class RegisterView(APIView):
    def post(self, request):
        serializer = UserRegistrationSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response({'message': 'User registered successfully.'}, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class FileUploadView(APIView):
    permission_classes = [IsAuthenticated]
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request):
        uploaded_file = request.FILES.get('file')
        if not uploaded_file:
            return Response({'error': 'No file provided.'}, status=400)
        filename = uploaded_file.name
        user = request.user
        collection_name = get_chroma_collection_name(user)
        # Save file in DB
        file_obj = File.objects.create(user=user, file=uploaded_file, filename=filename, chroma_collection=collection_name)
        # Save file to disk for processing
        file_path = file_obj.file.path
        process_and_store_file(user, file_path, collection_name)
        return Response(FileSerializer(file_obj).data, status=201)

class FileListView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        files = File.objects.filter(user=request.user)
        serializer = FileSerializer(files, many=True)
        return Response(serializer.data)

class FileDeleteView(APIView):
    permission_classes = [IsAuthenticated]

    def delete(self, request, pk):
        try:
            file_obj = File.objects.get(pk=pk, user=request.user)
        except File.DoesNotExist:
            return Response({'error': 'File not found.'}, status=404)
        # Remove from ChromaDB
        if file_obj.chroma_collection:
            collection = chroma_client.get_or_create_collection(file_obj.chroma_collection)
            collection.delete(ids=[str(file_obj.id)])
        # Delete file from storage and DB
        file_obj.file.delete(save=False)
        file_obj.delete()
        return Response({'message': 'File deleted.'}, status=204)

class ChatListCreateView(generics.ListCreateAPIView):
    serializer_class = ChatSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return Chat.objects.filter(user=self.request.user)

    def perform_create(self, serializer):
        serializer.save(user=self.request.user)

class MessageListCreateView(generics.ListCreateAPIView):
    serializer_class = MessageSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        chat_id = self.kwargs['chat_id']
        return Message.objects.filter(chat__id=chat_id, chat__user=self.request.user)

    def perform_create(self, serializer):
        chat_id = self.kwargs['chat_id']
        chat = Chat.objects.get(id=chat_id, user=self.request.user)
        message = serializer.save(sender=self.request.user, chat=chat)
        # Trigger RAG pipeline on user message
        rag_result = run_rag_pipeline(self.request.user, message.content)
        rag_answer = rag_result["answer"]
        rag_sources = rag_result["sources"]
        # Save the RAG response as a system message
        Message.objects.create(chat=chat, sender=self.request.user, content=rag_answer)
        # Attach sources to the response for debugging
        self._rag_sources = rag_sources

    def create(self, request, *args, **kwargs):
        response = super().create(request, *args, **kwargs)
        # Add sources to the response if available
        if hasattr(self, '_rag_sources'):
            response.data["rag_sources"] = self._rag_sources
        return response
