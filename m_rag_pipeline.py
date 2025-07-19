pip install pymupdf langchain faiss-cpu openai tiktoken langchain-community

import os
import fitz  # PyMuPDF
from PIL import Image
import numpy as np
from io import BytesIO
from tqdm import tqdm
from langchain.embeddings import OpenAIEmbeddings as OE
from langchain.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

# Load BLIP for image captioning
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
blip_model.to(device)

# API Key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", userdata.get('key'))

# Embedding models
text_emb = OE(model="text-embedding-3-small")
image_emb = SentenceTransformer("clip-ViT-B-32")

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
                if pix.colorspace.n not in [1, 3]:  # Not grayscale or RGB
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                buf = pix.tobytes("png")
                path = f"{image_dir}/pg{i}_img{xref}.png"
                Image.open(BytesIO(buf)).save(path)
                imgs.append(path)
            except Exception as e:
                #print(f"Skipping image {xref} on page {i}: {e}")
                print("", end="")

        pages.append({"page": i, "text": txt, "images": imgs})
    return pages

def build_embeddings(pages, text_batch_size=50):
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
    if texts:  # Only attempt to embed if there is text or captions
        for i in tqdm(range(0, len(texts), text_batch_size), desc="Embedding text and captions"):
            batch = texts[i : i + text_batch_size]
            text_vecs.extend(text_emb.embed_documents(batch))
        text_vecs = np.array(text_vecs)

    return text_vecs, texts, metadatas

def create_vectorstore(pages, index_path="mm_faiss"):
    idx, docs, metas = build_embeddings(pages)
    if not docs:  # Check if there are any documents to embed
        print("No text or images found in the document.")
        return None  # Return None if no vector store is created
    text_embedding_pairs = list(zip(docs, idx))

    vs = FAISS.from_embeddings(
        text_embedding_pairs,
        embedding=text_emb,
        metadatas=metas
    )
    vs.save_local(index_path)
    return vs

def load_or_create(pdf_path):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"The file {pdf_path} was not found.")
    if not pdf_path.lower().endswith(".pdf"):
        raise ValueError("The input file must be a PDF.")

    if os.path.exists("mm_faiss/index.faiss"):
        print("Loading existing FAISS index.")
        vs = FAISS.load_local("mm_faiss", text_emb, allow_dangerous_deserialization=True)
    else:
        print("Creating new FAISS index.")
        pages = extract_pdf_pages(pdf_path)
        vs = create_vectorstore(pages)
    return vs

def hybrid_rag_chain(vs, k=5):
    if vs is None:
        print("No vector store available to create RAG chain.")
        return None
    retriever = vs.as_retriever(search_kwargs={"k": k})
    llm = ChatOpenAI(model_name="gpt-4.1-mini", temperature=0.2)
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

def query_rag(rag, user_q):
    if rag is None:
        print("RAG chain not initialized.")
        return "Could not process the query.", []
    result = rag({"query": user_q})
    return result["result"], result["source_documents"]

# MAIN
if __name__ == "__main__":
    # Replace with your actual PDF path
    path = "/content/drive/MyDrive/AI Summary Trial Run/rexing combined missed variable rate and flow doesnt add up-sm.pdf"
    query = "What are the key takeaways from the document?"

    try:
        vs = load_or_create(path)
        rag = hybrid_rag_chain(vs, k=5)
        answer, docs = query_rag(rag, query)

        print("\n Answer:\n", answer)
        print("\n Retrieved Sources:")
        for d in docs:
            print(d.metadata, d.page_content[:100], "..." if len(d.page_content) > 100 else "")
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")