# Django Retrieval-Augmented Generation (RAG) Application

## Overview
This project is a Django-based Retrieval-Augmented Generation (RAG) backend that enables users to upload PDF files, extract both text and image captions, store them in a per-user ChromaDB vector store, and interact with a chat interface powered by OpenAI LLMs. The system supports JWT authentication, multi-modal ingestion (text and images), and robust context retrieval for question answering.

## Features
- **User Authentication:** JWT-based registration and login (SimpleJWT)
- **File Upload:** Upload PDFs, extract text and image captions (BLIP)
- **Vector Storage:** Store embeddings in per-user ChromaDB directories
- **Chat & Messaging:** Create chats, send messages, and receive context-aware LLM answers
- **Advanced Chunking:** Overlapping chunking for improved retrieval
- **API Documentation:** OpenAPI schema available for Postman/Swagger
- **PostgreSQL Database:** Stores users, files, chats, and messages
- **Media Management:** Uploaded files and images stored in `media/` and `imgs/`

## Tech Stack
- Django, Django REST Framework, SimpleJWT
- LangChain (Chroma, RetrievalQA, OpenAIEmbeddings, PromptTemplate, RecursiveCharacterTextSplitter)
- ChromaDB (per-user, persistent)
- OpenAI (embeddings and LLM)
- BLIP (image captioning)
- PostgreSQL

## Directory Structure
```
M_RAG_APPC/
├── core/                # Main Django app (models, views, serializers)
├── rag_app/             # Django project settings
├── chroma_data/         # Per-user ChromaDB vector stores (ignored by git)
├── imgs/                # Extracted images from PDFs (ignored by git)
├── media/               # Uploaded files (ignored by git)
├── requierments.txt     # Python dependencies
├── README.md            # Project documentation
└── manage.py            # Django management script
```

## Setup Instructions

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd M_RAG_APPC
```

### 2. Create and Activate a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requierments.txt
```

### 4. Configure Environment Variables
- Set your OpenAI API key in the environment or in `core/views.py` (for development only).
- Update PostgreSQL credentials in `rag_app/settings.py` as needed.

### 5. Run Migrations
```bash
python manage.py migrate
```

### 6. Create a Superuser (Optional)
```bash
python manage.py createsuperuser
```

### 7. Run the Development Server
```bash
python manage.py runserver
```

## API Usage

### Authentication
- **Register:** `POST /register/` (username, email, password)
- **Login:** `POST /token/` (username, password) → returns access/refresh tokens

### File Management
- **Upload File:** `POST /files/upload/` (multipart/form-data, field: `file`)
- **List Files:** `GET /files/`
- **Delete File:** `DELETE /files/<id>/delete/`

### Chat & Messaging
- **Create Chat:** `POST /chats/` (title)
- **List Chats:** `GET /chats/`
- **Send Message:** `POST /chats/<chat_id>/messages/` (content)
- **List Messages:** `GET /chats/<chat_id>/messages/`

### API Documentation
- **OpenAPI Schema:** `GET /schema/` (importable in Postman/Swagger)

## Vector Store & Data Management
- **ChromaDB:** Each user has a separate vector store in `chroma_data/user_<id>/`.
- **File Deletion:** When a file is deleted, all its vectors are removed from ChromaDB.
- **Media:** Uploaded files and extracted images are stored in `media/` and `imgs/` (both git-ignored).

## Development & Contribution
- Follow PEP8 and Django best practices.
- Use pull requests for new features and bug fixes.
- Update tests in `core/tests.py` as needed.

## Security Notes
- Do not commit `.env` files, API keys, or user data.
- Production deployments should set `DEBUG = False` and use secure credentials.

## License
MIT License (add your own if different)

## Acknowledgements
- [Django](https://www.djangoproject.com/)
- [LangChain](https://python.langchain.com/)
- [ChromaDB](https://www.trychroma.com/)
- [OpenAI](https://platform.openai.com/)
- [BLIP](https://huggingface.co/Salesforce/blip-image-captioning-base) 