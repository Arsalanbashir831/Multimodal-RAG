# Multimodal-RAG

This project is a multimodal retrieval augmented generation (RAG) system that integrates various data modalities for enhanced information retrieval and generation.

## Features

- Upload and manage user files securely with Supabase Storage integration.
- Store and retrieve file embeddings efficiently using ChromaDB vector store.
- User profile management including profile picture uploads.
- REST API endpoints for file handling, user profiles, and authentication.
- Support for PDF file filtering in user file listings.
- Automatic cleanup of temporary files to optimize storage usage.

## Setup

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd Multimodal-RAG
   ```

2. Create and activate a Python virtual environment:

   ```bash
   python -m venv venv
   venv\Scripts\activate  # On Windows
   # source venv/bin/activate  # On Linux/macOS
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the root directory with the following environment variables:

   ```ini
   SUPABASE_URL=your_supabase_url
   SUPABASE_KEY=your_supabase_key
   OPENAI_API_KEY=your_openai_api_key
   DB_NAME=your_db_name
   DB_USER=your_db_user
   DB_PASSWORD=your_db_password
   DB_HOST=your_db_host
   DB_PORT=your_db_port
   ```

   Replace the placeholders with your actual credentials.

5. Apply database migrations:

   ```bash
   python manage.py migrate
   ```

6. Run the development server:

   ```bash
   python manage.py runserver
   ```

## API Endpoints

- **User Authentication and Management:**
  - `/register/` - Register a new user.
  - `/login/` - Authenticate user login.
  - `/password-reset/` - Request password reset.
 

- **File Management:**
  - `/files/upload/` - Upload files to Supabase storage and create embeddings.
  - `/files/list/` - List user files with option to filter PDFs.
  - `/user-files/<file_name>/` - Delete user files along with their embeddings and storage.

- **Profile Picture:**
  - `/profile-picture/upload/` - Upload user profile pictures directly to Supabase storage.

- **Chat and Message Management:**
  - `/chats/` and `/messages/` - Endpoints to manage chat sessions and messages.

## Maintenance

- Local temporary files generated during uploads are automatically deleted after processing.
- File deletions remove related embeddings in ChromaDB and storage objects to keep data consistent.

## Contributing

Contributions are welcome! Please submit pull requests or raise issues for improvements and bug fixes.

## License

Specify your project license here.