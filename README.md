# Multimodal-RAG

This project is a Django-based web application integrated with Supabase for user authentication, file storage, and database management to provide a powerful retrieval-augmented generation pipeline for multi-modal data.

## Features

- User registration, login, password reset, and OTP verification via Supabase Auth.
- File upload and storage using Supabase Storage.
- Chat and message management stored in Supabase database.
- Integration with OpenAI API for generative AI capabilities.
- Support for multi-modal retrieval-augmented generation (RAG) pipelines.

## Setup Instructions

1. Clone the repository.
2. Create and activate a Python virtual environment.
   
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file with the following environment variables:

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

   Replace values with your actual credentials.

5. Apply migrations and run the server:

   ```bash
   python manage.py migrate
   python manage.py runserver
   ```

## API Endpoints

- `/register/` - User registration
- `/login/` - User login
- `/password-reset/` - Password reset
- `/verify-otp/` - OTP verification
- `/files/upload/` - Upload files
- `/files/list/` - List user files
- `/chats/` and `/messages/` - Manage chats and messages

## Notes

- While Django's PostgreSQL database settings are still present, Supabase handles the primary data storage and authentication.
- For file storage and user management, Supabase services are fully integrated.
- Ensure environment variables are set properly for smooth operation.

## License

MIT License