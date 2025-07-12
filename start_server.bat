@echo off
echo Starting Multimodal RAG API server...

REM Check if Python is installed
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python and try again
    pause
    exit /b 1
)

REM Check if virtual environment exists
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
    if %ERRORLEVEL% NEQ 0 (
        echo Error creating virtual environment
        pause
        exit /b 1
    )
)

REM Activate virtual environment and install dependencies
echo Activating virtual environment and installing dependencies...
call venv\Scripts\activate.bat
pip install -r requirements.txt
if %ERRORLEVEL% NEQ 0 (
    echo Error installing dependencies
    pause
    exit /b 1
)

REM Check if OPENAI_API_KEY is set
if "%OPENAI_API_KEY%"=="" (
    echo WARNING: OPENAI_API_KEY environment variable is not set
    echo You will need to set your API key in app.py before using the API
    timeout /t 5
)

REM Start the server
echo Starting server...
python app.py

REM If server exits, deactivate venv
call venv\Scripts\deactivate.bat

pause