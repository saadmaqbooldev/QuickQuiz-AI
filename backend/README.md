# QuickQuiz AI - Backend

FastAPI backend for QuickQuiz AI, integrating Gemini LLM for text summarization and quiz generation.

## Quick Start

1. Copy `.env.example` to `.env` and add your Gemini API key
2. Create virtual environment: `python -m venv venv`
3. Activate: `source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows)
4. Install: `pip install -r requirements.txt`
5. Run: `uvicorn main:app --reload --host 0.0.0.0 --port 8000`

## API Documentation

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/summarize` | Generate text summary |
| POST | `/quiz` | Generate quiz questions |