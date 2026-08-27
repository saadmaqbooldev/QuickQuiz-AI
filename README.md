# QuickQuiz AI 🚀

An AI-powered application that generates text summaries and interactive quizzes using Google's Gemini LLM. Built with FastAPI backend and designed for seamless integration with Streamlit frontend.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)
![Gemini](https://img.shields.io/badge/Gemini-2.0-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Application](#running-the-application)
- [API Documentation](#api-documentation)
- [API Endpoints](#api-endpoints)
- [Testing](#testing)
- [Error Handling](#error-handling)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## 🎯 Overview

QuickQuiz AI is a hackathon project that leverages Google's Gemini LLM to:
- Generate concise summaries from input text
- Create interactive multiple-choice quizzes
- Evaluate user responses and provide scores
- Deliver results through RESTful API endpoints

## ✨ Features

### Backend Features
- **Text Summarization**: Generate concise summaries using Gemini AI
- **Quiz Generation**: Create 3 multiple-choice questions with 4 options each
- **Quiz Evaluation**: Evaluate user answers and calculate scores
- **Health Check**: Monitor API server status
- **Input Validation**: Robust validation using Pydantic models
- **Error Handling**: Graceful error responses with appropriate status codes
- **CORS Support**: Cross-origin resource sharing for frontend integration
- **Environment Management**: Secure API key handling using .env files
- **Interactive Docs**: Swagger UI and ReDoc for API exploration

### AI Capabilities
- **Smart Summarization**: Key points extraction with temperature control
- **Context-Aware Quizzes**: Questions based on text comprehension
- **JSON Response Mode**: Structured output for reliable parsing
- **Fallback Mechanisms**: Automatic retry without strict JSON mode

## 🛠️ Tech Stack

| Technology | Version | Purpose |
|------------|---------|---------|
| Python | 3.9+ | Core programming language |
| FastAPI | 0.115+ | Web framework |
| Uvicorn | 0.34+ | ASGI server |
| Google GenAI | 1.0+ | Gemini LLM integration |
| Pydantic | 2.10+ | Data validation |
| python-dotenv | 1.0+ | Environment management |

## 📁 Project Structure
AI-Mini_Hackathon/
│
├── .env.example # Template for environment variables
├── .gitignore # Git ignore rules
├── README.md # Project documentation
├── AI_Hackathon_Lab_Guide.md # Lab guide documentation
├── Week9_lab4.docx # Lab documentation
│
└── backend/
├── main.py # FastAPI application (all endpoints)
├── requirements.txt # Python dependencies
└── README.md # Backend-specific documentation


## 📦 Prerequisites

Before running this project, ensure you have:

- **Python 3.9 or higher** installed
  ```bash
  python --version

  Google Gemini API Key

Get it from Google AI Studio

Git (for version control)

bash
git --version

🚀 Installation
Step 1: Clone the Repository
bash
git clone https://github.com/your-username/AI-Mini_Hackathon.git
cd AI-Mini_Hackathon
Step 2: Create Virtual Environment
Windows:

powershell
python -m venv venv
venv\Scripts\Activate.ps1
Mac/Linux:

bash
python -m venv venv
source venv/bin/activate
Step 3: Install Dependencies
bash
pip install -r backend/requirements.txt
Step 4: Configure Environment
bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your API key
# GEMINI_API_KEY=your_actual_api_key_here
⚙️ Configuration
Environment Variables
Variable	Description	Required	Default
GEMINI_API_KEY	Google Gemini API key	Yes	None
GEMINI_MODEL	Gemini model to use	No	gemini-2.0-flash
Available Gemini Models
Model	Description	Best For
gemini-2.5-flash	Fast, efficient	Quick responses
gemini-2.5-pro	Most capable	Complex tasks
gemini-2.0-flash	Previous gen, stable	General use
gemini-2.0-flash-lite	Fastest, cheapest	Simple tasks
🏃 Running the Application
Start the Server
bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
Alternative:

bash
cd backend
python main.py
