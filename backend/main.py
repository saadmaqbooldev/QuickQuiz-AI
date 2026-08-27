# backend/main.py

"""
QuickQuiz AI - FastAPI Backend
==============================

A production-ready FastAPI backend that integrates with Google's Gemini LLM
to provide text summarization and quiz generation services.

Endpoints:
    - GET  /health     : Health check
    - POST /summarize  : Generate a summary from input text
    - POST /quiz       : Generate quiz questions from input text

Author: QuickQuiz AI Team
Version: 1.0.0
"""

import json
import logging
import os
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# Import Google GenAI SDK
try:
    from google import genai
    from google.genai import types
except ImportError:
    raise ImportError(
        "google-genai package is not installed. "
        "Run: pip install -r requirements.txt"
    )

# ---------------------------------------------------------------------------
# Environment & Configuration
# ---------------------------------------------------------------------------

# Load environment variables from .env file
load_dotenv()

# Configure logging for production visibility
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Constants
GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")
# The live Gemini API now recommends gemini-3.5-flash-lite for new users.
# Use the currently supported model name to avoid 404 and temporary 503 issues.
GEMINI_MODEL: str = "gemini-3.5-flash-lite"
MAX_TEXT_LENGTH: int = 8000
QUIZ_QUESTION_COUNT: int = 3
QUIZ_OPTIONS_COUNT: int = 4

# Validate API key at startup
if not GEMINI_API_KEY:
    logger.warning(
        "GEMINI_API_KEY not found in environment variables. "
        "API endpoints will fail until configured."
    )

# Initialize Gemini client (lazy initialization to avoid startup crash without key)
_gemini_client: Optional[genai.Client] = None


def get_gemini_client() -> genai.Client:
    """
    Get or create the Gemini client instance.

    Returns:
        genai.Client: Configured Gemini client

    Raises:
        HTTPException: 500 if API key is not configured
    """
    global _gemini_client
    if not GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY is not configured")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server configuration error: API key not set. "
                   "Please configure GEMINI_API_KEY in .env file.",
        )
    
    if _gemini_client is None:
        _gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        logger.info("Gemini client initialized successfully")
    
    return _gemini_client


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class TextRequest(BaseModel):
    """
    Request model for text input endpoints.

    Attributes:
        text (str): The input text to process (1-8000 characters)
    """
    text: str = Field(
        ...,
        description="The text to process",
        min_length=1,
        max_length=MAX_TEXT_LENGTH,
    )

    @field_validator("text")
    @classmethod
    def validate_text_not_empty(cls, v: str) -> str:
        """
        Validate that text is not empty or whitespace-only.

        Args:
            v (str): The input text

        Returns:
            str: The validated text

        Raises:
            ValueError: If text is empty or whitespace-only
        """
        if not v or not v.strip():
            raise ValueError("Text cannot be empty or whitespace-only")
        return v.strip()

    @field_validator("text")
    @classmethod
    def validate_text_length(cls, v: str) -> str:
        """
        Validate that text does not exceed maximum length.

        Args:
            v (str): The input text

        Returns:
            str: The validated text

        Raises:
            ValueError: If text exceeds MAX_TEXT_LENGTH characters
        """
        if len(v) > MAX_TEXT_LENGTH:
            raise ValueError(
                f"Text exceeds maximum length of {MAX_TEXT_LENGTH} characters"
            )
        return v

    class Config:
        """Pydantic model configuration."""
        json_schema_extra = {
            "example": {
                "text": "Artificial Intelligence is transforming the way we "
                        "interact with technology. Machine learning algorithms "
                        "can now understand natural language, recognize images, "
                        "and make predictions with remarkable accuracy."
            }
        }


class SummaryResponse(BaseModel):
    """
    Response model for text summarization.

    Attributes:
        summary (str): The generated summary
    """
    summary: str = Field(..., description="The generated text summary")


class QuizQuestion(BaseModel):
    """
    Model representing a single quiz question.

    Attributes:
        question (str): The question text
        options (List[str]): Exactly 4 answer options
        correct_index (int): Index of the correct option (0-3)
    """
    question: str = Field(..., description="The quiz question text")
    options: List[str] = Field(
        ...,
        description="Four answer options",
        min_length=QUIZ_OPTIONS_COUNT,
        max_length=QUIZ_OPTIONS_COUNT,
    )
    correct_index: int = Field(
        ...,
        description="Index of the correct answer (0-3)",
        ge=0,
        le=QUIZ_OPTIONS_COUNT - 1,
    )

    @field_validator("options")
    @classmethod
    def validate_options_count(cls, v: List[str]) -> List[str]:
        """
        Ensure exactly 4 options are provided and none are empty.

        Args:
            v (List[str]): List of options

        Returns:
            List[str]: Validated options

        Raises:
            ValueError: If options count is not 4 or any option is empty
        """
        if len(v) != QUIZ_OPTIONS_COUNT:
            raise ValueError(
                f"Exactly {QUIZ_OPTIONS_COUNT} options are required"
            )
        
        if any(not opt.strip() for opt in v):
            raise ValueError("Options cannot be empty")
        
        return [opt.strip() for opt in v]


class QuizResponse(BaseModel):
    """
    Response model for quiz generation.

    Attributes:
        questions (List[QuizQuestion]): Exactly 3 quiz questions
    """
    questions: List[QuizQuestion] = Field(
        ...,
        description="List of quiz questions",
        min_length=QUIZ_QUESTION_COUNT,
        max_length=QUIZ_QUESTION_COUNT,
    )

    @field_validator("questions")
    @classmethod
    def validate_questions_count(cls, v: List[QuizQuestion]) -> List[QuizQuestion]:
        """
        Ensure exactly 3 questions are provided.

        Args:
            v (List[QuizQuestion]): List of questions

        Returns:
            List[QuizQuestion]: Validated questions

        Raises:
            ValueError: If questions count is not 3
        """
        if len(v) != QUIZ_QUESTION_COUNT:
            raise ValueError(
                f"Exactly {QUIZ_QUESTION_COUNT} questions are required"
            )
        return v


class QuizSubmission(BaseModel):
    """
    Model representing a single quiz submission for evaluation.

    Attributes:
        question (str): The question text
        options (List[str]): Exactly 4 answer options
        correct_index (int): Index of the correct option (0-3)
        user_answer (int): Index selected by the user (0-3)

    Example:
        {
            "question": "What is the capital of France?",
            "options": ["Berlin", "Paris", "Rome", "Madrid"],
            "correct_index": 1,
            "user_answer": 1
        }
    """
    question: str = Field(..., description="The quiz question text")
    options: List[str] = Field(
        ...,
        description="Four answer options",
        min_length=QUIZ_OPTIONS_COUNT,
        max_length=QUIZ_OPTIONS_COUNT,
    )
    correct_index: int = Field(
        ...,
        description="Index of the correct answer (0-3)",
        ge=0,
        le=QUIZ_OPTIONS_COUNT - 1,
    )
    user_answer: int = Field(
        ...,
        description="Index of the user's selected answer (0-3)",
        ge=0,
        le=QUIZ_OPTIONS_COUNT - 1,
    )

    @field_validator("question")
    @classmethod
    def validate_question_not_empty(cls, v: str) -> str:
        """Ensure the question text is not empty or whitespace-only."""
        if not v or not v.strip():
            raise ValueError("Question cannot be empty or whitespace-only")
        return v.strip()

    @field_validator("options")
    @classmethod
    def validate_options_count(cls, v: List[str]) -> List[str]:
        """Ensure exactly 4 options are provided and none are empty."""
        if len(v) != QUIZ_OPTIONS_COUNT:
            raise ValueError(
                f"Exactly {QUIZ_OPTIONS_COUNT} options are required"
            )

        if any(not opt.strip() for opt in v):
            raise ValueError("Options cannot be empty")

        return [opt.strip() for opt in v]


class QuizSubmissionRequest(BaseModel):
    """
    Request model for evaluating user answers.

    Attributes:
        questions (List[QuizSubmission]): Quiz submissions to evaluate

    Example:
        {
            "questions": [
                {
                    "question": "What is the capital of France?",
                    "options": ["Berlin", "Paris", "Rome", "Madrid"],
                    "correct_index": 1,
                    "user_answer": 1
                }
            ]
        }
    """
    questions: List[QuizSubmission] = Field(
        ...,
        description="Questions submitted for evaluation",
        min_length=1,
        max_length=10,
    )

    @field_validator("questions")
    @classmethod
    def validate_questions_count(cls, v: List[QuizSubmission]) -> List[QuizSubmission]:
        """Ensure between 1 and 10 questions are provided."""
        if len(v) < 1 or len(v) > 10:
            raise ValueError("The number of questions must be between 1 and 10")
        return v


class QuizResult(BaseModel):
    """
    Model representing the evaluation result for a single question.

    Attributes:
        question (str): The question text
        user_answer (int): The answer selected by the user
        correct_answer (int): The correct answer index
        is_correct (bool): Whether the answer was correct
        explanation (str): Brief explanation of the result
    """
    question: str = Field(..., description="The question text")
    user_answer: int = Field(
        ...,
        description="The user's selected answer index",
        ge=0,
        le=QUIZ_OPTIONS_COUNT - 1,
    )
    correct_answer: int = Field(
        ...,
        description="The correct answer index",
        ge=0,
        le=QUIZ_OPTIONS_COUNT - 1,
    )
    is_correct: bool = Field(..., description="Whether the user answered correctly")
    explanation: str = Field(
        ...,
        description="Brief explanation of why the answer is correct or incorrect",
    )


class QuizEvaluationResponse(BaseModel):
    """
    Response model for quiz evaluation.

    Attributes:
        results (List[QuizResult]): Evaluation results for each question
        total_questions (int): Total number of questions evaluated
        correct_count (int): Number of correct answers
        score_percentage (float): Percentage score from 0 to 100
        passed (bool): True if score >= 60%
    """
    results: List[QuizResult] = Field(
        ...,
        description="Evaluation results for each submitted question",
    )
    total_questions: int = Field(
        ...,
        description="Total number of questions evaluated",
        ge=1,
    )
    correct_count: int = Field(
        ...,
        description="Number of questions answered correctly",
        ge=0,
    )
    score_percentage: float = Field(
        ...,
        description="Final percentage score",
        ge=0,
        le=100,
    )
    passed: bool = Field(
        ...,
        description="Whether the user passed the quiz (>= 60%)",
    )


# ---------------------------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="QuickQuiz AI",
    description="AI-powered text summarization and quiz generation using Gemini LLM",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware configuration
# Allow all origins for development; restrict in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Restrict to specific frontend origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def _parse_gemini_json_response(response_text: str) -> dict:
    """
    Parse JSON from Gemini's text response, handling markdown code blocks.

    Args:
        response_text (str): Raw response text from Gemini

    Returns:
        dict: Parsed JSON object

    Raises:
        ValueError: If JSON parsing fails
    """
    # Clean the response text
    cleaned_text = response_text.strip()
    
    # Remove markdown code block markers if present
    if cleaned_text.startswith("```json"):
        cleaned_text = cleaned_text[7:]  # Remove ```json
    elif cleaned_text.startswith("```"):
        cleaned_text = cleaned_text[3:]  # Remove ```
    
    if cleaned_text.endswith("```"):
        cleaned_text = cleaned_text[:-3]  # Remove trailing ```
    
    cleaned_text = cleaned_text.strip()
    
    try:
        return json.loads(cleaned_text)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON from Gemini response: {e}")
        logger.debug(f"Raw response: {response_text[:500]}...")  # Log first 500 chars
        raise ValueError(f"Invalid JSON response from LLM: {e}")


def _extract_response_text(response) -> str:
    """
    Extract text content from Gemini API response.

    Args:
        response: Gemini API response object

    Returns:
        str: Extracted text content

    Raises:
        ValueError: If no text content is found
    """
    try:
        # Extract text from the response
        if response.candidates and response.candidates[0].content.parts:
            text_parts = [
                part.text for part in response.candidates[0].content.parts
                if hasattr(part, 'text') and part.text
            ]
            if text_parts:
                return " ".join(text_parts)
        
        raise ValueError("No text content found in Gemini response")
    except (AttributeError, IndexError) as e:
        logger.error(f"Failed to extract text from response: {e}")
        raise ValueError("Unexpected response format from Gemini")


def _generate_quiz_explanation(
    question: str,
    options: List[str],
    correct_index: int,
    user_answer: int,
    is_correct: bool,
) -> str:
    """
    Generate a brief explanation for a quiz answer.

    If Gemini is available, it can optionally enrich the explanation. If the
    API call fails, a generic explanation is returned instead.

    Args:
        question (str): The quiz question text
        options (List[str]): Answer choices
        correct_index (int): Correct answer index
        user_answer (int): User-selected answer index
        is_correct (bool): Whether the answer was correct

    Returns:
        str: Explanation text
    """
    correct_option = options[correct_index]

    if is_correct:
        return (
            f"Correct! '{correct_option}' is the right answer for this question."
        )

    generic_explanation = (
        f"Incorrect. The correct answer is '{correct_option}'. "
        f"You selected '{options[user_answer]}'."
    )

    try:
        if not GEMINI_API_KEY:
            return generic_explanation

        client = get_gemini_client()
        prompt = (
            f"Explain in one sentence why '{correct_option}' is the correct answer "
            f"for this question and why '{options[user_answer]}' is incorrect. "
            f"Question: {question}"
        )
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=200,
            ),
        )
        explanation = _extract_response_text(response)
        if explanation.strip():
            return explanation.strip()
    except Exception as e:
        logger.warning(
            f"Gemini explanation generation failed; using fallback message. Error: {e}"
        )

    return generic_explanation


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.get(
    "/health",
    summary="Health Check",
    description="Check if the API server is running",
    response_model=dict,
    tags=["Health"],
)
async def health_check() -> dict:
    """
    Health check endpoint to verify server is running.

    Returns:
        dict: {"status": "ok"} if server is healthy
    """
    return {"status": "ok"}


@app.post(
    "/summarize",
    summary="Generate Text Summary",
    description="Generate a concise summary of the input text using Gemini LLM",
    response_model=SummaryResponse,
    status_code=status.HTTP_200_OK,
    responses={
        422: {"description": "Validation error - empty or oversized text"},
        500: {"description": "Internal server error or LLM API failure"},
        502: {"description": "Failed to process LLM response"},
    },
    tags=["AI"],
)
async def summarize_text(request: TextRequest) -> SummaryResponse:
    """
    Generate a summary from the input text.

    Args:
        request (TextRequest): Contains the text to summarize

    Returns:
        SummaryResponse: The generated summary

    Raises:
        HTTPException: 500 for API errors, 502 for parse failures
    """
    logger.info(f"Summarization request received. Text length: {len(request.text)}")
    
    try:
        client = get_gemini_client()
        
        # Build prompt for summarization
        prompt = f"""
        Please provide a concise and accurate summary of the following text.
        Focus on the key points and main ideas while maintaining clarity.
        
        Text to summarize:
        {request.text}
        
        Summary:
        """
        
        # Call Gemini API
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.3,  # Lower temperature for more focused summaries
                max_output_tokens=1024,
            ),
        )
        
        # Extract summary text
        summary = _extract_response_text(response)
        
        if not summary.strip():
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="LLM returned an empty summary. Please try again.",
            )
        
        logger.info(f"Summarization successful. Summary length: {len(summary)}")
        return SummaryResponse(summary=summary.strip())
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except ValueError as e:
        logger.error(f"Value error during summarization: {e}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to process LLM response: {str(e)}",
        )
    except Exception as e:
        logger.error(f"Unexpected error during summarization: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate summary. Please try again later.",
        )


@app.post(
    "/quiz",
    summary="Generate Quiz Questions",
    description="Generate quiz questions from the input text using Gemini LLM",
    response_model=QuizResponse,
    status_code=status.HTTP_200_OK,
    responses={
        422: {"description": "Validation error - empty or oversized text"},
        500: {"description": "Internal server error or LLM API failure"},
        502: {"description": "Failed to parse LLM JSON response"},
    },
    tags=["AI"],
)
async def generate_quiz(request: TextRequest) -> QuizResponse:
    """
    Generate quiz questions from the input text.

    Args:
        request (TextRequest): Contains the text to generate quiz from

    Returns:
        QuizResponse: The generated quiz questions

    Raises:
        HTTPException: 500 for API errors, 502 for JSON parse failures
    """
    logger.info(f"Quiz generation request received. Text length: {len(request.text)}")
    
    try:
        client = get_gemini_client()
        
        # Build strict prompt for JSON quiz generation
        prompt = f"""
        Based on the following text, create {QUIZ_QUESTION_COUNT} multiple-choice quiz questions.
        
        IMPORTANT: You must respond ONLY with valid JSON in the exact format below.
        Do not include any additional text, explanations, or markdown formatting.
        
        Required JSON format:
        {{
            "questions": [
                {{
                    "question": "Question text here",
                    "options": ["Option A", "Option B", "Option C", "Option D"],
                    "correct_index": 0
                }},
                {{
                    "question": "Question text here",
                    "options": ["Option A", "Option B", "Option C", "Option D"],
                    "correct_index": 1
                }},
                {{
                    "question": "Question text here",
                    "options": ["Option A", "Option B", "Option C", "Option D"],
                    "correct_index": 2
                }}
            ]
        }}
        
        Rules:
        1. Create exactly {QUIZ_QUESTION_COUNT} questions
        2. Each question must have exactly {QUIZ_OPTIONS_COUNT} options
        3. correct_index must be between 0 and {QUIZ_OPTIONS_COUNT - 1}
        4. Questions should test comprehension of key concepts
        5. Only one option should be correct
        6. Make questions clear and unambiguous
        
        Text to analyze:
        {request.text}
        """
        
        # Call Gemini API with strict JSON mode if available
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.7,  # Balanced temperature for creative but accurate questions
                    max_output_tokens=2048,
                    response_mime_type="application/json",  # Request JSON response
                ),
            )
        except Exception as api_error:
            logger.warning(f"Strict JSON mode failed, retrying without: {api_error}")
            # Fallback: try without response_mime_type
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.7,
                    max_output_tokens=2048,
                ),
            )
        
        # Extract and parse JSON response
        response_text = _extract_response_text(response)
        logger.debug(f"Gemini response: {response_text[:500]}...")
        
        try:
            quiz_data = _parse_gemini_json_response(response_text)
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Failed to parse quiz data from LLM response. "
                       "The model returned invalid JSON. Please try again.",
            )
        
        # Validate parsed data using Pydantic model
        try:
            quiz_response = QuizResponse(**quiz_data)
        except Exception as validation_error:
            logger.error(f"Quiz data validation failed: {validation_error}")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="LLM returned quiz data that does not match the expected format. "
                       "Please try again.",
            )
        
        logger.info(f"Quiz generation successful. Questions: {len(quiz_response.questions)}")
        return quiz_response
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except ValueError as e:
        logger.error(f"Value error during quiz generation: {e}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to process LLM response: {str(e)}",
        )
    except Exception as e:
        logger.error(f"Unexpected error during quiz generation: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate quiz. Please try again later.",
        )


@app.post(
    "/evaluate-quiz",
    summary="Evaluate Quiz Answers",
    description="Evaluate user-submitted quiz answers against the correct answers and return a score",
    response_model=QuizEvaluationResponse,
    status_code=status.HTTP_200_OK,
    responses={
        422: {"description": "Validation error - invalid quiz submission"},
        500: {"description": "Internal server error during evaluation"},
    },
    tags=["AI"],
)
async def evaluate_quiz(request: QuizSubmissionRequest) -> QuizEvaluationResponse:
    """
    Evaluate submitted quiz answers and calculate the final score.

    This endpoint performs pure logic evaluation without calling the Gemini API.
    Explanations are generated for each question, and if Gemini is unavailable,
    a generic explanation is returned instead.

    Args:
        request (QuizSubmissionRequest): Quiz questions and selected answers

    Returns:
        QuizEvaluationResponse: Evaluation results, score, and pass/fail status

    Raises:
        HTTPException: 500 for unexpected evaluation errors
    """
    logger.info(
        f"Quiz evaluation request received. Total questions: {len(request.questions)}"
    )

    try:
        results: List[QuizResult] = []
        correct_count = 0

        for submission in request.questions:
            is_correct = submission.user_answer == submission.correct_index
            explanation = _generate_quiz_explanation(
                question=submission.question,
                options=submission.options,
                correct_index=submission.correct_index,
                user_answer=submission.user_answer,
                is_correct=is_correct,
            )

            if is_correct:
                correct_count += 1

            results.append(
                QuizResult(
                    question=submission.question,
                    user_answer=submission.user_answer,
                    correct_answer=submission.correct_index,
                    is_correct=is_correct,
                    explanation=explanation,
                )
            )

        total_questions = len(results)
        score_percentage = round((correct_count / total_questions) * 100, 2) if total_questions else 0.0
        passed = score_percentage >= 60

        response = QuizEvaluationResponse(
            results=results,
            total_questions=total_questions,
            correct_count=correct_count,
            score_percentage=score_percentage,
            passed=passed,
        )

        logger.info(
            "Quiz evaluation complete. "
            f"Correct answers: {correct_count}/{total_questions}, "
            f"Score: {score_percentage}%, Passed: {passed}"
        )
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during quiz evaluation: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to evaluate quiz. Please try again later.",
        )


# ---------------------------------------------------------------------------
# Application Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    
    # Run the FastAPI application with Uvicorn
    # Host 0.0.0.0 makes it accessible from other devices on the network
    # Port 8000 is the default FastAPI port
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Enable auto-reload for development
        log_level="info",
    )