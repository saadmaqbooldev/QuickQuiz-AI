import os
from typing import Any, Dict, List

import requests
import streamlit as st
from dotenv import load_dotenv


load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")


st.set_page_config(page_title="QuickQuiz AI", page_icon="🧠", layout="wide")


def call_backend(endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Send a JSON request to the FastAPI backend and return the parsed response."""
    response = requests.post(f"{BACKEND_URL}{endpoint}", json=payload, timeout=60)
    response.raise_for_status()
    return response.json()


def reset_flow() -> None:
    """Reset the quiz workflow state for a fresh user session."""
    st.session_state.pop("summary", None)
    st.session_state.pop("quiz", None)
    st.session_state.pop("evaluation", None)
    st.session_state.pop("answers", None)


st.title("QuickQuiz AI")
st.caption("Paste a text, generate a summary, answer a quiz, and check your score.")

if "summary" not in st.session_state:
    st.session_state.summary = None
if "quiz" not in st.session_state:
    st.session_state.quiz = None
if "evaluation" not in st.session_state:
    st.session_state.evaluation = None
if "answers" not in st.session_state:
    st.session_state.answers = {}

with st.sidebar:
    st.header("Controls")
    st.button("Start Fresh", on_click=reset_flow, use_container_width=True)

text_input = st.text_area(
    "Paste your text here",
    height=220,
    placeholder="Add an article, lecture notes, or any long-form content you want to learn from.",
)

if st.button("Generate Summary", use_container_width=True):
    if not text_input.strip():
        st.warning("Please enter some text before requesting a summary.")
    else:
        try:
            with st.spinner("Generating a summary..."):
                result = call_backend("/summarize", {"text": text_input})
            st.session_state.summary = result.get("summary", "")
            st.success("Summary generated successfully.")
        except requests.exceptions.RequestException:
            st.error("Backend is not responding. Please check that the FastAPI server is running.")
        except Exception as exc:  # pragma: no cover - UI safety
            st.error(f"Could not generate the summary: {exc}")

if st.session_state.summary:
    st.subheader("Summary")
    st.write(st.session_state.summary)

if st.button("Generate Quiz", use_container_width=True):
    if not text_input.strip():
        st.warning("Please enter some text before generating a quiz.")
    else:
        try:
            with st.spinner("Creating quiz questions..."):
                result = call_backend("/quiz", {"text": text_input})
            st.session_state.quiz = result.get("questions", [])
            st.session_state.answers = {}
            st.session_state.evaluation = None
            st.success("Quiz generated successfully.")
        except requests.exceptions.RequestException:
            st.error("Backend is not responding. Please check that the FastAPI server is running.")
        except Exception as exc:  # pragma: no cover - UI safety
            st.error(f"Could not generate the quiz: {exc}")

if st.session_state.quiz:
    st.subheader("Quiz")
    st.write("Answer the questions below and submit your responses.")

    for index, question in enumerate(st.session_state.quiz):
        options = question["options"]
        selected = st.radio(
            f"Question {index + 1}: {question['question']}",
            options,
            index=None,
            key=f"answer_{index}",
        )
        st.session_state.answers[index] = options.index(selected) if selected is not None else None

    if st.button("Submit Answers", type="primary", use_container_width=True):
        unanswered = [
            idx for idx, answer in st.session_state.answers.items() if answer is None
        ]

        if unanswered:
            st.warning("Please answer every question before submitting.")
        else:
            payload = {
                "questions": [
                    {
                        "question": question["question"],
                        "options": question["options"],
                        "correct_index": question["correct_index"],
                        "user_answer": st.session_state.answers[idx],
                    }
                    for idx, question in enumerate(st.session_state.quiz)
                ]
            }

            try:
                with st.spinner("Evaluating your answers..."):
                    result = call_backend("/evaluate-quiz", payload)
                st.session_state.evaluation = result
                st.success("Evaluation complete.")
            except requests.exceptions.RequestException:
                st.error("Backend is not responding. Please check that the FastAPI server is running.")
            except Exception as exc:  # pragma: no cover - UI safety
                st.error(f"Could not evaluate the quiz: {exc}")

if st.session_state.evaluation:
    evaluation = st.session_state.evaluation
    st.subheader("Evaluation Results")

    col1, col2, col3 = st.columns(3)
    col1.metric("Score", f"{evaluation['score_percentage']}%")
    col2.metric("Correct", f"{evaluation['correct_count']}/{evaluation['total_questions']}")
    col3.metric("Status", "Passed" if evaluation["passed"] else "Needs Review")

    for item in evaluation["results"]:
        status = "✅ Correct" if item["is_correct"] else "❌ Incorrect"
        st.markdown(f"### {status}")
        st.write(f"Question: {item['question']}")
        st.write(f"Your answer: {item['user_answer']}")
        st.write(f"Correct answer: {item['correct_answer']}")
        st.write(item["explanation"])
        st.markdown("---")
