"""
AI Based MCQ Generator
----------------------
Pipeline:
  Upload notes (PDF/TXT) -> extract text -> LangChain PromptTemplate
  -> Gemini (ChatGoogleGenerativeAI) -> Pydantic JSON output parser
  -> interactive quiz -> score.
"""

import os
from typing import List

import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from pypdf.errors import PyPdfError

from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field, field_validator, model_validator

# --------------------------------------------------------------------------
# 1. Configuration
# --------------------------------------------------------------------------
# Load variables from a local .env file (GOOGLE_API_KEY, GEMINI_MODEL).
load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY", "").strip()

# NOTE: Gemini 1.5 Flash has been retired by Google and its API calls now fail.
# "gemini-2.5-flash" is its free-tier Flash successor. Change it via .env.
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite").strip()
st.write("Model:", MODEL_NAME)
# Limit the characters sent to the model to keep requests fast and in free quota.
MAX_INPUT_CHARS = 30_000

# --------------------------------------------------------------------------
# 2. Output schema (what the LLM must return as JSON)
# --------------------------------------------------------------------------
class MCQ(BaseModel):
    """A single multiple-choice question."""

    question: str = Field(description="The question text")
    options: List[str] = Field(description="Exactly four distinct answer options")
    correct_answer: str = Field(
        description="The correct answer. Must be copied exactly from one of the options"
    )

    @field_validator("options")
    @classmethod
    def check_four_options(cls, v: List[str]) -> List[str]:
        # Guard against malformed model output.
        if len(v) != 4 or len(set(v)) != 4:
            raise ValueError("Each question needs exactly four distinct options")
        return v

    @model_validator(mode="after")
    def check_answer_in_options(self) -> "MCQ":
        # The correct answer must be one of the options so scoring works.
        if self.correct_answer not in self.options:
            raise ValueError("correct_answer must match one of the options")
        return self


class MCQSet(BaseModel):
    """The full quiz returned by the model."""

    questions: List[MCQ]


# --------------------------------------------------------------------------
# 3. LangChain chain: PromptTemplate | Gemini | OutputParser
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def build_chain(api_key: str, model_name: str):
    """Build the LangChain pipeline once and reuse it across Streamlit reruns."""

    # Parser converts the model's JSON text into validated Python objects.
    parser = PydanticOutputParser(pydantic_object=MCQSet)

    # Prompt template; {format_instructions} is auto-filled with the JSON schema.
    prompt = PromptTemplate(
        template=(
            "You are an expert teacher creating a high-quality quiz.\n"
            "Using ONLY the study notes below, write exactly {num_questions} "
            "multiple-choice questions.\n\n"
            "Rules:\n"
            "- Test understanding of key concepts, not trivial wording.\n"
            "- Each question has exactly 4 plausible, distinct options.\n"
            "- Exactly one option is correct, and 'correct_answer' must be copied "
            "word-for-word from the options.\n"
            "- Vary which position holds the correct answer.\n"
            "- Do not use information outside the notes.\n"
            "- Return ONLY valid JSON, with no extra commentary.\n\n"
            "{format_instructions}\n\n"
            "STUDY NOTES:\n{notes}\n"
        ),
        input_variables=["notes", "num_questions"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    # Gemini chat model via LangChain. The API key is passed explicitly.
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        api_key=api_key,
        temperature=0.4,
        max_retries=2,
        # Ask Gemini to emit raw JSON (no markdown fences).
        response_mime_type="application/json",
    )

    # LCEL pipe syntax: prompt -> LLM -> parser.
    return prompt | llm | parser


def generate_mcqs(notes: str, num_questions: int) -> List[MCQ]:
    """Send notes to Gemini through LangChain and return validated MCQs."""
    chain = build_chain(API_KEY, MODEL_NAME)
    result: MCQSet = chain.invoke(
        {"notes": notes[:MAX_INPUT_CHARS], "num_questions": num_questions}
    )
    # Trim in case the model returned more than requested.
    return result.questions[:num_questions]


# --------------------------------------------------------------------------
# 4. File processing
# --------------------------------------------------------------------------
def extract_text(uploaded_file) -> str:
    """Return plain text from an uploaded PDF or TXT file."""
    if uploaded_file.name.lower().endswith(".pdf"):
        reader = PdfReader(uploaded_file)
        if reader.is_encrypted:
            raise ValueError("This PDF is password protected.")
        # Join page texts; skip pages with no extractable text (e.g. scans).
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages).strip()

    raw = uploaded_file.read()
    try:
        return raw.decode("utf-8").strip()
    except UnicodeDecodeError:
        # Fallback for files saved in a legacy encoding.
        return raw.decode("latin-1").strip()


# --------------------------------------------------------------------------
# 5. Streamlit UI
# --------------------------------------------------------------------------
st.set_page_config(page_title="AI Based MCQ Generator")
st.title("AI Based MCQ Generator")

# Stop early with a clear message if the API key is missing.
if not API_KEY:
    st.error(
        "GOOGLE_API_KEY not found. Copy `.env.example` to `.env`, add your "
        "Gemini API key, and restart the app."
    )
    st.stop()

# Session state keeps the quiz alive across Streamlit reruns.
st.session_state.setdefault("mcqs", None)
st.session_state.setdefault("quiz_id", 0)

uploaded_file = st.file_uploader("Upload Notes", type=["txt", "pdf"])
num_questions = st.slider("Number of Questions", 1, 10, 5)

if uploaded_file:
    # ---- Extract and preview text ----
    try:
        text = extract_text(uploaded_file)
    except (PyPdfError, ValueError) as e:
        st.error(f"Could not read the file: {e}")
        st.stop()
    except Exception as e:  # unexpected file problems
        st.error(f"Unexpected error while reading the file: {e}")
        st.stop()

    if not text:
        st.warning("No readable text found. Scanned PDFs need OCR first.")
        st.stop()

    st.subheader("Uploaded Notes")
    st.write(text[:1000])
    if len(text) > MAX_INPUT_CHARS:
        st.caption(f"Only the first {MAX_INPUT_CHARS:,} characters will be used.")

    # ---- Generate MCQs ----
    if st.button("Generate MCQs"):
        with st.spinner("Generating questions with Gemini..."):
            try:
                st.session_state.mcqs = generate_mcqs(text, num_questions)
                st.session_state.quiz_id += 1  # new widget keys for the new quiz
            except OutputParserException:
                st.session_state.mcqs = None
                st.error("The model returned invalid JSON. Please try again.")
            except Exception as e:
                st.session_state.mcqs = None
                msg = str(e)
                if "429" in msg or "quota" in msg.lower():
                    st.error("Rate limit or free quota reached. Wait a minute and retry.")
                elif "API key" in msg or "401" in msg or "403" in msg:
                    st.error("Gemini rejected the API key. Check your .env file.")
                elif "404" in msg:
                    st.error(f"Model '{MODEL_NAME}' not found. Set GEMINI_MODEL in .env.")
                else:
                    st.error(f"Failed to generate MCQs: {msg}")

# ---- Interactive quiz ----
mcqs = st.session_state.mcqs
if mcqs:
    st.subheader("Generated MCQs")

    # A form means the page only reruns when "Submit Quiz" is pressed.
    with st.form("quiz_form"):
        answers = []
        for i, q in enumerate(mcqs, start=1):
            st.write(f"### Q{i}")
            st.write(q.question)
            answers.append(
                st.radio(
                    "Choose Answer",
                    q.options,
                    index=None,  # nothing pre-selected
                    key=f"q_{st.session_state.quiz_id}_{i}",
                )
            )
        submitted = st.form_submit_button("Submit Quiz")

    # ---- Score and review ----
    if submitted:
        score = sum(
            1 for q, a in zip(mcqs, answers) if a == q.correct_answer
        )
        st.success(f"Your Score: {score}/{len(mcqs)}")

        with st.expander("Review answers", expanded=True):
            for i, (q, a) in enumerate(zip(mcqs, answers), start=1):
                if a == q.correct_answer:
                    st.write(f"✅ **Q{i}** — Correct: {q.correct_answer}")
                else:
                    st.write(
                        f"❌ **Q{i}** — Your answer: {a or 'Not answered'} | "
                        f"Correct: {q.correct_answer}"
                    )
