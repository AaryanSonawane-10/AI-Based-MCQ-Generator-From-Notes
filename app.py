import os
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from pypdf.errors import PdfReadError
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from rag_engine import corrective_retrieve, build_note_chunks
from fuzzy_logic import assess_mcq
from schemas import MCQSet

load_dotenv()

st.set_page_config(page_title="AI MCQ Generator — Corrective RAG", page_icon="🧠", layout="wide")

def get_secret(name: str, default: str = "") -> str:
    """Read Streamlit Cloud secrets first, then local environment variables."""
    try:
        value = st.secrets.get(name, None)
    except Exception:
        value = None
    return str(value if value is not None else os.getenv(name, default)).strip()

API_KEY = get_secret("GOOGLE_API_KEY")
MODEL_NAME = get_secret("GEMINI_MODEL", "gemini-2.5-flash")
MAX_INPUT_CHARS = 50_000

st.title("🧠 AI-Based MCQ Generator from Notes")
st.caption("Corrective RAG • Gemini • Fuzzy difficulty and quality assessment")

with st.sidebar:
    st.header("Settings")
    num_questions = st.slider("Number of questions", 1, 10, 5)
    allow_web = st.toggle("Allow web search fallback", value=True,
                          help="Used only when the uploaded notes do not provide enough relevant evidence.")
    max_web_results = st.slider("Maximum web results", 1, 5, 3, disabled=not allow_web)
    st.info("API keys are read from Streamlit Secrets or your local .env file.")

uploaded_file = st.file_uploader("Upload your notes", type=["pdf", "txt"])

def extract_text(file) -> str:
    suffix = file.name.lower().rsplit(".", 1)[-1]
    if suffix == "txt":
        raw = file.getvalue()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("latin-1")
        return text.strip()
    if suffix == "pdf":
        reader = PdfReader(file)
        pages = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")
        return "\n\n".join(pages).strip()
    raise ValueError("Please upload a PDF or TXT file.")

notes = ""
if uploaded_file:
    try:
        notes = extract_text(uploaded_file)
        if not notes:
            st.warning("No readable text was found. If this is a scanned PDF, run OCR on it first and upload a text-based PDF.")
        else:
            st.success(f"Extracted {len(notes):,} characters from {uploaded_file.name}.")
            with st.expander("Preview extracted notes"):
                st.text_area("Notes preview", notes[:8000], height=220, disabled=True)
    except (ValueError, PdfReadError, Exception) as exc:
        st.error(f"Could not read this file: {exc}")

if st.button("Generate MCQs", type="primary", disabled=not notes):
    if not API_KEY:
        st.error("Gemini API key is missing. Add GOOGLE_API_KEY to your local .env file or Streamlit Cloud Secrets.")
    elif len(notes.strip()) < 80:
        st.error("There is not enough readable text to generate useful questions. Please upload more notes.")
    else:
        with st.spinner("Retrieving evidence and generating questions..."):
            try:
                # Keep the source text bounded for predictable hosting resource use.
                notes = notes[:MAX_INPUT_CHARS]
                chunks = build_note_chunks(notes)
                retrieval = corrective_retrieve(
                    notes=notes,
                    chunks=chunks,
                    query="Create exam-style multiple-choice questions covering the main concepts, definitions, relationships, and examples in these notes.",
                    allow_web=allow_web,
                    max_web_results=max_web_results,
                )

                if not retrieval["evidence"].strip():
                    st.error("The retrieval agent could not find enough evidence. Try a longer or clearer set of notes.")
                    st.stop()

                llm = ChatGoogleGenerativeAI(
                    model=MODEL_NAME,
                    google_api_key=API_KEY,
                    temperature=0.2,
                    max_retries=2,
                )
                schema = MCQSet
                system_prompt = f"""
You generate accurate multiple-choice questions for studying.
Use the evidence below as the factual basis. Evidence sources are labelled [NOTES] or [WEB].
Use notes as the primary source. Web evidence may supplement missing context, but do not pretend it came from the notes.
Generate exactly {num_questions} distinct questions.
Each question must have exactly four plausible, distinct options and exactly one correct answer.
The correct_answer field must match one option exactly.
Do not ask about facts not supported by the supplied evidence.
If the evidence is insufficient for the requested count, generate fewer supported questions rather than inventing facts.
Return only data matching this schema:
{schema.model_json_schema()}
"""
                result = llm.invoke([
                    SystemMessage(content=system_prompt),
                    HumanMessage(content="Retrieved evidence:\n\n" + retrieval["evidence"]),
                ])
                content = result.content
                if isinstance(content, list):
                    content = "\n".join(
                        part.get("text", "") if isinstance(part, dict) else str(part)
                        for part in content
                    )
                mcq_set = MCQSet.model_validate_json(str(content))
                if not mcq_set.questions:
                    st.error("Gemini did not return any valid questions. Please try again.")
                    st.stop()

                evaluated = []
                for item in mcq_set.questions:
                    q = item.model_dump()
                    fuzzy = assess_mcq(
                        question=q["question"],
                        options=q["options"],
                        correct_answer=q["correct_answer"],
                        evidence=retrieval["evidence"],
                    )
                    q["fuzzy"] = fuzzy
                    evaluated.append(q)

                st.session_state["generated_mcqs"] = evaluated
                st.session_state["retrieval_meta"] = {
                    "note_chunks": len(chunks),
                    "used_web": retrieval["used_web"],
                    "attempts": retrieval["attempts"],
                    "sources": retrieval["sources"],
                    "retrieval_note": retrieval["message"],
                }
                st.session_state["quiz_notes_name"] = uploaded_file.name if uploaded_file else "uploaded notes"
                st.success(f"Generated {len(evaluated)} question(s).")
            except ValidationError as exc:
                st.error("The generated response did not match the required MCQ format. Please try again.")
                with st.expander("Technical details"):
                    st.code(str(exc))
            except Exception as exc:
                error_text = str(exc)
                if "429" in error_text or "quota" in error_text.lower() or "resource_exhausted" in error_text.lower():
                    st.error("Gemini API quota or rate limit reached. Wait a little and try again.")
                elif "404" in error_text or "not found" in error_text.lower():
                    st.error(f"Gemini model '{MODEL_NAME}' was not found or is unavailable for this API key. Set GEMINI_MODEL in .env or Streamlit Secrets to a model available to your account.")
                elif "API key" in error_text.lower() or "api_key" in error_text.lower() or "unauthorized" in error_text.lower():
                    st.error("Gemini authentication failed. Check GOOGLE_API_KEY in .env or Streamlit Secrets.")
                else:
                    st.error("Something went wrong while generating questions.")
                    with st.expander("Technical details"):
                        st.code(error_text)

mcqs = st.session_state.get("generated_mcqs", [])
if mcqs:
    meta = st.session_state.get("retrieval_meta", {})
    st.divider()
    st.subheader("Retrieval report")
    c1, c2, c3 = st.columns(3)
    c1.metric("Note chunks", meta.get("note_chunks", 0))
    c2.metric("Correction attempts", meta.get("attempts", 0))
    c3.metric("Web fallback", "Used" if meta.get("used_web") else "Not needed")
    st.caption(meta.get("retrieval_note", ""))
    if meta.get("sources"):
        with st.expander("Evidence sources"):
            for source in meta["sources"]:
                st.write(f"- {source}")

    st.subheader("Question quality and difficulty")
    for i, q in enumerate(mcqs, start=1):
        fuzzy = q["fuzzy"]
        with st.expander(f"Q{i}: {q['question']}", expanded=False):
            a, b, c = st.columns(3)
            a.metric("Difficulty", fuzzy["difficulty"])
            b.metric("Quality score", f"{fuzzy['quality_score']}/100")
            c.metric("Evidence coverage", f"{fuzzy['evidence_coverage']}/100")
            if fuzzy["flags"]:
                st.warning("Review flag(s): " + "; ".join(fuzzy["flags"]))
            else:
                st.success("No heuristic quality flags.")

    st.download_button(
        "Download MCQs and assessment (TXT)",
        data="\n\n".join(
            f"Q{i}. {q['question']}\n" +
            "\n".join(f"  {chr(65+j)}. {opt}" for j, opt in enumerate(q["options"])) +
            f"\nCorrect answer: {q['correct_answer']}\nDifficulty: {q['fuzzy']['difficulty']}\nQuality score: {q['fuzzy']['quality_score']}/100\nFlags: {', '.join(q['fuzzy']['flags']) or 'None'}"
            for i, q in enumerate(mcqs, start=1)
        ),
        file_name="generated_mcqs_assessment.txt",
        mime="text/plain",
    )

    st.divider()
    st.subheader("Take the quiz")
    with st.form("quiz_form"):
        selected = {}
        for i, q in enumerate(mcqs, start=1):
            st.markdown(f"**Q{i}. {q['question']}**")
            selected[i] = st.radio(
                f"Choose an answer for question {i}",
                q["options"],
                index=None,
                key=f"answer_{i}",
                label_visibility="collapsed",
            )
        submitted = st.form_submit_button("Submit answers", type="primary")

    if submitted:
        score = 0
        for i, q in enumerate(mcqs, start=1):
            if selected[i] == q["correct_answer"]:
                score += 1
        st.subheader(f"Your score: {score} / {len(mcqs)}")
        st.progress(score / len(mcqs))
        for i, q in enumerate(mcqs, start=1):
            chosen = selected[i]
            is_correct = chosen == q["correct_answer"]
            st.markdown(f"**Q{i}. {q['question']}**")
            if chosen is None:
                st.warning(f"Not answered. Correct answer: {q['correct_answer']}")
            elif is_correct:
                st.success(f"Correct — {q['correct_answer']}")
            else:
                st.error(f"Your answer: {chosen}")
                st.info(f"Correct answer: {q['correct_answer']}")
            st.caption(
                f"Fuzzy assessment: {q['fuzzy']['difficulty']} difficulty, "
                f"quality {q['fuzzy']['quality_score']}/100. "
                "These are heuristic estimates, not proof of factual correctness."
            )
