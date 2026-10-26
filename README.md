# AI-Based MCQ Generator from Notes — Corrective RAG + Fuzzy Logic

A Streamlit study app that accepts PDF/TXT notes, retrieves relevant passages, generates MCQs with Gemini, estimates question difficulty and quality using an explainable fuzzy-rule system, and provides a scored quiz plus TXT download.

## Features

- PDF and TXT upload with text extraction.
- Chunking and TF-IDF retrieval over uploaded notes.
- Corrective retrieval loop: retry with a revised query when note retrieval is weak.
- Optional DuckDuckGo web fallback when note retrieval looks weak.
- Gemini MCQ generation with Pydantic schema validation.
- Fuzzy-rule difficulty estimate (Easy / Medium / Hard).
- Fuzzy heuristic quality score and review flags.
- Interactive quiz, score, answer review, and TXT download.
- API key from local environment or Streamlit Cloud Secrets.

## 1. Install

Python 3.10+ recommended.

```bash
python -m venv .venv
```

Activate it:

- Windows: `.venv\Scripts\activate`
- macOS/Linux: `source .venv/bin/activate`

Then install dependencies:

```bash
pip install -r requirements.txt
```

## 2. Add your API key

Copy `.env.example` to `.env` and replace the placeholder with your real Gemini API key. Never commit `.env`.

```env
GOOGLE_API_KEY=your_real_key
GEMINI_MODEL=gemini-2.5-flash
```

The model name is configurable because model availability depends on the account and API endpoint. If this model is unavailable for your key, set `GEMINI_MODEL` to a currently supported model from your Google AI Studio model list.

## 3. Run locally

```bash
streamlit run app.py
```

Upload a text-based PDF or TXT file and click **Generate MCQs**.

## 4. Deploy on Streamlit Community Cloud

Push the project files to your GitHub repository. Do not upload `.env`.

In Streamlit Community Cloud, open the app's settings and add these secrets:

```toml
GOOGLE_API_KEY = "your_real_key"
GEMINI_MODEL = "gemini-2.5-flash"
```

If your account does not have access to that model, replace it with one available to your API key.

## How Corrective RAG works here

1. Split the uploaded notes into overlapping passages.
2. Rank note passages using TF-IDF cosine similarity.
3. If retrieval appears weak, retry with a refined query.
4. If enabled and note retrieval still appears weak, search DuckDuckGo and add labelled web snippets.
5. Ask Gemini to use the retrieved evidence and label web-derived context separately from notes.

This is a lightweight, transparent corrective retrieval pipeline, not a fully autonomous production-grade agent. Web search may fail due to network/rate limits or provider changes. For important work, check the displayed source links and question answers.

## How fuzzy logic works here

The `fuzzy_logic.py` module defines triangular and trapezoidal membership functions and explicit fuzzy rules. Difficulty uses question length and reasoning/concept cues. The quality score combines lexical overlap with evidence, option diversity, answer validity, and option-length balance.

Important limitations:
- Lexical overlap is only a rough proxy for evidence support.
- A high quality score does not prove that a question or answer is factually correct.
- Review flagged questions before using them for formal assessment.
- Fuzzy difficulty is heuristic and should be calibrated against human review.

## Files

- `app.py` — Streamlit interface, Gemini call, quiz, and downloads.
- `schemas.py` — Pydantic MCQ schema and validation.
- `rag_engine.py` — chunking, TF-IDF retrieval, corrective retry, optional web fallback.
- `fuzzy_logic.py` — fuzzy membership functions, inference, quality score.
- `requirements.txt` — dependencies.
- `.env.example` — environment variable template.

## Security

- Never commit `.env`, API keys, or `.streamlit/secrets.toml`.
- Rotate your API key immediately if it is accidentally exposed.
- Web search results are external, may be inaccurate, and are used only as supplementary evidence when enabled.
