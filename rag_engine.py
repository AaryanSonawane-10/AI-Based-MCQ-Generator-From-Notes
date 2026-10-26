from typing import List, Dict, Any
import re

def split_into_chunks(text: str, chunk_size: int = 900, overlap: int = 140) -> List[str]:
    """Simple character chunking that preserves overlap between adjacent passages."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        # Prefer a sentence/word boundary near the end of a chunk.
        if end < len(text):
            boundary = max(text.rfind(". ", start + chunk_size // 2, end),
                           text.rfind("; ", start + chunk_size // 2, end),
                           text.rfind(" ", start + chunk_size // 2, end))
            if boundary > start:
                end = boundary + 1
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [c for c in chunks if c]

def build_note_chunks(notes: str) -> List[str]:
    return split_into_chunks(notes)

def _rank_chunks(query: str, chunks: List[str], top_k: int = 5) -> List[tuple]:
    """Rank note chunks with TF-IDF cosine similarity (no separate embedding API needed)."""
    if not chunks:
        return []
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        matrix = vectorizer.fit_transform(chunks + [query])
        scores = cosine_similarity(matrix[-1], matrix[:-1]).ravel()
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [(chunks[i], float(score)) for i, score in ranked[:top_k] if score > 0]
    except ValueError:
        return []

def _search_web(query: str, max_results: int = 3) -> List[Dict[str, str]]:
    """Optional DuckDuckGo fallback. Search is attempted only if note evidence is weak."""
    try:
        from ddgs import DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            return []
    results = []
    try:
        with DDGS() as ddgs:
            for item in ddgs.text(query, max_results=max_results):
                title = str(item.get("title", "")).strip()
                body = str(item.get("body", "")).strip()
                href = str(item.get("href", item.get("url", ""))).strip()
                if body:
                    results.append({"title": title, "body": body, "url": href})
    except Exception:
        return []
    return results

def corrective_retrieve(
    notes: str,
    chunks: List[str],
    query: str,
    allow_web: bool = True,
    max_web_results: int = 3,
) -> Dict[str, Any]:
    """
    A lightweight corrective retrieval loop:
    1) retrieve relevant chunks;
    2) check evidence overlap and score;
    3) retry with a refined query;
    4) if still weak, use web search when enabled.
    """
    attempts = 0
    ranked = _rank_chunks(query, chunks, top_k=6)
    attempts += 1

    if not ranked:
        # Retry with a compact query emphasizing common exam content.
        retry_query = "key concepts definitions examples processes relationships " + query
        ranked = _rank_chunks(retry_query, chunks, top_k=8)
        attempts += 1

    note_evidence = "\n\n".join(
        f"[NOTES | similarity={score:.2f}]\n{chunk}" for chunk, score in ranked
    )
    # This is a heuristic relevance check, not an LLM truth check.
    strong_enough = bool(ranked) and max(score for _, score in ranked) >= 0.015 and len(note_evidence) >= 250

    sources = ["Uploaded notes"] if note_evidence else []
    web_evidence = ""
    used_web = False
    if allow_web and not strong_enough:
        web_query = " ".join(query.split()[:24]) + " educational explanation"
        web_results = _search_web(web_query, max_results=max_web_results)
        attempts += 1
        if web_results:
            used_web = True
            snippets = []
            for result in web_results:
                snippets.append(
                    f"[WEB | {result['title']} | {result['url']}]\n{result['body']}"
                )
                if result["url"]:
                    sources.append(result["url"])
            web_evidence = "\n\n".join(snippets)

    evidence = "\n\n".join(part for part in [note_evidence, web_evidence] if part).strip()
    if not evidence and notes.strip():
        # Last-resort notes-only fallback: include initial chunks, still labelled.
        evidence = "\n\n".join(f"[NOTES | fallback passage]\n{x}" for x in chunks[:4])
        if evidence:
            sources = ["Uploaded notes (fallback passages)"]

    message = (
        "Relevant note passages were retrieved."
        if strong_enough else
        "Note retrieval looked weak; web fallback was used."
        if used_web else
        "Retrieval confidence was weak; using available note passages. Review generated questions carefully."
    )
    return {
        "evidence": evidence,
        "used_web": used_web,
        "attempts": attempts,
        "sources": list(dict.fromkeys(sources)),
        "message": message,
    }
