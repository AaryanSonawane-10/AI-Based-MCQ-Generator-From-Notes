import re
from typing import Dict, List

def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))

def _triangular(x: float, a: float, b: float, c: float) -> float:
    """Triangular fuzzy membership function."""
    if x <= a or x >= c:
        return 0.0
    if x == b:
        return 1.0
    if x < b:
        return (x - a) / (b - a) if b != a else 1.0
    return (c - x) / (c - b) if c != b else 1.0

def _trapezoid(x: float, a: float, b: float, c: float, d: float) -> float:
    if x < a or x > d:
        return 0.0
    if b <= x <= c:
        return 1.0
    if a <= x < b:
        return (x - a) / (b - a) if b != a else 1.0
    return (d - x) / (d - c) if d != c else 1.0

def _words(text: str) -> set:
    return set(re.findall(r"\b[a-zA-Z][a-zA-Z0-9-]*\b", text.lower()))

def assess_mcq(question: str, options: List[str], correct_answer: str, evidence: str) -> Dict:
    """
    Small explainable fuzzy rule system:
    - Difficulty inputs: question length, conceptual cue count, option similarity.
    - Quality inputs: lexical evidence coverage, option diversity, answer membership.
    No external fuzzy package required; memberships and rules are explicit.
    """
    q_words = _words(question)
    evidence_words = _words(evidence)
    meaningful_q = {w for w in q_words if len(w) > 3}
    coverage = (
        len(meaningful_q & evidence_words) / max(1, len(meaningful_q))
    )
    coverage = _clamp(coverage)

    complexity_terms = {
        "compare", "analyze", "analyse", "evaluate", "infer", "relationship",
        "consequence", "differentiate", "calculate", "justify", "interpret",
        "mechanism", "application", "scenario", "why", "how"
    }
    concept_count = len(q_words & complexity_terms)
    length_signal = _clamp(len(question.split()) / 28.0)
    concept_signal = _clamp(concept_count / 3.0)

    option_words = [_words(x) for x in options]
    overlaps = []
    for i in range(len(option_words)):
        for j in range(i + 1, len(option_words)):
            union = option_words[i] | option_words[j]
            if union:
                overlaps.append(len(option_words[i] & option_words[j]) / len(union))
    option_similarity = sum(overlaps) / max(1, len(overlaps))
    diversity = _clamp(1.0 - option_similarity)

    answer_valid = float(correct_answer in options)
    option_lengths = [len(x.split()) for x in options]
    length_balance = 1.0 - _clamp(
        (max(option_lengths) - min(option_lengths)) / max(1, max(option_lengths))
    )

    # Fuzzy memberships for difficulty.
    easy_mu = max(_trapezoid(length_signal, 0, 0, 0.22, 0.48),
                  _trapezoid(concept_signal, 0, 0, 0.1, 0.45))
    medium_mu = _triangular((length_signal + concept_signal) / 2, 0.2, 0.52, 0.82)
    hard_mu = max(_trapezoid(length_signal, 0.48, 0.7, 1, 1),
                  _trapezoid(concept_signal, 0.35, 0.65, 1, 1))

    # Simple fuzzy inference rules: strong hard membership wins; otherwise compare.
    memberships = {"Easy": easy_mu, "Medium": medium_mu, "Hard": hard_mu}
    difficulty = max(memberships, key=memberships.get)

    # Fuzzy quality score via weighted rule aggregation (0-100).
    # Evidence coverage and valid schema matter more than surface style.
    quality_raw = (
        0.55 * coverage +
        0.20 * diversity +
        0.15 * length_balance +
        0.10 * answer_valid
    )
    quality_score = round(100 * _clamp(quality_raw))
    coverage_score = round(100 * coverage)

    flags = []
    if coverage < 0.20:
        flags.append("Low lexical overlap with retrieved evidence")
    if diversity < 0.35:
        flags.append("Options may be too similar")
    if not answer_valid:
        flags.append("Correct answer is not one of the options")
    if length_balance < 0.35:
        flags.append("Option lengths are imbalanced")
    if quality_score < 55:
        flags.append("Low heuristic quality score")

    return {
        "difficulty": difficulty,
        "difficulty_memberships": {k: round(float(v), 3) for k, v in memberships.items()},
        "quality_score": quality_score,
        "evidence_coverage": coverage_score,
        "flags": flags,
        "method_note": "Explainable fuzzy heuristics; not a factuality guarantee."
    }
