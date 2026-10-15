"""F6: a real readability score for a blog draft - the Flesch Reading Ease (English).

206.835 - 1.015 x (words per sentence) - 84.6 x (syllables per word), clamped to 0-100,
with the usual band label. Syllables are counted with a vowel-group heuristic, which is
close enough for a score that is read as a band, not to the decimal.
"""
import re

_WORD = re.compile(r"[A-Za-z][A-Za-z'’-]*")
_SENTENCE_END = re.compile(r"[.!?]+(?:\s|$)")

BANDS = [
    (90, "Very easy"),
    (80, "Easy"),
    (70, "Fairly easy"),
    (60, "Plain English"),
    (50, "Fairly difficult"),
    (30, "Difficult"),
    (0, "Very difficult"),
]


def count_syllables(word: str) -> int:
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return 0
    if len(w) <= 3:
        return 1
    w = re.sub(r"(?:[^laeiouy]es|ed|[^laeiouy]e)$", "", w)
    w = re.sub(r"^y", "", w)
    groups = re.findall(r"[aeiouy]+", w)
    return max(1, len(groups))


def band_for(score: float) -> str:
    for floor, label in BANDS:
        if score >= floor:
            return label
    return BANDS[-1][1]


def flesch_reading_ease(text: str) -> dict | None:
    """{"score": 0-100 int, "band": str, "words": n, "sentences": n} or None for text too short to judge."""
    text = re.sub(r"[#*_`>\[\]()]", " ", str(text or ""))
    words = _WORD.findall(text)
    if len(words) < 30:
        return None
    sentences = max(1, len(_SENTENCE_END.findall(text)))
    syllables = sum(count_syllables(w) for w in words)
    score = 206.835 - 1.015 * (len(words) / sentences) - 84.6 * (syllables / len(words))
    score = max(0, min(100, round(score)))
    return {"score": score, "band": band_for(score), "words": len(words), "sentences": sentences}


def readability_of_sections(sections: list[dict]) -> dict | None:
    body = "\n\n".join(str((s or {}).get("content") or (s or {}).get("body") or "") for s in sections or [])
    return flesch_reading_ease(body)
