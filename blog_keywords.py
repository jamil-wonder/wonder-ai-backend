"""F2: keywords steer the topic - they are never inserted word for word.

The owner's keywords are handed to the writer as TOPICS to cover. A multi-word
keyword phrase may appear verbatim at most once in the article body; a draft
that repeats one is rejected and rewritten, and if it still does after the
retries the best attempt is kept and the repeats are listed for the owner.
(The title is free to use its keyword: that is what a title is for. Single
words cannot be varied, so only phrases of two or more words are counted.)"""
import re

MAX_VERBATIM = 1
MIN_WORDS_IN_PHRASE = 2
MAX_RETRIES = 2


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()


def clean_keyword_list(keywords, limit: int = 12) -> list[str]:
    """Distinct, non-empty keyword strings, in order."""
    seen, out = set(), []
    for k in keywords or []:
        text = re.sub(r"\s+", " ", str(k or "")).strip()
        key = _norm(text)
        if text and key and key not in seen:
            seen.add(key)
            out.append(text)
    return out[:limit]


def countable_phrases(keywords) -> list[str]:
    return [k for k in clean_keyword_list(keywords) if len(_norm(k).split()) >= MIN_WORDS_IN_PHRASE]


def count_verbatim(sections: list[dict], keywords) -> dict[str, int]:
    """How many times each multi-word keyword phrase appears word for word in the
    headings and body of the article (case, punctuation and spacing ignored)."""
    body = " " + _norm(" ".join(f"{s.get('heading', '')} {s.get('content', '')}" for s in sections)) + " "
    counts: dict[str, int] = {}
    for phrase in countable_phrases(keywords):
        needle = " " + _norm(phrase) + " "
        # overlapping occurrences count too ("a b a b a b" for "a b")
        n, start = 0, 0
        while True:
            idx = body.find(needle, start)
            if idx == -1:
                break
            n += 1
            start = idx + len(needle) - 1
        counts[phrase] = n
    return counts


def violations(sections: list[dict], keywords) -> dict[str, int]:
    """Phrases used word for word more than once."""
    return {k: n for k, n in count_verbatim(sections, keywords).items() if n > MAX_VERBATIM}


def topics_prompt_block(keywords) -> str:
    topics = clean_keyword_list(keywords)
    if not topics:
        return ""
    listed = "\n".join(f"      - {t}" for t in topics)
    return f"""
    TOPICS TO COVER (the owner's keywords - themes, NOT phrases to insert):
{listed}
    - Cover these subjects naturally, in your own words and with varied wording.
    - Never repeat a topic phrase word for word more than once in the whole article. Use synonyms, a different word order, or a pronoun for repeat references.
    - Do not stuff keywords into headings or bullet lists.
"""


def retry_feedback(bad: dict[str, int]) -> str:
    listed = "; ".join(f'"{k}" ({n} times)' for k, n in bad.items())
    return (
        "\n\n    REWRITE NOTE: your previous draft repeated these exact phrases more than once: "
        f"{listed}. Write the article again. Each phrase may appear word for word at most ONCE in the whole article; "
        "express every other mention with different wording."
    )
