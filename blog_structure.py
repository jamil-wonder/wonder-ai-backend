"""F4: blog length and structure.

  - 800-1,200 words
  - paragraphs of at most 100 words
  - H2 sections, with H3 sub-headings (### in the section text) where useful
  - an FAQ of 3-5 questions at the end
  - exactly one internal link (to one of the business's own pages)

Things that can be fixed without asking the writer again are fixed here (long
paragraphs split at a sentence, extra FAQs trimmed, extra or outside links
unlinked). What cannot - too short or too long, too few FAQs, no internal
link - is reported as a problem so the draft is rewritten."""
import re

MIN_WORDS = 800
MAX_WORDS = 1200
TARGET_WORDS = 1000
MAX_PARAGRAPH_WORDS = 100
FAQ_MIN = 3
FAQ_MAX = 5
MIN_SECTIONS = 4          # intro + body sections + FAQ

_LINK = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")
_FAQ_HEADING = re.compile(r"\b(faq|faqs|frequently asked)\b", re.IGNORECASE)


def _words(text: str) -> int:
    return len([w for w in re.split(r"\s+", text or "") if w.strip()])


def body_word_count(sections: list[dict]) -> int:
    return sum(_words(f"{s.get('heading', '')} {s.get('content', '')}") for s in sections)


def _norm_url(url: str) -> str:
    u = re.sub(r"^https?://(www\.)?", "", str(url or "").strip().lower())
    return u.rstrip("/")


def _is_internal(url: str, internal_links: list[str]) -> bool:
    """Is `url` one of the business's own pages? Pages may be saved as full URLs
    or as paths ("/menu"); a link matches when one ends with the other."""
    target = _norm_url(url)
    if len(target) < 2:
        return False
    for link in internal_links or []:
        want = _norm_url(link)
        if len(want) < 2:
            continue
        if target == want or target.endswith(want) or want.endswith(target):
            return True
    return False


# ----------------------------------------------------------------------------- prompt

def structure_prompt_block(internal_links: list[str] | None) -> str:
    link_rule = (
        f"- Include exactly ONE internal link, written as a markdown link [descriptive anchor text]({(internal_links or [''])[0]}), placed naturally inside a paragraph (not in the FAQ). Use no other links of any kind."
        if internal_links else
        "- Do not include any links."
    )
    return f"""
    LENGTH AND STRUCTURE (strict):
    - The whole article is {MIN_WORDS}-{MAX_WORDS} words (aim for about {TARGET_WORDS}). Shorter and tighter is better than padded.
    - Every paragraph is at most {MAX_PARAGRAPH_WORDS} words. Separate paragraphs with a blank line. Prefer 2-4 sentences.
    - Sections are H2 sections (the "heading" of each section). Inside a section, use "### Sub-heading" on its own line for H3 sub-headings where they help; use "- " bullet lines for lists.
    - Use {MIN_SECTIONS}-7 sections in total: an introduction, the body sections, and an FAQ section last.
    - The FAQ section (heading "Frequently asked questions") has {FAQ_MIN}-{FAQ_MAX} questions. Write each as "### The question?" on its own line followed by a short answer paragraph.
    {link_rule}
"""


def structure_feedback(problems: list[str]) -> str:
    return (
        "\n\n    STRUCTURE NOTE: your previous draft broke these rules: " + "; ".join(problems)
        + ". Write the article again and fix them."
    )


# ----------------------------------------------------------------------------- deterministic fixes

def _split_sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]


def _split_long_paragraph(paragraph: str, limit: int = MAX_PARAGRAPH_WORDS) -> list[str]:
    """Split at sentence boundaries into chunks of at most `limit` words. A single
    sentence longer than the limit is left whole (it is not ours to cut mid-sentence)."""
    if _words(paragraph) <= limit:
        return [paragraph]
    chunks, current = [], []
    for sentence in _split_sentences(paragraph):
        if current and _words(" ".join(current + [sentence])) > limit:
            chunks.append(" ".join(current))
            current = []
        current.append(sentence)
    if current:
        chunks.append(" ".join(current))
    return chunks


def _is_prose(block: str) -> bool:
    b = block.lstrip()
    return bool(b) and not b.startswith(("#", "- ", "* ")) and not re.match(r"^\d+[.)]\s", b)


def fix_structure(sections: list[dict], internal_links: list[str] | None) -> tuple[list[dict], dict]:
    """Return (sections, what was changed)."""
    fixes = {"paragraphsSplit": 0, "faqsTrimmed": 0, "linksRemoved": 0}
    out = [dict(s) for s in sections]

    # 1) long paragraphs -> split at a sentence
    for s in out:
        blocks = re.split(r"\n\s*\n", str(s.get("content", "")).strip())
        new_blocks = []
        for block in blocks:
            if _is_prose(block):
                parts = _split_long_paragraph(block)
                fixes["paragraphsSplit"] += len(parts) - 1
                new_blocks.extend(parts)
            else:
                new_blocks.append(block)
        s["content"] = "\n\n".join(b for b in new_blocks if b.strip())

    # 2) FAQ with more than 5 questions -> keep the first 5
    for s in out:
        if _FAQ_HEADING.search(str(s.get("heading", ""))):
            content = str(s.get("content", ""))
            parts = re.split(r"(?m)^(?=###\s)", content)
            intro, questions = (parts[0], parts[1:]) if parts and not parts[0].lstrip().startswith("###") else ("", parts)
            if len(questions) > FAQ_MAX:
                fixes["faqsTrimmed"] += len(questions) - FAQ_MAX
                s["content"] = (intro + "".join(questions[:FAQ_MAX])).strip()

    # 3) links: keep the first internal one, unlink everything else (keep the words)
    kept = [False]

    def relink(m):
        text, url = m.group(1), m.group(2)
        if (internal_links and _is_internal(url, internal_links)) and not kept[0]:
            kept[0] = True
            return m.group(0)
        fixes["linksRemoved"] += 1
        return text

    for s in out:
        s["content"] = _LINK.sub(relink, str(s.get("content", "")))
    return out, fixes


# ----------------------------------------------------------------------------- checks

def faq_question_count(sections: list[dict]) -> int:
    for s in reversed(sections):
        if _FAQ_HEADING.search(str(s.get("heading", ""))):
            return len(re.findall(r"(?m)^###\s", str(s.get("content", ""))))
    return 0


def internal_link_count(sections: list[dict], internal_links: list[str] | None) -> int:
    return sum(
        1 for s in sections for _, url in _LINK.findall(str(s.get("content", "")))
        if _is_internal(url, internal_links or [])
    )


def structure_problems(sections: list[dict], internal_links: list[str] | None) -> list[str]:
    """What is still wrong after fix_structure (empty = fine)."""
    problems = []
    words = body_word_count(sections)
    if words < MIN_WORDS:
        problems.append(f"it is {words} words - it must be {MIN_WORDS}-{MAX_WORDS}")
    elif words > MAX_WORDS:
        problems.append(f"it is {words} words - it must be {MIN_WORDS}-{MAX_WORDS}")
    if len(sections) < MIN_SECTIONS:
        problems.append(f"it has {len(sections)} sections - it needs at least {MIN_SECTIONS} (introduction, body sections, FAQ)")
    faqs = faq_question_count(sections)
    if faqs < FAQ_MIN:
        problems.append(f"the FAQ has {faqs} questions - it needs {FAQ_MIN}-{FAQ_MAX}, each written as a '### question' line")
    if internal_links and internal_link_count(sections, internal_links) < 1:
        problems.append(f"it has no internal link - include exactly one markdown link to {internal_links[0]}")
    return problems
