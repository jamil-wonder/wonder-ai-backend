"""F1: blog claims must come from the site or the profile.

A generated article may state facts about the business ONLY if they are in the
fact pack built from the saved profile and the website crawl. Each statement
the model considers a claim comes back tagged with its source:
  - a fact id (F3)  -> allowed, and listed so the owner can confirm it
  - "general"       -> allowed only if it is NOT specific (no figures, prices,
                       awards, ratings, years, studies, named third parties).
                       Specific "general" claims are dropped from the text;
                       absolute wording ("the best", "always") is softened.
Everything dropped / worth double-checking goes in a "Check before publishing"
list shown with the draft."""
import re

MAX_FACTS = 40

# ----------------------------------------------------------------------------- fact pack


def _clean(value) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", str(text or "")) if s and s.strip()]


def build_fact_pack(business_doc: dict) -> list[dict]:
    """[{"id": "F1", "text": "...", "source": "profile" | "website"}] from the
    saved profile and the website crawl. Nothing else is a fact."""
    doc = business_doc if isinstance(business_doc, dict) else {}
    scrape = doc.get("latest_scrape_result") if isinstance(doc.get("latest_scrape_result"), dict) else {}
    facts: list[dict] = []

    def add(text, source):
        t = _clean(text)
        if t and len(facts) < MAX_FACTS and all(f["text"].lower() != t.lower() for f in facts):
            facts.append({"id": f"F{len(facts) + 1}", "text": t[:240], "source": source})

    name = _clean(doc.get("businessName"))
    if name:
        add(f"The business is called {name}.", "profile")
    if _clean(doc.get("category")):
        add(f"Type of business: {_clean(doc.get('category'))}.", "profile")
    if _clean(doc.get("sub_type")):
        add(f"Specific type: {_clean(doc.get('sub_type'))}.", "profile")
    if _clean(doc.get("location")):
        add(f"Location: {_clean(doc.get('location'))}.", "profile")
    for service in (doc.get("services") if isinstance(doc.get("services"), list) else [])[:10]:
        add(f"Offers: {service}.", "profile")
    if _clean(doc.get("targetAudience")):
        add(f"Target audience: {_clean(doc.get('targetAudience'))}.", "profile")
    for key in ("businessDescription", "aiDescription"):
        for sentence in _sentences(doc.get(key))[:4]:
            add(sentence, "profile")

    # from the crawl of the business's own website
    for address in (scrape.get("addresses") if isinstance(scrape.get("addresses"), list) else [])[:2]:
        add(f"Address on the website: {address}.", "website")
    for phone in (scrape.get("phones") if isinstance(scrape.get("phones"), list) else [])[:1]:
        add(f"Phone number on the website: {phone}.", "website")
    for hours in (scrape.get("openingHours") if isinstance(scrape.get("openingHours"), list) else [])[:7]:
        add(f"Opening hours on the website: {hours}.", "website")
    if _clean(scrape.get("description")):
        for sentence in _sentences(scrape.get("description"))[:2]:
            add(sentence, "website")
    return facts


def facts_prompt_block(facts: list[dict]) -> str:
    if not facts:
        return "(no verified facts about this business are available - write only general advice and do not state facts about the business)"
    return "\n".join(f"{f['id']}: {f['text']}  [from the business's {f['source']}]" for f in facts)


CLAIM_RULES = """
    FACTS ABOUT THE BUSINESS - the ONLY facts about the business you may state:
{facts}

    CLAIM RULES (critical):
    - State nothing about the business that is not in the facts above: no prices, opening hours, awards, years in business, ratings, reviews, menu items, staff, history, customers or customer stories, statistics.
    - You may give general advice and how-to information about the topic, but keep it non-specific: no figures, no percentages, no named third parties, no "studies show" / "according to" / "research says".
    - Do not invent case examples, quotes or testimonials. If something would help but is not in the facts, leave it out.
    - Avoid absolutes ("the best", "always", "never", "guaranteed").
    - Return a "claims" array listing every sentence in the article that states a fact about the business or makes a specific factual statement about the world. Copy the sentence exactly, and give its source: the fact id it comes from (e.g. "F3"), or "general" for general-knowledge statements. Never label an invented statement with a fact id.
"""

# ----------------------------------------------------------------------------- vetting

_SPECIFIC = re.compile(
    r"(\d|[£$€%]|\b(?:award(?:s|ed)?|award-winning|michelin|rated|rating|ranked|ranking|reviews|stars?|"
    r"stud(?:y|ies)|research (?:shows|says|suggests|finds|indicates)|survey|statistics?|according to|since|founded|established|years? of|decades?|"
    r"customers say|critics?|voted|named best)\b)",
    re.IGNORECASE,
)

_SOFTEN = [
    (re.compile(r"\bthe best\b", re.I), "one of the best"),
    (re.compile(r"\bthe leading\b", re.I), "a leading"),
    (re.compile(r"\bworld-class\b", re.I), "high-quality"),
    (re.compile(r"\b(?:unmatched|unrivall?ed|unparalleled)\b", re.I), "strong"),
    (re.compile(r"\bguaranteed?\b", re.I), "designed to"),
    (re.compile(r"\balways\b", re.I), "often"),
    (re.compile(r"\bnever\b", re.I), "rarely"),
]


def is_specific(text: str) -> bool:
    return bool(_SPECIFIC.search(str(text or "")))


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()


def normalize_claims(raw_claims, facts: list[dict]) -> list[dict]:
    """Keep well-formed claims; an unknown source id becomes "general"."""
    by_id = {f["id"]: f for f in facts}
    out = []
    for item in raw_claims if isinstance(raw_claims, list) else []:
        if not isinstance(item, dict):
            continue
        text = _clean(item.get("text"))
        if not text:
            continue
        src = _clean(item.get("source")).upper()
        fact = by_id.get(src)
        out.append({
            "text": text[:400],
            "source": fact["id"] if fact else "general",
            "sourceLabel": f"Your {fact['source']}" if fact else "General knowledge",
            "fact": fact["text"] if fact else None,
        })
    return out[:60]


def _drop_sentence(content: str, claim_text: str) -> tuple[str, bool]:
    """Remove the sentence (or bullet line) that holds `claim_text`."""
    target = _norm(claim_text)
    if not target:
        return content, False
    pieces = re.split(r"(?<=[.!?])(\s+)|(\n+)", content)
    pieces = [p for p in pieces if p is not None]
    kept, dropped = [], False
    i = 0
    while i < len(pieces):
        piece = pieces[i]
        if piece.strip() and not dropped:
            n = _norm(piece)
            if n and (target in n or (len(n) > 25 and n in target)):
                dropped = True
                # also swallow the separator that followed it
                if i + 1 < len(pieces) and not pieces[i + 1].strip():
                    i += 1
                i += 1
                continue
        kept.append(piece)
        i += 1
    return "".join(kept).strip(), dropped


# Figures that only make sense as claims about a business. A sentence holding one
# that appears in none of the facts is invented, whether or not the model listed it.
_FIGURE = re.compile(
    r"([£$€]\s?\d[\d,.]*"
    r"|\d[\d,.]*\s?%"
    r"|\b(?:19|20)\d{2}\b"
    r"|\b\d[\d,]*\+?\s+(?:years?|guests|customers|clients|reviews|awards|stars|members|locations|branches)\b)",
    re.IGNORECASE,
)


def _unsupported_figures(sentence: str, facts_text: str) -> list[str]:
    return [m.group(0) for m in _FIGURE.finditer(sentence) if _norm(m.group(0)) not in facts_text]


def drop_unsupported_figures(sections: list[dict], facts: list[dict]) -> list[str]:
    """Safety net behind the model's own claim list: remove any sentence that
    states a price, percentage, year or count that is not in the fact pack."""
    facts_text = _norm(" ".join(f["text"] for f in facts))
    removed: list[str] = []
    for section in sections:
        for sentence in _sentences(section.get("content", "")):
            if _unsupported_figures(sentence, facts_text):
                new_content, ok = _drop_sentence(section.get("content", ""), sentence)
                if ok:
                    section["content"] = new_content
                    removed.append(sentence)
    return removed


def vet_blog(sections: list[dict], raw_claims, facts: list[dict]) -> dict:
    """Apply the claim rules to a generated article.
    Returns {"sections", "claims", "droppedClaims", "softenedCount", "checkBeforePublishing"}."""
    claims = normalize_claims(raw_claims, facts)
    out_sections = [dict(s) for s in sections]
    dropped: list[str] = []
    unplaced: list[str] = []
    softened = 0

    for claim in claims:
        if claim["source"] != "general":
            continue
        if is_specific(claim["text"]):
            placed = False
            for section in out_sections:
                new_content, removed = _drop_sentence(section.get("content", ""), claim["text"])
                if removed:
                    section["content"] = new_content
                    placed = True
                    break
            (dropped if placed else unplaced).append(claim["text"])
        else:
            for section in out_sections:
                content = section.get("content", "")
                if _norm(claim["text"]) and _norm(claim["text"]) in _norm(content):
                    # soften inside the sentence that carries the claim
                    new_content = content
                    for sentence in _sentences(content):
                        if _norm(claim["text"]) in _norm(sentence):
                            softer = sentence
                            for pattern, repl in _SOFTEN:
                                softer = pattern.sub(repl, softer)
                            if softer != sentence:
                                new_content = new_content.replace(sentence, softer, 1)
                                softened += 1
                            break
                    section["content"] = new_content
                    break

    # safety net: figures the model never listed as claims but that no fact supports
    dropped.extend(drop_unsupported_figures(out_sections, facts))

    # sections left empty by dropping are removed
    out_sections = [s for s in out_sections if _clean(s.get("content"))]

    claims = [c for c in claims if c["text"] not in dropped]
    check: list[str] = []
    seen_facts = set()
    for claim in claims:
        if claim["source"] != "general" and claim["source"] not in seen_facts and len(seen_facts) < 6:
            seen_facts.add(claim["source"])
            check.append(f"Confirm this is still true: \"{claim['fact']}\" (taken from {claim['sourceLabel'].lower()}).")
    if dropped:
        shown = "; ".join(f"\"{d[:90]}\"" for d in dropped[:2])
        check.append(f"We removed {len(dropped)} sentence{'s' if len(dropped) != 1 else ''} that made specific claims we couldn't source (e.g. {shown}). Add anything like this back only if it is true.")
    for text in unplaced[:2]:
        check.append(f"Check this statement - it makes a specific claim we couldn't source: \"{text[:120]}\".")
    check.append("Add a real detail only you know (a customer story, a dish, a project) - the draft deliberately doesn't invent one.")

    return {
        "sections": out_sections,
        "claims": claims,
        "droppedClaims": dropped,
        "softenedCount": softened,
        "checkBeforePublishing": check[:10],
    }
