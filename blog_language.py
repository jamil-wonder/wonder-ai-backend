"""F3: blogs are written in the business's own English.

Which English: (1) an explicit choice on the profile, else (2) the country of the
business's address, else (3) the region in the site's <html lang> ("en-GB") -
only after the address, because many site builders stamp "en-US" on every site -
else (4) the site's domain ending, else British English. The writer is told which English to
use, and the finished text is checked word by word for the other variety's
spellings (color / colour, organize / organise)."""
import re

# country (as the crawl normalises it) -> English locale
COUNTRY_LOCALE = {
    "UK": "en-GB", "United Kingdom": "en-GB", "England": "en-GB", "Scotland": "en-GB", "Wales": "en-GB",
    "Northern Ireland": "en-GB", "GB": "en-GB",
    "US": "en-US", "USA": "en-US", "United States": "en-US",
    "Ireland": "en-IE", "Australia": "en-AU", "New Zealand": "en-NZ", "Canada": "en-CA",
    "India": "en-IN", "South Africa": "en-ZA", "Singapore": "en-SG",
    "UAE": "en-GB", "Bangladesh": "en-GB",
}
TLD_LOCALE = {
    ".co.uk": "en-GB", ".uk": "en-GB", ".ie": "en-IE", ".com.au": "en-AU", ".au": "en-AU", ".co.nz": "en-NZ",
    ".nz": "en-NZ", ".ca": "en-CA", ".co.in": "en-IN", ".in": "en-IN", ".co.za": "en-ZA", ".za": "en-ZA", ".sg": "en-SG",
}
SUPPORTED = {"en-GB", "en-US", "en-IE", "en-AU", "en-NZ", "en-CA", "en-IN", "en-ZA", "en-SG"}
DEFAULT_LOCALE = "en-GB"

LABELS = {
    "en-GB": "British English", "en-US": "American English", "en-IE": "Irish English", "en-AU": "Australian English",
    "en-NZ": "New Zealand English", "en-CA": "Canadian English", "en-IN": "Indian English",
    "en-ZA": "South African English", "en-SG": "Singapore English",
}


def _clean(value) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def normalize_locale(value) -> str | None:
    """'en_gb' / 'EN-GB' / 'en-gb' -> 'en-GB'; None unless it is a supported English locale."""
    m = re.match(r"^\s*en[-_]([A-Za-z]{2})\s*$", str(value or ""), re.IGNORECASE)
    if not m:
        return None
    code = f"en-{m.group(1).upper()}"
    return code if code in SUPPORTED else None


def resolve_blog_language(business_doc: dict) -> dict:
    """{"code": "en-GB", "label": "British English", "source": "profile"|"website language"|"address"|"domain"|"default"}"""
    doc = business_doc if isinstance(business_doc, dict) else {}
    scrape = doc.get("latest_scrape_result") if isinstance(doc.get("latest_scrape_result"), dict) else {}

    def result(code, source):
        return {"code": code, "label": LABELS[code], "source": source}

    explicit = normalize_locale(doc.get("blogLanguage"))
    if explicit:
        return result(explicit, "profile")

    # The country of the business's address is the strongest signal: many site
    # builders stamp <html lang="en-US"> on every page, including London
    # restaurants, so the declared language only decides when the address does not.
    countries = []
    for loc in (scrape.get("locations") if isinstance(scrape.get("locations"), list) else []):
        if isinstance(loc, dict) and _clean(loc.get("country")):
            countries.append(_clean(loc.get("country")))
    for country in countries:
        if country in COUNTRY_LOCALE:
            return result(COUNTRY_LOCALE[country], "address")

    declared = normalize_locale(scrape.get("language"))
    if declared:
        return result(declared, "website language")

    host = re.sub(r"^https?://", "", _clean(doc.get("url") or doc.get("normalized_domain"))).split("/")[0].lower()
    for suffix in sorted(TLD_LOCALE, key=len, reverse=True):
        if host.endswith(suffix):
            return result(TLD_LOCALE[suffix], "domain")

    return result(DEFAULT_LOCALE, "default")


# ----------------------------------------------------------------------------- spelling

# spelling family of each locale: "us" | "ca" (colour but -ize) | "gb" (colour, -ise)
def spelling_family(code: str) -> str:
    if code == "en-US":
        return "us"
    if code == "en-CA":
        return "ca"
    return "gb"


_VERB_SUFFIXES = ("e", "es", "ed", "ing", "er", "ers", "ation", "ations", "ational")
_NOUN_SUFFIXES = ("", "s", "ed", "ing", "ful", "less", "ation", "ations", "y", "ies")

# (American stem, British stem, allowed endings)
_PAIRS: list[tuple[str, str, tuple]] = [
    # -or / -our
    ("color", "colour", _NOUN_SUFFIXES), ("favorite", "favourite", ("", "s")), ("flavor", "flavour", _NOUN_SUFFIXES),
    ("honor", "honour", _NOUN_SUFFIXES), ("neighbor", "neighbour", _NOUN_SUFFIXES + ("hood", "hoods")),
    ("behavior", "behaviour", ("", "s", "al")), ("labor", "labour", _NOUN_SUFFIXES), ("humor", "humour", _NOUN_SUFFIXES),
    ("harbor", "harbour", ("", "s")), ("rumor", "rumour", ("", "s")), ("savor", "savour", ("", "s", "ed", "ing", "y")),
    # -er / -re
    ("center", "centre", ("", "s", "d")), ("theater", "theatre", ("", "s")),
    ("liter", "litre", ("", "s")), ("fiber", "fibre", ("", "s")),   # not "meter": a parking/gas meter is spelt that way everywhere
    # -ize / -ise  (not applied to Canadian English, which keeps -ize)
    ("organiz", "organis", _VERB_SUFFIXES), ("recogniz", "recognis", _VERB_SUFFIXES), ("realiz", "realis", _VERB_SUFFIXES),
    ("optimiz", "optimis", _VERB_SUFFIXES), ("customiz", "customis", _VERB_SUFFIXES), ("specializ", "specialis", _VERB_SUFFIXES),
    ("personaliz", "personalis", _VERB_SUFFIXES), ("prioritiz", "prioritis", _VERB_SUFFIXES), ("emphasiz", "emphasis", _VERB_SUFFIXES),
    ("minimiz", "minimis", _VERB_SUFFIXES), ("maximiz", "maximis", _VERB_SUFFIXES), ("summariz", "summaris", _VERB_SUFFIXES),
    ("apologiz", "apologis", _VERB_SUFFIXES), ("finaliz", "finalis", _VERB_SUFFIXES), ("utiliz", "utilis", _VERB_SUFFIXES),
    ("localiz", "localis", _VERB_SUFFIXES), ("visualiz", "visualis", _VERB_SUFFIXES), ("authoriz", "authoris", _VERB_SUFFIXES),
    ("popularize", "popularise", ("", "d", "s")), ("standardiz", "standardis", _VERB_SUFFIXES), ("modernize", "modernise", ("", "d", "s")),
    # -yze / -yse
    ("analyz", "analys", ("e", "es", "ed", "ing", "er", "ers")), ("paralyz", "paralys", ("e", "es", "ed", "ing")),
    # -ense / -ence, others
    ("defense", "defence", ("", "s")), ("offense", "offence", ("", "s")), ("catalog", "catalogue", ("", "s")),
    ("gray", "grey", ("", "s", "ish")), ("jewelry", "jewellery", ("",)), ("mold", "mould", ("", "s", "y")),
    ("skeptic", "sceptic", ("", "s", "al")), ("aluminum", "aluminium", ("",)), ("cozy", "cosy", ("",)),
    ("pajamas", "pyjamas", ("",)), ("donut", "doughnut", ("", "s")),
]
# -ise endings only apply to the -ize group; Canadian English keeps -ize but still uses -our / -re
_IZE_STEMS = {"organiz", "recogniz", "realiz", "optimiz", "customiz", "specializ", "personaliz", "prioritiz", "emphasiz",
              "minimiz", "maximiz", "summariz", "apologiz", "finaliz", "utiliz", "localiz", "visualiz", "authoriz",
              "popularize", "standardiz", "modernize"}


def _swap_case(original: str, replacement: str) -> str:
    if original.isupper() and len(original) > 1:
        return replacement.upper()
    if original[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def localize_spelling(sections: list[dict], code: str, protect_text: str = "") -> tuple[list[dict], int]:
    """Rewrite the other variety's spellings in the article body. Returns
    (sections, number of words changed). Words that are part of a proper name
    (capitalised mid-sentence) or that appear in `protect_text` (the business
    name and facts) are left alone."""
    family = spelling_family(code)
    protected = {w for w in re.findall(r"[A-Za-z']+", protect_text.lower())}
    rules = []
    for us, gb, endings in _PAIRS:
        if family == "ca" and us in _IZE_STEMS:
            continue
        src, dst = (gb, us) if family == "us" else (us, gb)
        if src == dst:
            continue
        alt = "|".join(re.escape(e) for e in endings)
        rules.append((re.compile(rf"\b({re.escape(src)})({alt})\b", re.IGNORECASE), dst))

    fixes = 0

    def fix_text(text: str) -> str:
        nonlocal fixes
        out = text
        for pattern, dst in rules:
            def repl(m, dst=dst):
                nonlocal fixes
                word = m.group(0)
                # a Capitalised word in the middle of a sentence is likely a name (Center Parcs)
                before = out[: m.start()].rstrip()
                mid_sentence = bool(before) and before[-1] not in ".!?\n:#" and word[:1].isupper() and not word.isupper()
                if mid_sentence or word.lower() in protected:
                    return word
                fixes += 1
                return _swap_case(word, dst + m.group(2))
            out = pattern.sub(repl, out)
        return out

    new_sections = []
    for s in sections:
        s2 = dict(s)
        s2["content"] = fix_text(str(s.get("content", "")))
        s2["heading"] = fix_text(str(s.get("heading", "")))
        new_sections.append(s2)
    return new_sections, fixes


# ----------------------------------------------------------------------------- prompt

_VOCAB = {
    "gb": "Use British vocabulary and conventions: colour, organise, centre, favourite, programme (event), whilst is fine; holiday not vacation, takeaway not takeout, autumn not fall, car park not parking lot, bill not check, mobile not cell phone, queue not line. Write dates as 14 October 2026 and use 24-hour or am/pm times consistently.",
    "us": "Use American vocabulary and conventions: color, organize, center, favorite; vacation, takeout, fall, parking lot, check (the bill), cell phone, line. Write dates as October 14, 2026.",
    "ca": "Use Canadian conventions: colour, centre, favourite, but -ize endings (organize, recognize); dates as October 14, 2026 or 14 October 2026 consistently.",
}


def language_prompt_block(code: str) -> str:
    return f"""
    LANGUAGE: Write in {LABELS.get(code, 'British English')} ({code}). {_VOCAB[spelling_family(code)]} Do not mix varieties, and do not change spellings inside names, quotes or the facts above.
"""
