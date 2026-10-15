"""L1: did we actually get the website? A scan of an error page, a security-check
page or an empty shell must never turn into a Wonder Score, so the scraper
reports whether the fetch worked - and, if not, why - alongside its result."""
import re

MIN_VISIBLE_TEXT_CHARS = 100

_TAGS_TO_DROP = re.compile(r"<(script|style|noscript)\b.*?</\1>", re.I | re.S)
_ANY_TAG = re.compile(r"<[^>]+>")


def visible_text_length(html: str) -> int:
    text = _ANY_TAG.sub(" ", _TAGS_TO_DROP.sub(" ", html or ""))
    return len(re.sub(r"\s+", " ", text).strip())


def assess_fetch(http_status, html: str, blocked: bool = False) -> dict:
    """{"ok", "http_status", "reason"}. `reason` is a plain sentence for the user."""
    status = http_status if isinstance(http_status, int) else None
    if status is not None and status >= 400:
        return {"ok": False, "http_status": status, "reason": f"The website answered with an error (HTTP {status})."}
    if blocked:
        return {
            "ok": False,
            "http_status": status,
            "reason": "The website blocked our scan (it asked for a security check).",
        }
    if visible_text_length(html) < MIN_VISIBLE_TEXT_CHARS:
        return {"ok": False, "http_status": status, "reason": "The website's page came back empty."}
    return {"ok": True, "http_status": status, "reason": None}
