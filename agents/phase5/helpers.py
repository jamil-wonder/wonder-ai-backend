import json
import re

from .config import NON_COMPETITOR_DOMAINS


def _normalize_domain(value: str) -> str:
    raw = str(value or "").strip().lower()
    if not raw:
        return ""
    raw = raw.replace("www.", "")
    raw = re.sub(r"^https?://", "", raw)
    raw = raw.split("/")[0]
    return raw


def _safe_json_parse(text: str) -> dict:
    payload = (text or "").strip()
    if payload.startswith("```json"):
        payload = payload[7:]
    if payload.endswith("```"):
        payload = payload[:-3]
    payload = payload.strip()
    try:
        return json.loads(payload)
    except Exception:
        start = payload.find("{")
        end = payload.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(payload[start:end + 1])
            except Exception:
                return {}
    return {}


def _is_rate_limited_error(exc: Exception) -> bool:
    message = str(exc or "").lower()
    return (
        "429" in message
        or "resource_exhausted" in message
        or "rate limit" in message
        or "too many requests" in message
        or "quota" in message
    )


def _is_target_domain_match(candidate: str, target_domain: str) -> bool:
    c = _normalize_domain(candidate)
    t = _normalize_domain(target_domain)
    if not c or not t:
        return False
    return c == t or c.endswith(f".{t}")


def _extract_domains_from_text(text: str) -> list[str]:
    if not text:
        return []
    pattern = r"(?:https?://)?(?:www\.)?([a-z0-9-]+(?:\.[a-z0-9-]+)+)"
    found = re.findall(pattern, text.lower())
    clean = []
    for item in found:
        d = _normalize_domain(item)
        if d and "vertexaisearch.cloud.google.com" not in d:
            clean.append(d)
    return list(dict.fromkeys(clean))


def _is_non_competitor_domain(domain: str) -> bool:
    d = _normalize_domain(domain)
    if not d:
        return True

    if d in NON_COMPETITOR_DOMAINS:
        return True
    for banned in NON_COMPETITOR_DOMAINS:
        try:
            root = str(banned).split(".")[0]
        except Exception:
            root = banned
        if root and root in d:
            return True
    return False


def _looks_like_platform_domain(domain: str) -> bool:
    """Heuristic to detect platforms, directories, or profile hosts we should exclude from competitor lists."""
    d = _normalize_domain(domain)
    if not d:
        return False
    patterns = [
        "yelp",
        "tripadvisor",
        "booking",
        "expedia",
        "airbnb",
        "facebook",
        "instagram",
        "linkedin",
        "wordpress",
        "wixsite",
        "pages",
        "google",
        "bing",
        "timeout",
        "craigslist",
        "yellowpages",
        "opentable",
        "kayak",
    ]
    low = d.lower()
    for p in patterns:
        if p in low:
            return True
    if low.count(".") >= 2 and low.endswith(".wordpress.com"):
        return True
    return False


def _align_reasoning_with_status(
    *,
    reasoning: str,
    status: str,
    domain: str,
    position: int | None,
    source_count: int,
    reference_count: int,
) -> str:
    text = str(reasoning or "").strip()
    low = text.lower()

    positive_markers = [
        "appears",
        "mentioned",
        "listed",
        "rank",
        "position",
        "in top",
        "strong position",
    ]
    negative_markers = [
        "not mentioned",
        "absent",
        "not present",
        "not listed",
        "does not appear",
        "missing",
    ]

    if status == "Not Mentioned":
        contradicts = any(m in low for m in positive_markers) and not any(n in low for n in negative_markers)
        if not text or contradicts:
            return (
                f"Target {domain} was not verified in top returned results. "
                f"Evidence captured from third-party domains only (sources={source_count}, references={reference_count})."
            )
        return text

    contradicts = any(n in low for n in negative_markers)
    if not text or contradicts:
        if isinstance(position, int):
            return f"Target {domain} appears in returned results around rank #{position} based on captured evidence."
        return f"Target {domain} appears in returned results based on captured evidence."
    return text


def _build_fallback_concise_answer(
    *,
    question_text: str,
    status: str,
    position: int | None,
    references: list[str],
    sources: list[str],
    competitor_scores: list[dict],
    domain: str,
) -> str:
    seen_domains: list[str] = []
    for d in [*(references or []), *(sources or [])]:
        n = _normalize_domain(str(d))
        if n and n not in seen_domains:
            seen_domains.append(n)
    top_refs = seen_domains[:3]

    comp_names: list[str] = []
    for c in (competitor_scores or [])[:3]:
        d = _normalize_domain(str(c.get("domain") or ""))
        if d and d not in comp_names:
            comp_names.append(d)

    if status == "Mentioned":
        if isinstance(position, int):
            lead = f"For \"{question_text}\", your brand appears in the returned set around rank #{position}."
        else:
            lead = f"For \"{question_text}\", your brand appears in the returned set."
        src = (
            f"Evidence was seen on domains like {', '.join(top_refs)}."
            if top_refs
            else "Evidence was captured from returned third-party results."
        )
        comp = f"Other visible alternatives include {', '.join(comp_names)}." if comp_names else ""
        return " ".join([lead, src, comp]).strip()

    lead = f"For \"{question_text}\", your brand ({domain}) was not clearly verified in the top returned results."
    src = (
        f"The strongest visible evidence came from domains like {', '.join(top_refs)}."
        if top_refs
        else "No strong third-party domain evidence was captured for this query."
    )
    comp = f"Competing domains that surfaced include {', '.join(comp_names)}." if comp_names else ""
    return " ".join([lead, src, comp]).strip()


def _flatten_multi_result(r: dict) -> dict:
    """Helper to unify normal and multi-provider results into a single flat structure."""
    if not isinstance(r, dict):
        return {}
    if "providers" not in r:
        return r

    p = r["providers"]
    flat = {
        "status": "Not Mentioned",
        "position": None,
        "references": [],
        "sources": [],
        "source_urls": [],
        "idea_candidates": [],
        "competitors": [],
        "reasoning": "",
    }

    for prov in ["perplexity", "chatgpt", "gemini"]:
        d = p.get(prov, {})
        if isinstance(d, dict) and d.get("status") == "Mentioned":
            flat["status"] = "Mentioned"
            pos = d.get("position")
            if isinstance(pos, int):
                if flat["position"] is None or pos < flat["position"]:
                    flat["position"] = pos

    for prov, d in p.items():
        if not isinstance(d, dict):
            continue
        for key in ["references", "sources", "source_urls", "idea_candidates", "competitors"]:
            val = d.get(key, [])
            if isinstance(val, list):
                flat[key].extend(val)

    for key in ["references", "sources", "source_urls", "idea_candidates", "competitors"]:
        flat[key] = list(dict.fromkeys(flat[key]))

    return flat


def _estimate_target_visibility_score(seed_results: dict) -> int:
    rows = list(seed_results.values()) if isinstance(seed_results, dict) else []
    if not rows:
        return 45

    points = 0.0
    seen = 0
    for raw_r in rows:
        if not isinstance(raw_r, dict):
            continue
        r = _flatten_multi_result(raw_r)
        status = str(r.get("status", ""))
        pos = r.get("position")
        if isinstance(pos, int) and 1 <= pos <= 10:
            points += max(20, 110 - (pos * 10))
            seen += 1
        elif status == "Mentioned":
            points += 50
            seen += 1

    if seen == 0:
        return 35

    avg = points / seen
    coverage = seen / max(1, len(rows))
    score = int(round(avg * 0.7 + (coverage * 100) * 0.3))
    return max(20, min(95, score))


def _normalize_url(url: str) -> str:
    value = str(url or "").strip()
    if not value:
        return ""
    if not re.match(r"^https?://", value, re.I):
        value = f"https://{value}"
    return value


def _extract_brand_terms(url: str, ctx: dict) -> tuple[set[str], set[str], str]:
    domain = _normalize_domain(url)
    domain_tokens = [t for t in re.split(r"[.\-]", domain) if t]

    raw_name = str(ctx.get("name") or "").strip().lower()
    name_phrase = re.sub(r"\s+", " ", raw_name)
    name_tokens = re.findall(r"[a-z0-9']+", name_phrase)

    keep_words = {
        "restaurant", "restaurants", "london", "belgravia", "bar", "hotel", "cafe",
        "grill", "kitchen", "club", "store", "shop", "salon", "spa", "clinic",
    }
    blocked_tokens: set[str] = set()
    for token in [*domain_tokens, *name_tokens]:
        t = token.strip().lower()
        if len(t) < 4:
            continue
        if t in keep_words:
            continue
        blocked_tokens.add(t)

    blocked_phrases: set[str] = set()
    if name_phrase and len(name_phrase) >= 5:
        blocked_phrases.add(name_phrase)

    return blocked_tokens, blocked_phrases, domain


def _is_branded_question(text: str, blocked_tokens: set[str], blocked_phrases: set[str], domain: str) -> bool:
    raw = str(text or "").strip().lower()
    if not raw:
        return True

    if domain and domain in raw:
        return True
    if "http://" in raw or "https://" in raw or "www." in raw:
        return True

    for phrase in blocked_phrases:
        if phrase and phrase in raw:
            return True

    for token in blocked_tokens:
        if re.search(rf"\b{re.escape(token)}\b", raw):
            return True

    return False


def _is_low_quality_query(text: str) -> bool:
    raw = re.sub(r"\s+", " ", str(text or "")).strip().lower()
    if not raw:
        return True

    noisy_phrases = [
        "where to find us",
        "find us",
        "about us",
        "our story",
        "contact us",
        "get in touch",
        "book now",
        "learn more",
        "read more",
        "view more",
        "click here",
        "home page",
        "privacy policy",
        "terms of service",
    ]
    if any(p in raw for p in noisy_phrases):
        return True

    if re.search(r"\b(business|company|place)\b", raw):
        return True

    if "??" in raw:
        return True

    return False


def _pick_vertical_terms(url: str, ctx: dict) -> tuple[str, str]:
    domain = _normalize_domain(url)
    haystack = " ".join(
        [
            domain,
            str(ctx.get("category") or ""),
            str(ctx.get("name") or ""),
            str(ctx.get("description") or ""),
            " ".join([str(s) for s in (ctx.get("services") or [])]),
        ]
    ).lower()

    if any(k in haystack for k in ["restaurant", "dining", "menu", "bar", "food", "brunch", "lunch", "dinner"]):
        return "restaurant", "restaurants"
    if any(k in haystack for k in ["hotel", "resort", "stay", "accommodation"]):
        return "hotel", "hotels"
    if any(k in haystack for k in ["clinic", "dental", "doctor", "medical", "health"]):
        return "clinic", "clinics"
    if any(k in haystack for k in ["law", "attorney", "legal"]):
        return "law firm", "law firms"
    if any(k in haystack for k in ["agency", "studio", "portfolio", "designer", "developer"]):
        return "agency", "agencies"
    if any(k in haystack for k in ["salon", "spa", "beauty", "barber"]):
        return "salon", "salons"
    if any(k in haystack for k in ["shop", "store", "retail", "boutique"]):
        return "store", "stores"

    return "service", "services"


def _clean_service_hint(raw_services: list) -> str:
    for item in raw_services:
        text = re.sub(r"\s+", " ", str(item or "")).strip()
        if not text:
            continue
        if "?" in text:
            continue
        if len(text) < 4 or len(text) > 45:
            continue
        lowered = text.lower()
        if any(token in lowered for token in ["book now", "learn more", "contact", "party", "event"]):
            continue
        return text
    return ""


def _build_non_branded_fallback_questions(url: str, ctx: dict) -> list[str]:
    singular, plural = _pick_vertical_terms(url, ctx)
    location = str(ctx.get("location") or "").strip()
    services = [str(s).strip() for s in (ctx.get("services") or []) if str(s).strip()]
    service_hint = _clean_service_hint(services)
    service_phrase = service_hint if service_hint else singular

    where = f" in {location}" if location else " nearby"

    templates = [
        f"Best {plural}{where}?",
        f"Top-rated {plural}{' in ' + location if location else ''} with great value?",
        f"Best newly opened {plural}{' in ' + location if location else ''} this year?",
        f"Upscale {plural}{where}?",
        f"Which {plural} have the best customer reviews{' in ' + location if location else ''}?",
        f"Where can I find standout {service_phrase}{' in ' + location if location else ''}?",
        f"Best {plural} for special occasions{' in ' + location if location else ''}?",
        f"Top {plural} known for service quality{' in ' + location if location else ''}?",
        f"Most recommended {plural}{' in ' + location if location else ''} right now?",
        f"Which {plural} offer the best overall experience{' in ' + location if location else ''}?",
        f"Great {plural} for date night{' in ' + location if location else ''}?",
        f"Best {plural} for business dinners{' in ' + location if location else ''}?",
        f"Where can I book a premium {singular} experience{' in ' + location if location else ''}?",
        f"Top hidden-gem {plural}{' in ' + location if location else ''}?",
        f"Best value-for-money {plural}{' in ' + location if location else ''}?",
        f"What are the most talked-about {plural}{' in ' + location if location else ''}?",
        f"Which {plural} are trending this month{' in ' + location if location else ''}?",
        f"Best {plural} with a strong menu selection{' in ' + location if location else ''}?",
        f"Top choices for high-quality {service_phrase}{' in ' + location if location else ''}?",
        f"Where should I go for reliable {plural}{' in ' + location if location else ''}?",
        f"Best premium {plural} alternatives{' in ' + location if location else ''}?",
        f"Most consistent {plural} for quality and atmosphere{' in ' + location if location else ''}?",
    ]

    cleaned: list[str] = []
    seen: set[str] = set()
    for q in templates:
        text = re.sub(r"\s+", " ", q).strip()
        key = text.lower().rstrip("?.!")
        if not text or key in seen:
            continue
        seen.add(key)
        cleaned.append(text if text.endswith("?") else f"{text}?")
    return cleaned


def _extract_text_value(node) -> str:
    if node is None:
        return ""
    if isinstance(node, str):
        return node.strip()
    if isinstance(node, list):
        for item in node:
            text = _extract_text_value(item)
            if text:
                return text
    if isinstance(node, dict):
        for key in ["name", "text", "value", "description"]:
            text = _extract_text_value(node.get(key))
            if text:
                return text
    return ""
