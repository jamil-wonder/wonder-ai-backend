import asyncio
import json
import re

from .config import PHASE5_QUESTION_GEN_TIMEOUT_SEC
from .context import _fetch_page_context
from .helpers import (
    _extract_brand_terms,
    _is_branded_question,
    _is_low_quality_query,
    _normalize_domain,
    _safe_json_parse,
)
from .providers import _call_openai_chat_json, _call_perplexity_with_retry
from .question_profile import natural_terms, question_profile, uses_location, violates_profile


def _clean_question(value: str) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip())
    text = re.sub(r"^[\-\*\d\.\)\s]+", "", text).strip()
    if text and text[0].isalpha():
        text = text[0].upper() + text[1:]
    return text if not text or text.endswith("?") else f"{text}?"


def _text(value) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def _merge_context(page_context: dict, business_context: dict | None) -> dict:
    ctx = dict(page_context or {})
    supplied = business_context or {}

    field_map = {
        "name": ["name", "businessName", "business_name"],
        "category": ["category", "businessType", "business_type"],
        "location": ["location", "address", "city"],
        "description": ["description", "businessDescription", "aiDescription"],
    }
    for target, keys in field_map.items():
        for key in keys:
            value = _text(supplied.get(key))
            if value:
                ctx[target] = value
                break

    for target, keys in {"sub_type": ["sub_type", "subType"], "audience": ["target_audience", "targetAudience", "audience"]}.items():
        for key in keys:
            value = _text(supplied.get(key))
            if value:
                ctx[target] = value
                break

    services = supplied.get("services")
    if isinstance(services, list) and services:
        ctx["services"] = [_text(item) for item in services if _text(item)]

    return ctx


def _location_aliases(location: str) -> set[str]:
    raw = _text(location).lower()
    if not raw:
        return set()

    # The full location string and its comma/slash-separated parts are kept
    # regardless of length — a real place name that happens to be short
    # ("Rye", "UK", "NYC") must not be filtered out here, or every local SEO
    # question for that business silently fails its location check forever.
    aliases = {raw}
    parts = [p.strip() for p in re.split(r"[,/|-]", raw) if p.strip()]
    aliases.update(parts)
    # Extra single-word tokens pulled out of a longer phrase (so "London,
    # UK" also matches on just "London") do keep a length floor — those are
    # substrings extracted from something bigger, not the deliberate name
    # itself, so a short one here is more likely noise ("of", "in", "at").
    words = [w for w in re.findall(r"[a-z0-9]+", raw) if len(w) >= 4]
    aliases.update(words)
    return {alias for alias in aliases if alias}


def spread_local_over_locations(local_items: list[str], primary: str, extras: list[str]) -> list[str]:
    """G4: a business that tracks several locations gets its local questions split across them.
    Question i goes to location i of [primary, *extras] (round robin); a question generated for the
    primary location is re-pointed by swapping the place name, so every question still names exactly one place."""
    places = [p for p in [_text(primary), *[_text(e) for e in extras]] if p]
    if len(places) < 2 or not _text(primary):
        return list(local_items)
    out = []
    for i, text in enumerate(local_items):
        target = places[i % len(places)]
        if target.lower() == _text(primary).lower():
            out.append(text)
            continue
        swapped = re.sub(re.escape(_text(primary)), target, text, flags=re.I)
        if swapped == text:
            # the question named the place by a short alias (e.g. just the city): swap that instead
            for alias in sorted(_location_aliases(primary), key=len, reverse=True):
                swapped = re.sub(rf"\b{re.escape(alias)}\b", target, text, count=1, flags=re.I)
                if swapped != text:
                    break
        out.append(swapped)
    return out


def _includes_location(text: str, location: str) -> bool:
    raw = _text(text).lower()
    if not raw:
        return False
    # Word-boundary match, not a bare substring check — otherwise a short
    # real alias like "uk" would also match inside unrelated words like
    # "truck" or "look".
    return any(re.search(rf"\b{re.escape(alias)}\b", raw) for alias in _location_aliases(location))


def _includes_brand(text: str, brand_name: str) -> bool:
    raw = _text(text).lower()
    brand = _text(brand_name).lower()
    if not raw or not brand:
        return False
    return brand in raw


def _includes_category_or_service(text: str, category: str, services: list[str]) -> bool:
    raw = _text(text).lower()
    category_tokens = [
        token
        for token in re.findall(r"[a-z0-9]+", _text(category).lower())
        if len(token) >= 4 and token not in {"best", "near", "with"}
    ]
    service_tokens = [
        token
        for service in services
        for token in re.findall(r"[a-z0-9]+", service.lower())
        if len(token) >= 4
    ]
    terms = list(dict.fromkeys([*category_tokens, *service_tokens]))
    return any(term in raw for term in terms)


def _matches_best_in_location_pattern(text: str, category: str, location: str, services: list[str]) -> bool:
    """Detects the classic "Best <category> in <location>?" local-search
    shape (e.g. "Best restaurant in Belgravia?"). Structural, not just a
    keyword check — the category term must appear before " in " and the
    location must appear after it, so "Best in class {category}, {location}
    reviews?" doesn't false-positive just because both words are present
    somewhere in the sentence."""
    raw = _text(text).lower()
    if not raw.startswith("best "):
        return False
    if " in " not in raw:
        return False
    in_idx = raw.find(" in ")
    before_in = raw[:in_idx]
    after_in = raw[in_idx + 4:]
    return _includes_category_or_service(before_in, category, services) and _includes_location(after_in, location)


def _normalize_question_counts(question_counts: dict | None) -> dict[str, int]:
    source = question_counts if isinstance(question_counts, dict) else {}
    def _safe_count(key: str, default: int) -> int:
        try:
            return max(0, int(source.get(key, default) or 0))
        except Exception:
            return default
    counts = {
        "branded": _safe_count("branded", 5),
        "nonBranded": _safe_count("nonBranded", 5),
        "localSeo": _safe_count("localSeo", 5),
        "broadSeo": _safe_count("broadSeo", 5),
    }
    total = sum(counts.values())
    if total <= 0:
        return {"branded": 5, "nonBranded": 5, "localSeo": 5, "broadSeo": 5}
    if total > 20:
        overflow = total - 20
        for key in ["broadSeo", "localSeo", "nonBranded", "branded"]:
            remove = min(counts[key], overflow)
            counts[key] -= remove
            overflow -= remove
            if overflow <= 0:
                break
    elif total < 20:
        return {"branded": 5, "nonBranded": 5, "localSeo": 5, "broadSeo": 5}
    return counts


def _extract_query_groups(response: dict | None) -> tuple[list[str], list[str], list[str], list[str]]:
    if not isinstance(response, dict):
        return [], [], [], []

    parsed = response
    if not any(isinstance(parsed.get(key), list) for key in ["branded_queries", "non_branded_queries", "local_seo_queries", "broad_seo_queries", "queries"]):
        maybe = _safe_json_parse(str(parsed.get("_meta_response_text") or ""))
        if isinstance(maybe, dict):
            parsed = maybe

    branded = parsed.get("branded_queries")
    non_branded = parsed.get("non_branded_queries")
    local_seo = parsed.get("local_seo_queries") or parsed.get("local_queries")
    broad_seo = parsed.get("broad_seo_queries") or parsed.get("broad_queries")
    if isinstance(branded, list) or isinstance(non_branded, list) or isinstance(local_seo, list) or isinstance(broad_seo, list):
        return (
            [str(item) for item in branded or []],
            [str(item) for item in non_branded or []],
            [str(item) for item in local_seo or []],
            [str(item) for item in broad_seo or []],
        )

    queries = parsed.get("queries")
    if isinstance(queries, list):
        values = [str(item) for item in queries]
        return values[:5], [], values[5:], []

    response_text = str(response.get("_meta_response_text") or "")
    lines = [
        line.strip("- *1234567890. ")
        for line in response_text.split("\n")
        if len(line.strip()) > 10 and "?" in line
    ]
    return lines[:5], [], lines[5:], []


async def generate_preview_questions(
    url: str,
    business_context: dict | None = None,
) -> list[dict]:
    """One real, un-retried AI call producing up to 3 sample questions (one
    branded, one non-branded, one local-SEO) for the anonymous /scan "full
    results" preview. `generate_brand_questions` is deliberately NOT reused
    here — it's tuned for the authenticated locked-20 flow and retries up to
    6 times until it hits an exact 20-question quality bar, which is the
    right trade-off when the result is a durable weekly baseline but is a
    2-3 minute wait for a public preview that only ever samples 3 of them.
    A single honest pass is the correct trade for this surface: real
    content, just not exhaustively re-tried."""
    page_ctx = await _fetch_page_context(url)
    ctx = _merge_context(page_ctx, business_context)

    brand_name = _text(ctx.get("name"))
    category = _text(ctx.get("category"))
    location = _text(ctx.get("location"))
    description = _text(ctx.get("description"))
    services = ctx.get("services") if isinstance(ctx.get("services"), list) else []
    services = [_text(item) for item in services if _text(item)][:8]

    if not brand_name or not category:
        return []

    prompt = f"""
    Business name: {brand_name}
    Category: {category}
    Location: {location or "unknown"}
    Description: {description or "no description available"}
    Services: {", ".join(services) if services else "unknown"}

    Write 3 real questions a customer might ask an AI assistant (like ChatGPT or Perplexity) while
    looking for a business like this one:
    1. "branded" — mentions "{brand_name}" by name.
    2. "nonBranded" — a category/service search that does NOT name this business, e.g. "Which {category} is best for X?".
    3. "localSeo" — a "Best {category} in {location or '[location]'}?" style local-search question.

    Return JSON only, this exact shape:
    {{"branded": "...", "nonBranded": "...", "localSeo": "..."}}

    Rules:
    - Every question must be realistic, specific to this real business's category/location, and end with "?".
    - Ask about what customers of this kind of business really care about; never use venue wording (book, table, stay, visit, menu) unless it is a restaurant, hotel or similar venue.
    - Never mention the business name in nonBranded or localSeo.
    - JSON only, no markdown.
    """
    try:
        parsed = await _call_openai_chat_json(prompt, timeout_sec=PHASE5_QUESTION_GEN_TIMEOUT_SEC)
    except Exception as e:
        print(f"[Phase5] preview question-gen failed: {e}")
        return []

    if not isinstance(parsed, dict):
        return []

    out: list[dict] = []
    for key, type_key, label in [
        ("branded", "branded", "Branded"),
        ("nonBranded", "non-branded", "Non-Branded"),
        ("localSeo", "local-seo", "Local SEO"),
    ]:
        text = _clean_question(parsed.get(key))
        if text and not _is_low_quality_query(text) and not violates_profile(text, question_profile(category, ctx.get("sub_type"))):
            out.append({"text": text, "type": type_key, "label": label})
    return out


async def generate_brand_questions(
    url: str,
    business_context: dict | None = None,
    question_counts: dict | None = None,
    return_groups: bool = False,
) -> list[str] | dict[str, list[str]]:
    """
    Generate advanced Phase 5 questions using the saved business mix.
    """
    counts = _normalize_question_counts(question_counts)
    branded_target = counts["branded"]
    non_branded_target = counts["nonBranded"]
    local_seo_target = counts["localSeo"]
    broad_seo_target = counts["broadSeo"]
    candidate_counts = {
        "branded": branded_target + (3 if branded_target else 0),
        "nonBranded": non_branded_target + (3 if non_branded_target else 0),
        "localSeo": local_seo_target + (3 if local_seo_target else 0),
        "broadSeo": broad_seo_target + (3 if broad_seo_target else 0),
    }
    # Real user local-intent searches skew heavily toward the classic
    # "Best <category> in <location>?" shape (see _matches_best_in_location_
    # pattern below) — at 5 local SEO questions that's 2, at 10 it's 4. This
    # is a floor enforced by re-prompting the AI for real candidates that
    # match it (see the retry loop), never by splicing in a templated string.
    local_pattern_min_wanted = max(1, round(local_seo_target * 0.4)) if local_seo_target > 0 else 0
    page_ctx = await _fetch_page_context(url)
    ctx = _merge_context(page_ctx, business_context)

    brand_name = _text(ctx.get("name"))
    category = _text(ctx.get("category"))
    location = _text(ctx.get("location"))
    description = _text(ctx.get("description"))
    services = ctx.get("services") if isinstance(ctx.get("services"), list) else []
    services = [_text(item) for item in services if _text(item)][:10]
    sub_type = _text(ctx.get("sub_type"))
    audience = _text(ctx.get("audience"))
    # L2: what this kind of business's customers ask - and whether a place belongs in it
    profile = question_profile(category, sub_type)
    use_loc = uses_location(profile, location)
    # services plus everyday words for this kind of business, for the on-topic checks below
    match_services = services + natural_terms(profile, sub_type)
    avoid_questions = ctx.get("avoidQuestions") if isinstance(ctx.get("avoidQuestions"), list) else []
    avoid_questions = [_clean_question(item) for item in avoid_questions if _text(item)][:30]

    missing = []
    if not brand_name:
        missing.append("business name")
    if not category:
        missing.append("category")
    if not location and profile["location_mode"] == "required":
        missing.append("location")
    if missing:
        raise ValueError(
            "Question generation needs saved business context before it can create useful prompts. "
            f"Missing: {', '.join(missing)}."
        )

    local_pattern_min = local_pattern_min_wanted if use_loc else 0
    domain = _normalize_domain(url)
    if use_loc:
        location_line = f"Primary location to use in every local query: {location}"
    elif location:
        location_line = f"Based in: {location} (use it only if it fits a question naturally)"
    else:
        location_line = "Location: not a factor for this business - do not put a place name in the questions"
    context_block = "\n".join(
        [
            f"Business name: {brand_name}",
            f"Website domain: {domain}",
            f"Business category: {category}" + (f" (specifically: {sub_type})" if sub_type else ""),
            f"Kind of business: {profile['label']}",
            location_line,
            f"Who it serves: {audience}" if audience else "",
            f"Description: {description[:400]}" if description else "",
            f"Services / offers: {', '.join(services)}" if services else "",
        ]
    ).strip()
    intent_text = "; ".join(profile["intents"])
    place_bound = use_loc and profile["location_mode"] == "required"
    non_branded_place_rule = (
        f'Every question MUST name the city or area: "{location}" (someone choosing this kind of business is choosing it in a place).'
        if place_bound else "They do NOT have to include a place."
    )
    wording_rule = (
        ""
        if profile.get("hospitality_words_ok")
        else "- This is NOT a hospitality or venue business. Never use the words book, booking, reserve, reservation, table, stay, visit, check-in or menu, and never write about visiting, booking a table or staying at it.\n"
    )
    visitors = use_loc and profile["family"] == "hospitality" and re.search(r"hotel|hospitality|travel|tourism|lodging|resort|hostel|b&b|bed and breakfast|venue|events|entertainment", f"{category} {sub_type}", re.I)
    if use_loc:
        local_rules = f"""- Do NOT include "{brand_name}", "{domain}", the website URL, or any obvious brand variant.
- Every question MUST include the same business location: "{location}".
- Every question must be specific to this category: "{category}".
- Use service/details from the context when available.
- Cover these intents: {intent_text}.
- Do not use placeholder words such as "business", "company", or "place".
- Avoid repeated wording patterns.
- At least {local_pattern_min} of the {candidate_counts["localSeo"]} local_seo_queries MUST use the classic, high-volume real-search shape "Best <category> in <location>?" - questions shaped like "Best {category} in {location}?", using the exact location "{location}" every time and a natural real-world variant of the category or a specific service each time (do not repeat the identical sentence). Example shape only (do not reuse this exact wording): "{profile['example']}"
- The remaining local_seo_queries should use other natural local-intent phrasings rather than all following the same "Best X in Y" shape."""
        broad_rules = f"""- Do NOT include "{brand_name}", "{domain}", the website URL, or any obvious brand variant.
- Identify the actual real-world neighboring areas, neighborhoods, boroughs, or districts surrounding "{location}" and write questions using those real neighboring names, e.g. "Which [category] near [Neighboring Area] would people recommend?".
- The goal is to see if the AI engine recommends "{brand_name}" to people searching from these nearby surrounding areas.
- Do NOT use the exact string "{location}" in these questions. Instead, use the real surrounding areas.
- Every question must be specific to this category: "{category}" or the listed services.
- Avoid repeated wording patterns."""
        if visitors:
            # travellers do not live next door: they plan a trip to the place from somewhere else
            broad_rules = f"""- Do NOT include "{brand_name}", "{domain}", the website URL, or any obvious brand variant.
- These are written by VISITORS planning a trip to the area around "{location}" from somewhere else (another city or country): where to stay or what to book for a weekend, a first visit, a family trip, a special occasion, a business trip.
- Name the wider destination (the city or the well-known district of "{location}") - not nearby neighbourhoods.
- Every question must be specific to this category: "{category}" or the listed services.
- Avoid repeated wording patterns."""
    else:
        place_note = f" It is based in {location}, but a place name is not required." if location else ""
        local_rules = f"""- Do NOT include "{brand_name}", "{domain}", the website URL, or any obvious brand variant.
- These are specific, market-focused discovery questions: a particular need, audience, or use case this kind of business serves.{place_note}
- Every question must be specific to this category: "{category}" or the listed services.
- Cover these intents: {intent_text}.
- Do not use placeholder words such as "business", "company", or "place".
- Avoid repeated wording patterns. Example shape only (do not reuse this exact wording): "{profile['example']}" """
        broad_rules = f"""- Do NOT include "{brand_name}", "{domain}", the website URL, or any obvious brand variant.
- These are wider comparison and alternative-seeking questions: "alternatives to ...", "which option is best for [audience or use case]", "what is the difference between ...", for the services in this category.
- Do NOT put any place name in these questions.
- Every question must be specific to this category: "{category}" or the listed services.
- Avoid repeated wording patterns."""
    avoid_block = "\n".join([f"- {item}" for item in avoid_questions[:20]])

    base_prompt = f"""
You are a senior SEO search-intent strategist for AI answer engines.

Business context:
{context_block}

Generate a candidate pool of search questions for testing AI visibility.
The system will validate and keep the best {sum(counts.values())} questions.

Required structure:
- branded_queries: exactly {candidate_counts["branded"]} candidate questions.
- non_branded_queries: exactly {candidate_counts["nonBranded"]} candidate questions.
- local_seo_queries: exactly {candidate_counts["localSeo"]} candidate questions.
- broad_seo_queries: exactly {candidate_counts["broadSeo"]} candidate questions.

Important quality rules:
- Do not write thin generic prompts that could apply to any business.
- Do not write broken prompts such as "Best options for <business name>?".
- Do not write general advice prompts such as "What should I look for...", "How do I compare...", "How do I check...", or "What is the best way to...".
- Every question must carry a real buyer intent for THIS kind of business. The intents that matter here: {intent_text}.
{wording_rule}- Prefer natural full questions a real customer would type into Google, ChatGPT, Claude, or Perplexity.
- Use the services and the audience given above rather than generic wording.
- The category label above is only our own filing label. Write the way customers talk: use the everyday name of what the business is or offers (for example "solicitors", "dentists", "online store builder"), and never paste the category label itself into a question unless it is already a natural phrase.
- If previous questions are provided below, do not repeat them and do not create near-duplicates with only tiny wording changes.

Previous questions to avoid:
{avoid_block if avoid_block else "- None"}

Rules for branded_queries:
- Each question MUST include the exact business name: "{brand_name}".
- Each question must also use the real category{" or location" if use_loc else ""} context.
- Make them useful for checking what AI knows about the brand: {intent_text}.
- They must be SEO-friendly questions a real customer could ask.

Rules for non_branded_queries:
- Do NOT include "{brand_name}", "{domain}", the website URL, or any obvious brand variant.
- These are category/service discovery prompts. {non_branded_place_rule}
- Every question must be specific to this category: "{category}" or the listed services.
- Cover these intents: {intent_text}.
- Write prompts that could surface recommended providers, competitors, or ranked options.
- Prefer patterns like "Which [category]...", "What [category] options...", "Are there [category]...".
- Do NOT write advice/how-to prompts that only teach the user how to choose, compare, or check a provider.
- Avoid repeated wording patterns.

Rules for local_seo_queries:
{local_rules}

Rules for broad_seo_queries:
{broad_rules}

Return ONLY valid JSON:
{{
  "branded_queries": ["...", "... exactly {candidate_counts["branded"]}"],
  "non_branded_queries": ["...", "... exactly {candidate_counts["nonBranded"]}"],
  "local_seo_queries": ["...", "... exactly {candidate_counts["localSeo"]}"],
  "broad_seo_queries": ["...", "... exactly {candidate_counts["broadSeo"]}"]
}}
"""

    blocked_tokens, blocked_phrases, blocked_domain = _extract_brand_terms(url, ctx)
    blocked_tokens.add(_normalize_domain(brand_name))

    # More real attempts, not a synthetic fill-in — this only costs extra
    # time on the runs that are already retrying because a category came up
    # short; a run that succeeds on attempt 1 still returns on attempt 1.
    perplexity_attempts = 3
    openai_attempts = 3
    quality_attempts = perplexity_attempts + openai_attempts
    last_counts = {"branded": 0, "nonBranded": 0, "localSeo": 0, "broadSeo": 0}
    best_branded: list[str] = []
    best_non_branded: list[str] = []
    best_local_seo: list[str] = []
    best_broad_seo: list[str] = []

    def _format_result(
        branded_items: list[str],
        non_branded_items: list[str],
        local_seo_items: list[str],
        broad_seo_items: list[str],
    ) -> list[str] | dict[str, list[str]]:
        groups = {
            "branded": branded_items,
            "nonBranded": non_branded_items,
            "localSeo": local_seo_items,
            "broadSeo": broad_seo_items,
        }
        extras = business_context.get("extraLocations") if isinstance(business_context, dict) else None
        if use_loc and isinstance(extras, list) and extras:
            local_seo_items = spread_local_over_locations(local_seo_items, location, extras)
            groups["localSeo"] = local_seo_items
        flat = [*branded_items, *non_branded_items, *local_seo_items, *broad_seo_items]
        if return_groups:
            return {"questions": flat, "questionGroups": groups}
        return flat

    for attempt in range(1, quality_attempts + 1):
        provider_name = "perplexity" if attempt <= perplexity_attempts else "openai"
        prompt = base_prompt
        if attempt > 1:
            prompt += (
                f"\n\nThe previous output failed validation. Regenerate a larger candidate pool with exactly "
                f"{candidate_counts['branded']} branded, {candidate_counts['nonBranded']} non-branded, "
                f"{candidate_counts['localSeo']} local SEO, and {candidate_counts['broadSeo']} broad SEO candidates. "
                + (
                    f"Keep the exact location \"{location}\" in every local SEO question. "
                    f"Remember: at least {local_pattern_min} of the local_seo_queries MUST be the classic "
                    f"\"Best {category} in {location}?\" shape (real category/service wording, exact location) - "
                    f"this attempt was short on that pattern last time, so include more of them, not fewer."
                    if use_loc else
                    "Keep every question specific to this kind of business and its services, with no place names."
                )
            )

        if provider_name == "openai":
            prompt += (
                "\n\nUse OpenAI fallback mode: prioritize valid branded questions first. "
                f"Every branded question must contain this exact business name: {brand_name}. "
                "Return only the JSON object."
            )
            try:
                response = await _call_openai_chat_json(
                    prompt,
                    timeout_sec=PHASE5_QUESTION_GEN_TIMEOUT_SEC,
                )
            except Exception as e:
                print(f"[Phase5] OpenAI question-gen fallback unavailable: {type(e).__name__}: {e}")
                response = None
        else:
            response = await _call_perplexity_with_retry(
                prompt,
                retry_once=True,
                timeout_sec=PHASE5_QUESTION_GEN_TIMEOUT_SEC,
            )
        if response is None:
            await asyncio.sleep(0.3)
            continue

        raw_branded, raw_non_branded, raw_local_seo, raw_broad_seo = _extract_query_groups(response)

        branded: list[str] = []
        non_branded: list[str] = []
        local_seo: list[str] = []
        broad_seo: list[str] = []
        seen: set[str] = set()

        for item in raw_branded:
            text = _clean_question(item)
            key = text.lower().rstrip("?.!")
            if len(text) < 12 or key in seen:
                continue
            quality_text = re.sub(re.escape(brand_name), "", text, flags=re.IGNORECASE)
            if _is_low_quality_query(quality_text) or violates_profile(text, profile):
                continue
            if not _includes_brand(text, brand_name):
                continue
            if not (use_loc and _includes_location(text, location)) and not _includes_category_or_service(text, category, match_services):
                continue
            seen.add(key)
            branded.append(text)
            if len(branded) == branded_target:
                break

        for item in raw_non_branded:
            text = _clean_question(item)
            key = text.lower().rstrip("?.!")
            if len(text) < 12 or key in seen:
                continue
            if _is_low_quality_query(text) or violates_profile(text, profile):
                continue
            if _is_branded_question(text, blocked_tokens, blocked_phrases, blocked_domain):
                continue
            if not _includes_category_or_service(text, category, match_services):
                continue
            if place_bound and not _includes_location(text, location):
                continue
            seen.add(key)
            non_branded.append(text)
            if len(non_branded) == non_branded_target:
                break

        # Two passes so a pattern-matching candidate can never get crowded
        # out just because it happened to appear later in the AI's raw
        # output than local_seo_target other valid-but-non-pattern
        # candidates. Pass 1 fills the local_pattern_min quota from real
        # "Best <category> in <location>?" candidates only; pass 2 fills
        # whatever's left with any other valid local-intent candidate
        # (pattern-matching ones included, since they're valid too).
        valid_local_candidates: list[str] = []
        for item in raw_local_seo:
            text = _clean_question(item)
            key = text.lower().rstrip("?.!")
            if len(text) < 12 or key in seen:
                continue
            if _is_low_quality_query(text) or violates_profile(text, profile):
                continue
            if _is_branded_question(text, blocked_tokens, blocked_phrases, blocked_domain):
                continue
            if use_loc and not _includes_location(text, location):
                continue
            if not _includes_category_or_service(text, category, match_services):
                continue
            seen.add(key)
            valid_local_candidates.append(text)

        for text in valid_local_candidates:
            if len(local_seo) >= local_pattern_min:
                break
            if _matches_best_in_location_pattern(text, category, location, match_services):
                local_seo.append(text)

        for text in valid_local_candidates:
            if len(local_seo) >= local_seo_target:
                break
            if text not in local_seo:
                local_seo.append(text)

        local_pattern_count = sum(
            1 for text in local_seo if _matches_best_in_location_pattern(text, category, location, match_services)
        )

        for item in raw_broad_seo:
            text = _clean_question(item)
            key = text.lower().rstrip("?.!")
            if len(text) < 12 or key in seen:
                continue
            if _is_low_quality_query(text) or violates_profile(text, profile):
                continue
            if _is_branded_question(text, blocked_tokens, blocked_phrases, blocked_domain):
                continue
            if not _includes_category_or_service(text, category, match_services):
                continue
            seen.add(key)
            broad_seo.append(text)
            if len(broad_seo) == broad_seo_target:
                break

        last_counts = {"branded": len(branded), "nonBranded": len(non_branded), "localSeo": len(local_seo), "broadSeo": len(broad_seo)}
        if len(branded) > len(best_branded):
            best_branded = branded[:]
        if len(non_branded) > len(best_non_branded):
            best_non_branded = non_branded[:]
        # Prefer whichever attempt clears the pattern quota first, then
        # whichever simply has more items — a fuller list that's short on
        # the required "Best X in Y" shape isn't actually the better result.
        best_local_pattern_count = sum(
            1 for text in best_local_seo if _matches_best_in_location_pattern(text, category, location, match_services)
        )
        local_is_better = (
            min(local_pattern_count, local_pattern_min) > min(best_local_pattern_count, local_pattern_min)
            or (
                min(local_pattern_count, local_pattern_min) == min(best_local_pattern_count, local_pattern_min)
                and len(local_seo) > len(best_local_seo)
            )
        )
        if local_is_better:
            best_local_seo = local_seo[:]
        if len(broad_seo) > len(best_broad_seo):
            best_broad_seo = broad_seo[:]

        local_seo_ok = len(local_seo) == local_seo_target and local_pattern_count >= local_pattern_min
        if len(branded) == branded_target and len(non_branded) == non_branded_target and local_seo_ok and len(broad_seo) == broad_seo_target:
            return _format_result(branded, non_branded, local_seo, broad_seo)

        print(
            "[Phase5] advanced question-gen retry "
            f"provider={provider_name} attempt={attempt}/{quality_attempts} branded={len(branded)}/{branded_target} "
            f"non_branded={len(non_branded)}/{non_branded_target} "
            f"local_seo={len(local_seo)}/{local_seo_target} (pattern {local_pattern_count}/{local_pattern_min}) "
            f"broad_seo={len(broad_seo)}/{broad_seo_target}"
        )

    # No hardcoded/templated fallback questions, for any category — every
    # question a customer sees must come from the AI actually reasoning
    # about this business, never a fill-in-the-blank template standing in
    # for a real one. If validation kept rejecting the AI's candidates for
    # a category, the honest result is fewer real questions in that
    # category, not a synthetic one dressed up to look real.
    final_branded = best_branded[:branded_target]
    final_non_branded = best_non_branded[:non_branded_target]
    final_local_seo = best_local_seo[:local_seo_target]
    final_broad_seo = best_broad_seo[:broad_seo_target]

    total_generated = len(final_branded) + len(final_non_branded) + len(final_local_seo) + len(final_broad_seo)
    if total_generated > 0:
        final_local_pattern_count = sum(
            1 for text in final_local_seo if _matches_best_in_location_pattern(text, category, location, match_services)
        )
        print(
            "[Phase5] advanced question-gen completed with validated output "
            f"branded={len(final_branded)}/{branded_target} "
            f"non_branded={len(final_non_branded)}/{non_branded_target} "
            f"local_seo={len(final_local_seo)}/{local_seo_target} (pattern {final_local_pattern_count}/{local_pattern_min}) "
            f"broad_seo={len(final_broad_seo)}/{broad_seo_target}"
        )
        return _format_result(final_branded, final_non_branded, final_local_seo, final_broad_seo)

    raise ValueError(
        "Question generation failed validation. The AI provider did not return any usable prompts. "
        "Please improve the saved business name, category, and location, then retry."
    )


def accept_rewrite(original: str, rewritten: str, brand_name: str, location: str, branded: bool) -> str | None:
    """J8: a "more human" rewrite is only taken if it keeps what made the question valid:
    still a question, still names the brand (branded) or still does not (non-branded), and keeps the place if it had one."""
    if not str(rewritten or "").strip().endswith("?"):
        return None
    text = _clean_question(rewritten)
    if not text or len(text) > 220:
        return None
    has_brand = bool(brand_name) and brand_name.lower() in text.lower()
    if branded and brand_name and not has_brand:
        return None
    if not branded and has_brand:
        return None
    if location and _includes_location(original, location) and not _includes_location(text, location):
        return None
    return text


async def rewrite_question_human(
    question: str,
    brand_name: str = "",
    category: str = "",
    location: str = "",
    branded: bool = False,
) -> str | None:
    """J8: rewrite one tracked question the way a real customer would type it. None when the AI is unavailable
    or its rewrite breaks the rules above - the caller then keeps the original."""
    original = _text(question)
    if not original:
        return None
    prompt = f"""
    Rewrite this question so it sounds like a real person typing it into ChatGPT or Google - natural, casual, specific - without changing what is being asked.

    Question: "{original}"
    Business: {brand_name or "unknown"} ({category or "unknown category"}){f", based in {location}" if location else ""}

    Rules:
    - Keep it a single question that ends with "?".
    - {"Keep the business name " + repr(brand_name) + " in it." if branded and brand_name else "Do NOT mention the business name."}
    {f"- Keep the place name: {location}." if location and _includes_location(original, location) else ""}
    - No marketing words, no quotes, no list.
    Return JSON only: {{"question": "..."}}
    """
    try:
        parsed = await _call_openai_chat_json(prompt, timeout_sec=PHASE5_QUESTION_GEN_TIMEOUT_SEC)
    except Exception as e:
        print(f"[Phase5] question rewrite failed: {e}")
        return None
    if not isinstance(parsed, dict):
        return None
    return accept_rewrite(original, str(parsed.get("question") or ""), _text(brand_name), _text(location), branded)
