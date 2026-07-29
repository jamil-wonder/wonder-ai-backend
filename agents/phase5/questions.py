import asyncio
import json
import re

from .config import PERPLEXITY_PHASE5_TIMEOUT_SEC
from .context import _fetch_page_context
from .helpers import (
    _extract_brand_terms,
    _is_branded_question,
    _is_low_quality_query,
    _normalize_domain,
    _safe_json_parse,
)
from .providers import _call_openai_chat_json, _call_perplexity_with_retry


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

    services = supplied.get("services")
    if isinstance(services, list) and services:
        ctx["services"] = [_text(item) for item in services if _text(item)]

    return ctx


def _location_aliases(location: str) -> set[str]:
    raw = _text(location).lower()
    if not raw:
        return set()

    aliases = {raw}
    parts = [p.strip() for p in re.split(r"[,/|-]", raw) if p.strip()]
    aliases.update(parts)
    words = [w for w in re.findall(r"[a-z0-9]+", raw) if len(w) >= 4]
    aliases.update(words)
    return {alias for alias in aliases if len(alias) >= 4}


def _includes_location(text: str, location: str) -> bool:
    raw = _text(text).lower()
    if not raw:
        return False
    return any(alias in raw for alias in _location_aliases(location))


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


def _append_unique_valid_question(
    target: list[str],
    candidate: str,
    seen: set[str],
    *,
    validators: list,
    brand_name_for_quality: str = "",
) -> bool:
    text = _clean_question(candidate)
    key = text.lower().rstrip("?.!")
    if len(text) < 12 or key in seen:
        return False
    quality_text = text
    if brand_name_for_quality:
        quality_text = re.sub(re.escape(_text(brand_name_for_quality)), "", quality_text, flags=re.IGNORECASE)
    if _is_low_quality_query(quality_text):
        return False
    for validator in validators:
        if not validator(text):
            return False
    seen.add(key)
    target.append(text)
    return True


def _deterministic_branded_questions(
    *,
    brand_name: str,
    category: str,
    location: str,
    services: list[str],
) -> list[str]:
    service = services[0] if services else category
    return [
        f"Is {brand_name} a good {category} in {location}?",
        f"What is {brand_name} known for as a {category} in {location}?",
        f"Does {brand_name} have good reviews for {service} in {location}?",
        f"Is {brand_name} suitable for booking {service} in {location}?",
        f"How does {brand_name} compare with other {category} options in {location}?",
    ]


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
    page_ctx = await _fetch_page_context(url)
    ctx = _merge_context(page_ctx, business_context)

    brand_name = _text(ctx.get("name"))
    category = _text(ctx.get("category"))
    location = _text(ctx.get("location"))
    description = _text(ctx.get("description"))
    services = ctx.get("services") if isinstance(ctx.get("services"), list) else []
    services = [_text(item) for item in services if _text(item)][:8]
    avoid_questions = ctx.get("avoidQuestions") if isinstance(ctx.get("avoidQuestions"), list) else []
    avoid_questions = [_clean_question(item) for item in avoid_questions if _text(item)][:30]

    missing = []
    if not brand_name:
        missing.append("business name")
    if not category:
        missing.append("category")
    if not location:
        missing.append("location")
    if missing:
        raise ValueError(
            "Question generation needs saved business context before it can create useful prompts. "
            f"Missing: {', '.join(missing)}."
        )

    domain = _normalize_domain(url)
    context_block = "\n".join(
        [
            f"Business name: {brand_name}",
            f"Website domain: {domain}",
            f"Business category: {category}",
            f"Primary location to use in every local query: {location}",
            f"Description: {description[:260]}" if description else "",
            f"Services / offers: {', '.join(services)}" if services else "",
        ]
    ).strip()
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
- Do not write thin generic prompts such as "Best Hotel in City of London?".
- Do not write broken prompts such as "Best Hotel options for Angler?".
- Do not write general advice prompts such as "What should I look for...", "How do I compare...", "How do I check...", or "What is the best way to...".
- Every question must include a useful buyer intent such as reviews, booking, menu/service details, suitability, occasion, price/value, comparison, trust, amenities, or availability.
- Prefer natural full questions a real customer would type into Google, ChatGPT, Claude, or Perplexity.
- If previous questions are provided below, do not repeat them and do not create near-duplicates with only tiny wording changes.

Previous questions to avoid:
{avoid_block if avoid_block else "- None"}

Rules for branded_queries:
- Each question MUST include the exact business name: "{brand_name}".
- Each question must also use the real category or location context.
- Make them useful for checking brand understanding, reviews, menu/services, location, trust, booking, or suitability.
- They must be SEO-friendly questions a real customer could ask.

Rules for non_branded_queries:
- Do NOT include "{brand_name}", "{domain}", the website URL, or any obvious brand variant.
- These are category/service discovery prompts and do NOT have to include the location.
- Every question must be specific to this category: "{category}" or the listed services.
- Cover discovery, comparison, quality, reviews, booking/availability, price/value, occasion/use-case, and trust intent.
- Write prompts that could surface recommended providers, competitors, or ranked options.
- Prefer patterns like "Which [category]...", "What [category] options...", "Are there [category]...", "Where can I book...".
- Do NOT write advice/how-to prompts that only teach the user how to choose, compare, or check a provider.
- Avoid repeated wording patterns.

Rules for local_seo_queries:
- Do NOT include "{brand_name}", "{domain}", the website URL, or any obvious brand variant.
- Every question MUST include the same business location: "{location}".
- Every question must be specific to this category: "{category}".
- Use service/details from the context when available.
- Cover discovery, comparison, quality, reviews, booking/availability, price/value, occasion/use-case, and trust intent.
- Do not use placeholder words such as "business", "company", or "place".
- Do not produce generic prompts like "best restaurant near me" unless the exact location and category/service detail are included.
- Avoid repeated wording patterns.

Rules for broad_seo_queries:
- Do NOT include "{brand_name}", "{domain}", the website URL, or any obvious brand variant.
- Identify the actual real-world neighboring areas, neighborhoods, boroughs, or districts surrounding "{location}" (for example, if "{location}" is "City of London", surrounding areas could include "Shoreditch", "Southwark", "Holborn", "Clerkenwell", "Whitechapel", etc.).
- Write questions using these actual neighboring location names (e.g., "Do people in [Neighboring Area] recommend any good [Category]?" or "What are the best [Category] options around [Neighboring Area]?").
- The goal is to see if the AI engine recommends "{brand_name}" to people searching from these nearby surrounding areas.
- Do NOT use the exact string "{location}" in these questions. Instead, use the real surrounding areas.
- Every question must be specific to this category: "{category}" or the listed services.
- Avoid repeated wording patterns.

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

    perplexity_attempts = 2
    openai_attempts = 2
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
                "Keep the exact location in every local SEO question. Avoid short generic 'Best X in Y' prompts."
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
                    timeout_sec=max(PERPLEXITY_PHASE5_TIMEOUT_SEC, 30),
                )
            except Exception as e:
                print(f"[Phase5] OpenAI question-gen fallback unavailable: {type(e).__name__}: {e}")
                response = None
        else:
            response = await _call_perplexity_with_retry(
                prompt,
                retry_once=True,
                timeout_sec=PERPLEXITY_PHASE5_TIMEOUT_SEC,
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
            if _is_low_quality_query(quality_text):
                continue
            if not _includes_brand(text, brand_name):
                continue
            if not _includes_location(text, location) and not _includes_category_or_service(text, category, services):
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
            if _is_low_quality_query(text):
                continue
            if _is_branded_question(text, blocked_tokens, blocked_phrases, blocked_domain):
                continue
            if not _includes_category_or_service(text, category, services):
                continue
            seen.add(key)
            non_branded.append(text)
            if len(non_branded) == non_branded_target:
                break

        for item in raw_local_seo:
            text = _clean_question(item)
            key = text.lower().rstrip("?.!")
            if len(text) < 12 or key in seen:
                continue
            if _is_low_quality_query(text):
                continue
            if _is_branded_question(text, blocked_tokens, blocked_phrases, blocked_domain):
                continue
            if not _includes_location(text, location):
                continue
            if not _includes_category_or_service(text, category, services):
                continue
            seen.add(key)
            local_seo.append(text)
            if len(local_seo) == local_seo_target:
                break

        for item in raw_broad_seo:
            text = _clean_question(item)
            key = text.lower().rstrip("?.!")
            if len(text) < 12 or key in seen:
                continue
            if _is_low_quality_query(text):
                continue
            if _is_branded_question(text, blocked_tokens, blocked_phrases, blocked_domain):
                continue
            if not _includes_category_or_service(text, category, services):
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
        if len(local_seo) > len(best_local_seo):
            best_local_seo = local_seo[:]
        if len(broad_seo) > len(best_broad_seo):
            best_broad_seo = broad_seo[:]

        if len(branded) == branded_target and len(non_branded) == non_branded_target and len(local_seo) == local_seo_target and len(broad_seo) == broad_seo_target:
            return _format_result(branded, non_branded, local_seo, broad_seo)

        print(
            "[Phase5] advanced question-gen retry "
            f"provider={provider_name} attempt={attempt}/{quality_attempts} branded={len(branded)}/{branded_target} "
            f"non_branded={len(non_branded)}/{non_branded_target} local_seo={len(local_seo)}/{local_seo_target} "
            f"broad_seo={len(broad_seo)}/{broad_seo_target}"
        )

    seen = {q.lower().rstrip("?.!") for q in [*best_branded, *best_non_branded, *best_local_seo, *best_broad_seo]}
    final_branded = best_branded[:branded_target]
    final_non_branded = best_non_branded[:non_branded_target]
    final_local_seo = best_local_seo[:local_seo_target]
    final_broad_seo = best_broad_seo[:broad_seo_target]

    if len(final_branded) < branded_target:
        for candidate in _deterministic_branded_questions(
            brand_name=brand_name,
            category=category,
            location=location,
            services=services,
        ):
            _append_unique_valid_question(
                final_branded,
                candidate,
                seen,
                validators=[
                    lambda text: _includes_brand(text, brand_name),
                    lambda text: _includes_location(text, location)
                    or _includes_category_or_service(text, category, services),
                ],
                brand_name_for_quality=brand_name,
            )
            if len(final_branded) == branded_target:
                break

    total_generated = len(final_branded) + len(final_non_branded) + len(final_local_seo) + len(final_broad_seo)
    if total_generated > 0:
        print(
            "[Phase5] advanced question-gen completed with validated output "
            f"branded={len(final_branded)}/{branded_target} "
            f"non_branded={len(final_non_branded)}/{non_branded_target} "
            f"local_seo={len(final_local_seo)}/{local_seo_target} "
            f"broad_seo={len(final_broad_seo)}/{broad_seo_target}"
        )
        return _format_result(final_branded, final_non_branded, final_local_seo, final_broad_seo)

    raise ValueError(
        "Question generation failed validation. The AI provider did not return any usable prompts. "
        "Please improve the saved business name, category, and location, then retry."
    )
