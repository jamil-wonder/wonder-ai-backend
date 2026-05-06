import asyncio
import json
import re

from .config import PERPLEXITY_PHASE5_TIMEOUT_SEC
from .context import _fetch_page_context
from .helpers import (
    _extract_brand_terms,
    _is_branded_question,
    _is_low_quality_query,
    _safe_json_parse,
)
from .providers import _call_perplexity_with_retry


async def generate_brand_questions(url: str) -> list[str]:
    """
    Generate 20 realistic user-like search questions for a target domain.
    """
    ctx = await _fetch_page_context(url)
    lines = []
    if ctx.get("name"):
        lines.append(f"Business name: {ctx['name']}")
    if ctx.get("category"):
        lines.append(f"Business type: {ctx['category']}")
    if ctx.get("location"):
        lines.append(f"Location: {ctx['location']}")
    if ctx.get("description"):
        lines.append(f"Description: {str(ctx['description'])[:200]}")
    services = ctx.get("services") or []
    if isinstance(services, list) and services:
        lines.append(f"Services: {', '.join([str(s) for s in services[:8]])}")
    lines.append(f"Website: {url}")
    context_block = "\n".join(lines)

    base_prompt = f"""
    You are a senior search-intent strategist.
    Business context:
    {context_block}

    Generate exactly 20 natural search queries real customers would type.

    Requirements:
    - Cover intent diversity: discovery, comparison, trust, pricing, urgency, and location.
    - All 20 queries must be non-branded.
    - Do NOT include the business name, any brand name, website URL, or domain in any query.
    - Do NOT include direct entity-name variants, abbreviations, or possessive forms.
    - Include local intent queries when location context exists.
    - Keep queries conversational and realistic.
    - Ensure queries are specific to the business category and services in the context.
    - Avoid repetitive wording patterns.
    - Return broad, high-intent user wording only.
    - Never use placeholder words like "business", "company", or "place".
    - Never use website section labels like "Where to find us", "About us", "Contact us", or "Learn more".

    Return ONLY valid JSON with this schema:
    {{
      "queries": ["query1", "query2", ... exactly 20 items]
    }}
    """

    blocked_tokens, blocked_phrases, blocked_domain = _extract_brand_terms(url, ctx)

    def _extract_raw_questions(response: dict | None) -> list[str]:
        if not isinstance(response, dict):
            return []

        parsed = response
        if not isinstance(parsed.get("queries"), list):
            maybe = _safe_json_parse(str(parsed.get("_meta_response_text") or ""))
            if isinstance(maybe, dict):
                parsed = maybe

        structured: list[str] = []
        if isinstance(parsed, dict) and isinstance(parsed.get("queries"), list):
            structured = [str(x) for x in parsed.get("queries", [])]
        elif isinstance(parsed, list):
            structured = [str(x) for x in parsed]

        response_text = str(response.get("_meta_response_text") or "")
        prose_lines = [
            line.strip("- *1234567890. ")
            for line in response_text.split("\n")
            if len(line.strip()) > 10 and "?" in line
        ]

        return [*structured, *prose_lines]

    quality_attempts = 3
    best_cleaned: list[str] = []
    for attempt in range(1, quality_attempts + 1):
        prompt = base_prompt
        if attempt > 1:
            prompt += "\n\nPrevious output had invalid items. Regenerate exactly 20 better non-branded queries."

        response = await _call_perplexity_with_retry(
            prompt,
            retry_once=True,
            timeout_sec=PERPLEXITY_PHASE5_TIMEOUT_SEC,
        )
        if response is None:
            await asyncio.sleep(0.3)
            continue

        raw_questions = _extract_raw_questions(response)
        cleaned: list[str] = []
        seen = set()
        for q in raw_questions:
            text = re.sub(r"\s+", " ", str(q).strip())
            if text and text[0].isalpha():
                text = text[0].upper() + text[1:]
            key = text.lower().rstrip("?.!")
            if len(text) < 12 or key in seen:
                continue
            if _is_low_quality_query(text):
                continue
            if _is_branded_question(text, blocked_tokens, blocked_phrases, blocked_domain):
                continue
            seen.add(key)
            cleaned.append(text if text.endswith("?") else f"{text}?")

        if len(cleaned) >= 20:
            return cleaned[:20]

        if len(cleaned) > len(best_cleaned):
            best_cleaned = cleaned[:]

        print(f"[Phase5] question-gen retry attempt={attempt}/{quality_attempts} valid={len(cleaned)}")

    min_seed_for_expand = 10
    if len(best_cleaned) >= min_seed_for_expand:
        needed = max(0, 20 - len(best_cleaned))
        if needed > 0:
            expand_prompt = f"""
            You are a search-intent strategist.
            We already have valid non-branded queries for a business website.

            Existing valid queries:
            {json.dumps(best_cleaned)}

            Generate exactly {needed} NEW additional queries that satisfy ALL rules:
            - non-branded only
            - no business name, no URL/domain, no brand variants
            - no placeholder words like business/company/place
            - no website section labels (about us/contact us/where to find us/learn more)
            - realistic user intent wording
            - no duplicates or near-duplicates of existing queries

            Return JSON only:
            {{"queries": ["...", "..."]}}
            """

            extra_response = await _call_perplexity_with_retry(
                expand_prompt,
                retry_once=True,
                timeout_sec=PERPLEXITY_PHASE5_TIMEOUT_SEC,
            )

            extra_raw = []
            if isinstance(extra_response, dict):
                parsed_extra = extra_response
                if not isinstance(parsed_extra.get("queries"), list):
                    maybe = _safe_json_parse(str(parsed_extra.get("_meta_response_text") or ""))
                    if isinstance(maybe, dict):
                        parsed_extra = maybe
                if isinstance(parsed_extra, dict) and isinstance(parsed_extra.get("queries"), list):
                    extra_raw = [str(x) for x in parsed_extra.get("queries", [])]
                else:
                    extra_text = str(extra_response.get("_meta_response_text") or "")
                    extra_raw = [
                        line.strip("- *1234567890. ")
                        for line in extra_text.split("\n")
                        if len(line.strip()) > 10 and "?" in line
                    ]

            merged = best_cleaned[:]
            seen = {q.lower().rstrip("?.!") for q in merged}
            for q in extra_raw:
                text = re.sub(r"\s+", " ", str(q).strip())
                if text and text[0].isalpha():
                    text = text[0].upper() + text[1:]
                key = text.lower().rstrip("?.!")
                if len(text) < 12 or key in seen:
                    continue
                if _is_low_quality_query(text):
                    continue
                if _is_branded_question(text, blocked_tokens, blocked_phrases, blocked_domain):
                    continue
                seen.add(key)
                merged.append(text if text.endswith("?") else f"{text}?")
                if len(merged) >= 20:
                    break

            if len(merged) >= 20:
                return merged[:20]

            print(f"[Phase5] question-gen partial success valid={len(merged)} (returned without fallback)")
            return merged

        return best_cleaned[:20]

    raise ValueError("Question generation failed: AI did not return enough valid queries. Please retry.")
