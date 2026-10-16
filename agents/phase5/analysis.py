import asyncio
import re
from google import genai
from google.genai import types
from utils.gemini_utils import generate_with_fallback

from .config import (
    ANTHROPIC_PHASE5_MAX_RETRIES,
    ANTHROPIC_PHASE5_TIMEOUT_SEC,
    MAX_RETRIES,
    OPENAI_PHASE5_MAX_RETRIES,
    OPENAI_PHASE5_TIMEOUT_SEC,
    PERPLEXITY_PHASE5_MAX_RETRIES,
    PERPLEXITY_PHASE5_TIMEOUT_SEC,
    PHASE5_ENABLE_GEMINI,
    PHASE5_FAST_MODE,
    PHASE5_MODEL_CALL_TIMEOUT_SEC,
    PHASE5_VOTE_SAMPLING,
    PHASE5_VALIDATE_COMPETITORS,
    NON_COMPETITOR_DOMAINS,
)
from .context import _fetch_page_context
from .helpers import (
    _align_reasoning_with_status,
    _why_for_verdict,
    _build_fallback_concise_answer,
    _extract_domains_from_text,
    _is_non_competitor_domain,
    _is_target_domain_match,
    _normalize_domain,
    _safe_json_parse,
)
from .providers import (
    Phase5RateLimitError,
    _call_claude_with_retry,
    _call_gemini_with_retry,
    _call_openai_with_retry,
    _call_perplexity_with_retry,
    _extract_grounding_signals,
    get_client,
)


def _target_site_context_block(url: str, ctx: dict | None) -> str:
    ctx = ctx or {}
    lines = [f"Target website: {url}"]
    if ctx.get("name"):
        lines.append(f"Name: {str(ctx.get('name'))[:120]}")
    if ctx.get("category"):
        lines.append(f"Category: {str(ctx.get('category'))[:80]}")
    if ctx.get("location"):
        lines.append(f"Location: {str(ctx.get('location'))[:80]}")
    if ctx.get("description"):
        lines.append(f"Description: {str(ctx.get('description'))[:400]}")
    services = ctx.get("services") or []
    if isinstance(services, list) and services:
        lines.append(f"Visible services/topics: {', '.join([str(s)[:80] for s in services[:10]])}")
    return "\n".join(lines)


def _target_site_relevance_fallback(
    url: str,
    question_text: str,
    ctx: dict | None,
    raw_check: dict | None = None,
) -> dict:
    domain_match = re.search(r"(?:https?://)?(?:www\.)?([^/]+)", url)
    domain = _normalize_domain(domain_match.group(1).lower() if domain_match else url.lower())
    ctx = ctx or {}
    raw_check = raw_check if isinstance(raw_check, dict) else {}

    raw_status = str(raw_check.get("status") or raw_check.get("relevance") or "").strip().lower()
    if raw_status in {"matched", "match", "relevant", "yes"}:
        status = "matched"
    elif raw_status in {"partial", "partially_relevant", "some"}:
        status = "partial"
    elif raw_status in {"no_match", "not_relevant", "none", "no"}:
        status = "no_match"
    else:
        content_parts = [
            str(ctx.get("name") or ""),
            str(ctx.get("category") or ""),
            str(ctx.get("location") or ""),
            str(ctx.get("description") or ""),
            " ".join([str(s) for s in (ctx.get("services") or []) if s]),
        ]
        haystack = re.sub(r"[^a-z0-9\s]", " ", " ".join(content_parts).lower())
        tokens = [
            t
            for t in re.findall(r"[a-z0-9]{4,}", question_text.lower())
            if t not in {"where", "what", "which", "with", "near", "best", "find", "does", "have", "that", "this", "from", "for"}
        ]
        matches = []
        for token in tokens:
            if token in haystack and token not in matches:
                matches.append(token)
        if len(matches) >= 3:
            status = "matched"
        elif matches:
            status = "partial"
        else:
            status = "no_match"

    matched_facts = raw_check.get("matched_facts") if isinstance(raw_check.get("matched_facts"), list) else []
    missing_facts = raw_check.get("missing_facts") if isinstance(raw_check.get("missing_facts"), list) else []
    summary = str(raw_check.get("summary") or "").strip()

    if not matched_facts:
        if ctx.get("category"):
            matched_facts.append(str(ctx.get("category"))[:120])
        if ctx.get("location"):
            matched_facts.append(str(ctx.get("location"))[:120])
        services = ctx.get("services") or []
        if isinstance(services, list):
            matched_facts.extend([str(s)[:120] for s in services[:3]])
    if not missing_facts and status != "matched":
        missing_facts = ["Clearer page content for this exact search intent"]
    if not summary:
        if status == "matched":
            summary = "The target site appears to contain content that supports this prompt."
        elif status == "partial":
            summary = "The target site has some related signals, but the intent is not fully covered."
        else:
            summary = "The target site does not show strong visible content for this prompt yet."

    return {
        "checked": True,
        "status": status,
        "source_domain": domain,
        "source_url": url,
        "matched_facts": [str(x).strip()[:180] for x in matched_facts if str(x).strip()][:5],
        "missing_facts": [str(x).strip()[:180] for x in missing_facts if str(x).strip()][:5],
        "summary": summary[:360],
    }


async def _validate_direct_competitors(
    client: genai.Client,
    query_text: str,
    target_domain: str,
    candidate_domains: list[str],
) -> list[dict]:
    if not candidate_domains:
        return []

    prompt = f"""
    You are a market intelligence analyst.
    Determine which candidate domains are DIRECT competitors to the target business for this exact search intent.

    Query: '{query_text}'
    Target domain: '{target_domain}'
    Candidate domains: {candidate_domains}

    Direct competitor rules:
    - Same primary business category/service as the target.
    - Similar customer intent fit for this query.
    - If query is local intent (near me/nearby/city/area), competitor should be in relevant nearby geography.
    - Reject marketplaces/OTAs/directories/review or media platforms, unless they are the same primary business type as target.
    - Never include the target domain.

    Return JSON only:
    {{
      "validated": [
        {{
          "domain": "example.com",
          "position": 1,
          "category_overlap": 0,
          "geo_overlap": 0,
          "confidence": 0,
          "reason": "short reason"
        }}
      ]
    }}

    Constraints:
    - Max 5 items.
    - position must be 1..10 or null.
    - overlap/confidence must be integers 0..100.
    - JSON only, no markdown.
    """

    try:
        response = await _call_gemini_with_retry(
            client,
            prompt,
            config=types.GenerateContentConfig(
                tools=[{"google_search": {}}],
                temperature=0.0,
                top_p=0.9,
                max_output_tokens=900,
            ),
        )
        if response is None:
            return []
        parsed = _safe_json_parse(response.text or "")
        validated = parsed.get("validated", []) if isinstance(parsed, dict) else []
        if not isinstance(validated, list):
            return []

        output: list[dict] = []
        seen = set()
        for item in validated:
            if not isinstance(item, dict):
                continue
            d = _normalize_domain(item.get("domain", ""))
            if (
                not d
                or d == target_domain
                or d in NON_COMPETITOR_DOMAINS
                or d in seen
            ):
                continue

            pos = item.get("position")
            if not isinstance(pos, int) or pos < 1 or pos > 10:
                pos = None

            def _clamp_int(v):
                return max(0, min(100, int(v))) if isinstance(v, int) else 0

            output.append(
                {
                    "domain": d,
                    "position": pos,
                    "category_overlap": _clamp_int(item.get("category_overlap")),
                    "geo_overlap": _clamp_int(item.get("geo_overlap")),
                    "confidence": _clamp_int(item.get("confidence")),
                    "reason": str(item.get("reason", "")).strip()[:180],
                }
            )
            seen.add(d)
            if len(output) >= 5:
                break

        return output
    except Exception:
        return []


_NEGATION = re.compile(
    r"\b(no|not|never|without|isn't|is not|aren't|doesn't|does not|didn't|unable|cannot|can't|absent|missing|lack|lacks)\b"
)


def _named_in_answer(answer_text: str, domain: str, business_name: str | None = None) -> tuple:
    """B1: is the business NAMED in the answer text? Returns (named, evidence).

    Matches the business name, the domain, or the domain's brand word as whole
    words, and ignores sentences that say it is absent ("no mention of X").
    `evidence` is the sentence that names it, for the Query modal."""
    text = str(answer_text or "")
    if not text.strip():
        return False, None
    names = []
    for raw in (business_name, _normalize_domain(domain), (_normalize_domain(domain).split(".")[0] if domain else "")):
        t = re.sub(r"\s+", " ", str(raw or "").strip().lower())
        if len(t) >= 4 and t not in names:
            names.append(t)
    if not names:
        return False, None
    pattern = re.compile(r"(?<![a-z0-9])(" + "|".join(re.escape(n) for n in names) + r")(?![a-z0-9])")
    for sentence in re.split(r"(?<=[.!?\n])\s+", text):
        low = sentence.lower()
        if pattern.search(low) and not _NEGATION.search(low):
            return True, sentence.strip()[:240]
    return False, None


async def _analyze_single_question_openai(
    url: str,
    question: dict,
    include_competitors: bool = False,
    provider_name: str = "openai",
    target_context: dict | None = None,
) -> dict:
    domain_match = re.search(r"(?:https?://)?(?:www\.)?([^/]+)", url)
    raw_domain = domain_match.group(1).lower() if domain_match else url.lower()
    domain = _normalize_domain(raw_domain)
    target_context_block = _target_site_context_block(url, target_context)

    prompt = f"""
    Analyze this user-intent query for brand visibility using your general web/search understanding.

    Query: '{question['text']}'
    Target domain: '{domain}'
    Target site context to check separately:
    {target_context_block}

    Return strict JSON only with this schema:
    {{
      "target": {{
        "mentioned": true or false,
        "position": <1-10 or null>,
        "source_domains": ["domain.com"]
      }},
      "references": ["domain1.com", "domain2.com"],
      "idea_candidates": ["competitor.com"],
      "ranked_competitors": [
        {{"domain": "competitor.com", "position": 1, "evidence": "short reason"}}
      ],
      "reasoning": "1 short sentence",
      "concise_answer": "Natural user-facing answer to the query (90-180 words), mentioning top options and practical guidance.",
      "target_site_check": {{
        "status": "matched | partial | no_match",
        "matched_facts": ["facts from target site context that relate to the query"],
        "missing_facts": ["important facts missing from target site context"],
        "summary": "short explanation of whether the target site supports this query"
      }}
    }}

    Rules:
    - Keep lists short and realistic.
    - target.mentioned must be true ONLY if the target appears in organic answer/search evidence, not because target site context was provided.
    - Sources/references must be third-party domains only.
    - Do not count the provided target site context as organic visibility.
    - Evaluate target_site_check only from the provided target site context.
    - Do not include target domain in competitors.
    - position must be 1..10 or null.
    - concise_answer must sound like a real assistant reply to the query (no analytics wording, no "mentioned/not mentioned" wording).
    - concise_answer should be compact but useful: top picks, brief why, and one practical tip.
    - JSON only.
    """

    if provider_name == "claude":
        data = await _call_claude_with_retry(
            prompt,
            retry_once=True,
            timeout_sec=ANTHROPIC_PHASE5_TIMEOUT_SEC,
        )
    else:
        data = await _call_openai_with_retry(
            prompt,
            retry_once=True,
            timeout_sec=OPENAI_PHASE5_TIMEOUT_SEC,
        )
    if not isinstance(data, dict):
        data = {}

    target = data.get("target", {}) if isinstance(data, dict) else {}
    target_mentioned = bool(target.get("mentioned", False)) if isinstance(target, dict) else False
    raw_position = target.get("position") if isinstance(target, dict) else None
    position = raw_position if isinstance(raw_position, int) and 1 <= raw_position <= 10 else None

    raw_sources = target.get("source_domains", []) if isinstance(target, dict) else []
    if not isinstance(raw_sources, list):
        raw_sources = []
    target_seen_in_sources = any(_is_target_domain_match(str(s), domain) for s in raw_sources)
    clean_sources = []
    for s in raw_sources:
        d = _normalize_domain(s)
        if d and d != domain and d not in NON_COMPETITOR_DOMAINS:
            clean_sources.append(d)

    raw_references = data.get("references", []) if isinstance(data, dict) else []
    if not isinstance(raw_references, list):
        raw_references = []
    target_seen_in_references = any(_is_target_domain_match(str(r), domain) for r in raw_references)
    clean_references = []
    for r in raw_references:
        d = _normalize_domain(r)
        if d and d != domain and d not in NON_COMPETITOR_DOMAINS:
            clean_references.append(d)

    meta_domains = data.get("_meta_source_domains", []) if isinstance(data, dict) else []
    if isinstance(meta_domains, list):
        for md in meta_domains:
            d = _normalize_domain(md)
            if d and d != domain and d not in NON_COMPETITOR_DOMAINS:
                clean_references.append(d)

    idea_candidates: list[str] = []
    raw_ideas = data.get("idea_candidates", []) if isinstance(data, dict) else []
    if isinstance(raw_ideas, list):
        for item in raw_ideas:
            d = _normalize_domain(item)
            if d and d != domain and d not in NON_COMPETITOR_DOMAINS:
                idea_candidates.append(d)

    competitor_scores = []
    competitors = []
    if include_competitors:
        raw_ranked = data.get("ranked_competitors", []) if isinstance(data, dict) else []
        if isinstance(raw_ranked, list):
            for item in raw_ranked[:5]:
                if not isinstance(item, dict):
                    continue
                c_domain = _normalize_domain(item.get("domain", ""))
                c_pos = item.get("position")
                if (
                    not c_domain
                    or c_domain == domain
                    or c_domain in NON_COMPETITOR_DOMAINS
                    or not isinstance(c_pos, int)
                    or c_pos < 1
                    or c_pos > 10
                ):
                    continue
                score = max(0, min(100, 110 - (c_pos * 10)))
                competitor_scores.append(
                    {
                        "domain": c_domain,
                        "position": c_pos,
                        "score": score,
                        "evidence": str(item.get("evidence", "")).strip()[:180],
                    }
                )
        competitors = [c["domain"] for c in competitor_scores]

    clean_sources = list(dict.fromkeys(clean_sources))
    clean_references = list(dict.fromkeys(clean_references))
    idea_candidates = list(dict.fromkeys(idea_candidates))[:12]

    brand_token = re.sub(r"[^a-z0-9]", "", (domain.split(".")[0] if domain else "").lower())
    if not target_mentioned and position is None and (target_seen_in_sources or target_seen_in_references):
        target_mentioned = True

    concise_answer = str(data.get("concise_answer", "") if isinstance(data, dict) else "").strip()
    response_text_raw = str(data.get("_meta_response_text", "") if isinstance(data, dict) else "").strip()
    response_text = response_text_raw.lower()

    def _is_negative_text_mention(text: str, token: str) -> bool:
        if not token:
            return False
        safe = re.escape(token)
        return bool(
            re.search(rf"(no|not|without)\W+(clear\W+)?mention(?:ed)?[^\n\r.]{{0,40}}{safe}", text)
            or re.search(rf"{safe}[^\n\r.]{{0,40}}(not|no)\W+mention(?:ed)?", text)
            or re.search(rf"{safe}[^\n\r.]{{0,60}}(absent|missing|not present|not listed|does not appear|isn't listed|not in top)", text)
            or re.search(rf"(absent|missing|not present|not listed|does not appear|isn't listed|not in top)[^\n\r.]{{0,60}}{safe}", text)
        )

    def _is_positive_text_mention(text: str, token: str) -> bool:
        if not token:
            return False
        safe = re.escape(token)
        return bool(
            re.search(rf"{safe}[^\n\r.]{{0,60}}(appears|listed|featured|included|shown|ranking|ranked|in top)", text)
            or re.search(rf"(appears|listed|featured|included|shown|ranking|ranked|in top)[^\n\r.]{{0,60}}{safe}", text)
        )

    token_domain = _normalize_domain(domain)
    text_has_domain = bool(token_domain and token_domain in response_text)
    text_has_brand = bool(brand_token and len(brand_token) >= 4 and brand_token in response_text)
    text_negative = _is_negative_text_mention(response_text, token_domain) or _is_negative_text_mention(response_text, brand_token)
    text_positive = _is_positive_text_mention(response_text, token_domain)

    # only the answer itself: the raw reply also holds JSON fields (domain,
    # reasoning, ...) that echo the target and would always "name" it
    named_in_answer, named_evidence = _named_in_answer(concise_answer or "", domain, (target_context or {}).get("name"))
    cited_as_source = bool(target_seen_in_sources or target_seen_in_references)
    # B1: a mention is being named in the answer text OR cited as a source
    target_evidence = bool(
        cited_as_source
        or named_in_answer
        or (text_has_domain and text_positive and not text_negative)
    )

    if not target_evidence:
        target_mentioned = False
        position = None
    elif not target_mentioned:
        target_mentioned = True

    status = "Mentioned" if (target_mentioned or position is not None) else "Not Mentioned"
    reasoning_raw = str(data.get("reasoning", "")).strip() if isinstance(data, dict) else ""
    reasoning = _why_for_verdict(
        status=status,
        domain=domain,
        named_in_answer=named_in_answer,
        named_evidence=named_evidence,
        cited_as_source=cited_as_source,
        position=position,
        source_count=len(clean_sources),
        reference_count=len(clean_references),
        model_reasoning=reasoning_raw,
    )
    answer_text = (
        concise_answer[:4000]
        if concise_answer
        else _build_fallback_concise_answer(
            question_text=str(question.get("text") or "").strip(),
            status=status,
            position=position,
            references=clean_references,
            sources=clean_sources,
            competitor_scores=competitor_scores,
            domain=domain,
        )[:4000]
    )
    target_site = _target_site_relevance_fallback(
        url=url,
        question_text=str(question.get("text") or ""),
        ctx=target_context,
        raw_check=data.get("target_site_check") if isinstance(data, dict) else None,
    )

    return {
        "id": question["id"],
        "status": status,
        "position": position,
        "sources": clean_sources[:30],
        "source_urls": [],
        "references": clean_references[:30],
        "idea_candidates": idea_candidates,
        "competitors": competitors[:5],
        "competitor_scores": competitor_scores[:5],
        "reasoning": reasoning or None,
        "llm_response": answer_text,
        "target_site": target_site,
        "named_in_answer": named_in_answer,
        "cited_as_source": cited_as_source,
        "named_evidence": named_evidence,
    }


async def _analyze_single_question_perplexity(
    url: str,
    question: dict,
    include_competitors: bool = False,
    target_context: dict | None = None,
) -> dict:
    domain_match = re.search(r"(?:https?://)?(?:www\.)?([^/]+)", url)
    raw_domain = domain_match.group(1).lower() if domain_match else url.lower()
    domain = _normalize_domain(raw_domain)
    target_context_block = _target_site_context_block(url, target_context)

    prompt = f"""
    Analyze this user-intent query for brand visibility using web-aware reasoning.

    Query: '{question['text']}'
    Target domain: '{domain}'
    Target site context to check separately:
    {target_context_block}

    Return strict JSON only with this schema:
    {{
      "target": {{
        "mentioned": true or false,
        "position": <1-10 or null>,
        "source_domains": ["domain.com"]
      }},
      "references": ["domain1.com", "domain2.com"],
      "idea_candidates": ["competitor.com"],
      "ranked_competitors": [
        {{"domain": "competitor.com", "position": 1, "evidence": "short reason"}}
      ],
      "reasoning": "1 short sentence",
      "concise_answer": "Natural user-facing answer to the query (90-180 words), mentioning top options and practical guidance.",
      "target_site_check": {{
        "status": "matched | partial | no_match",
        "matched_facts": ["facts from target site context that relate to the query"],
        "missing_facts": ["important facts missing from target site context"],
        "summary": "short explanation of whether the target site supports this query"
      }}
    }}

    Rules:
    - Keep lists short and realistic.
    - target.mentioned must be true ONLY if the target appears in organic answer/search evidence, not because target site context was provided.
    - Third-party sources/references must not include the target domain.
    - Do not count the provided target site context as organic visibility.
    - Evaluate target_site_check only from the provided target site context.
    - Do not include target domain in competitors.
    - position must be 1..10 or null.
    - concise_answer must sound like a real assistant reply to the query (no analytics wording, no "mentioned/not mentioned" wording).
    - concise_answer should be compact but useful: top picks, brief why, and one practical tip.
    - JSON only.
    """

    data = await _call_perplexity_with_retry(
        prompt,
        retry_once=True,
        timeout_sec=PERPLEXITY_PHASE5_TIMEOUT_SEC,
    )
    if not isinstance(data, dict):
        data = {}

    target = data.get("target", {}) if isinstance(data, dict) else {}
    target_mentioned = bool(target.get("mentioned", False)) if isinstance(target, dict) else False
    raw_position = target.get("position") if isinstance(target, dict) else None
    position = raw_position if isinstance(raw_position, int) and 1 <= raw_position <= 10 else None

    raw_sources = target.get("source_domains", []) if isinstance(target, dict) else []
    if not isinstance(raw_sources, list):
        raw_sources = []
    target_seen_in_sources = any(_is_target_domain_match(str(s), domain) for s in raw_sources)
    clean_sources = []
    for s in raw_sources:
        d = _normalize_domain(s)
        if d and d != domain and d not in NON_COMPETITOR_DOMAINS:
            clean_sources.append(d)

    raw_references = data.get("references", []) if isinstance(data, dict) else []
    if not isinstance(raw_references, list):
        raw_references = []
    target_seen_in_references = any(_is_target_domain_match(str(r), domain) for r in raw_references)
    clean_references = []
    for r in raw_references:
        d = _normalize_domain(r)
        if d and d != domain and d not in NON_COMPETITOR_DOMAINS:
            clean_references.append(d)

    idea_candidates: list[str] = []
    raw_ideas = data.get("idea_candidates", []) if isinstance(data, dict) else []
    if isinstance(raw_ideas, list):
        for item in raw_ideas:
            d = _normalize_domain(item)
            if d and d != domain and d not in NON_COMPETITOR_DOMAINS:
                idea_candidates.append(d)

    competitor_scores = []
    competitors = []
    if include_competitors:
        raw_ranked = data.get("ranked_competitors", []) if isinstance(data, dict) else []
        if isinstance(raw_ranked, list):
            for item in raw_ranked[:5]:
                if not isinstance(item, dict):
                    continue
                c_domain = _normalize_domain(item.get("domain", ""))
                c_pos = item.get("position")
                if (
                    not c_domain
                    or c_domain == domain
                    or c_domain in NON_COMPETITOR_DOMAINS
                    or not isinstance(c_pos, int)
                    or c_pos < 1
                    or c_pos > 10
                ):
                    continue
                score = max(0, min(100, 110 - (c_pos * 10)))
                competitor_scores.append(
                    {
                        "domain": c_domain,
                        "position": c_pos,
                        "score": score,
                        "evidence": str(item.get("evidence", "")).strip()[:180],
                    }
                )
        competitors = [c["domain"] for c in competitor_scores]

    clean_sources = list(dict.fromkeys(clean_sources))
    clean_references = list(dict.fromkeys(clean_references))
    idea_candidates = list(dict.fromkeys(idea_candidates))[:12]

    meta_sources = data.get("_meta_source_domains", []) if isinstance(data, dict) else []
    meta_source_urls = data.get("_meta_source_urls", []) if isinstance(data, dict) else []
    source_urls: list[str] = []
    if isinstance(meta_source_urls, list):
        for u in meta_source_urls:
            u_str = str(u or "").strip()
            if u_str.startswith("http://") or u_str.startswith("https://"):
                source_urls.append(u_str)
        source_urls = list(dict.fromkeys(source_urls))[:50]

    if isinstance(meta_sources, list):
        for s in meta_sources:
            d = _normalize_domain(str(s))
            if d and d != domain and d not in NON_COMPETITOR_DOMAINS:
                clean_sources.append(d)
        clean_sources = list(dict.fromkeys(clean_sources))

    concise_answer = str(data.get("concise_answer", "") if isinstance(data, dict) else "").strip()
    response_text_raw = str(data.get("_meta_response_text", "") if isinstance(data, dict) else "").strip()
    response_text = response_text_raw.lower()
    brand_token = re.sub(r"[^a-z0-9]", "", (domain.split(".")[0] if domain else "").lower())

    def _is_negative_text_mention(text: str, token: str) -> bool:
        if not token:
            return False
        safe = re.escape(token)
        return bool(
            re.search(rf"(no|not|without)\W+(clear\W+)?mention(?:ed)?[^\n\r.]{{0,40}}{safe}", text)
            or re.search(rf"{safe}[^\n\r.]{{0,40}}(not|no)\W+mention(?:ed)?", text)
            or re.search(rf"{safe}[^\n\r.]{{0,60}}(absent|missing|not present|not listed|does not appear|isn't listed|not in top)", text)
            or re.search(rf"(absent|missing|not present|not listed|does not appear|isn't listed|not in top)[^\n\r.]{{0,60}}{safe}", text)
        )

    def _is_positive_text_mention(text: str, token: str) -> bool:
        if not token:
            return False
        safe = re.escape(token)
        return bool(
            re.search(rf"{safe}[^\n\r.]{{0,60}}(appears|listed|featured|included|shown|ranking|ranked|in top)", text)
            or re.search(rf"(appears|listed|featured|included|shown|ranking|ranked|in top)[^\n\r.]{{0,60}}{safe}", text)
        )

    token_domain = _normalize_domain(domain)
    text_has_domain = bool(token_domain and token_domain in response_text)
    text_has_brand = bool(brand_token and len(brand_token) >= 4 and brand_token in response_text)
    text_negative = _is_negative_text_mention(response_text, token_domain) or _is_negative_text_mention(response_text, brand_token)
    text_positive = _is_positive_text_mention(response_text, token_domain)

    # only the answer itself: the raw reply also holds JSON fields (domain,
    # reasoning, ...) that echo the target and would always "name" it
    named_in_answer, named_evidence = _named_in_answer(concise_answer or "", domain, (target_context or {}).get("name"))
    cited_as_source = bool(target_seen_in_sources or target_seen_in_references)
    # B1: a mention is being named in the answer text OR cited as a source
    target_evidence = bool(
        cited_as_source
        or named_in_answer
        or (text_has_domain and text_positive and not text_negative)
    )

    if not target_evidence:
        target_mentioned = False
        position = None
    elif not target_mentioned:
        target_mentioned = True

    status = "Mentioned" if (target_mentioned or position is not None) else "Not Mentioned"
    reasoning_raw = str(data.get("reasoning", "")).strip() if isinstance(data, dict) else ""
    reasoning = _why_for_verdict(
        status=status,
        domain=domain,
        named_in_answer=named_in_answer,
        named_evidence=named_evidence,
        cited_as_source=cited_as_source,
        position=position,
        source_count=len(clean_sources),
        reference_count=len(clean_references),
        model_reasoning=reasoning_raw,
    )
    answer_text = (
        concise_answer[:4000]
        if concise_answer
        else _build_fallback_concise_answer(
            question_text=str(question.get("text") or "").strip(),
            status=status,
            position=position,
            references=clean_references,
            sources=clean_sources,
            competitor_scores=competitor_scores,
            domain=domain,
        )[:4000]
    )
    target_site = _target_site_relevance_fallback(
        url=url,
        question_text=str(question.get("text") or ""),
        ctx=target_context,
        raw_check=data.get("target_site_check") if isinstance(data, dict) else None,
    )

    return {
        "id": question["id"],
        "status": status,
        "position": position,
        "sources": clean_sources[:30],
        "source_urls": source_urls,
        "references": clean_references[:30],
        "idea_candidates": idea_candidates,
        "competitors": competitors[:5],
        "competitor_scores": competitor_scores[:5],
        "reasoning": reasoning or None,
        "llm_response": answer_text,
        "target_site": target_site,
        "named_in_answer": named_in_answer,
        "cited_as_source": cited_as_source,
        "named_evidence": named_evidence,
    }


async def analyze_single_question(
    url: str,
    question: dict,
    model_provider: str = "perplexity",
    include_competitors: bool = False,
    target_context: dict | None = None,
) -> dict:
    """
    Analyzes a single question using Gemini with Google Search Grounding.
    Returns status, position, sources, and competitors dynamically.
    """
    normalized_provider = str(model_provider or "perplexity").strip().lower()
    if target_context is None:
        target_context = await _fetch_page_context(url)
    if normalized_provider == "gemini" and not PHASE5_ENABLE_GEMINI:
        return {
            "id": question["id"],
            "status": "Not Mentioned",
            "position": None,
            "sources": [],
            "source_urls": [],
            "references": [],
            "idea_candidates": [],
            "competitors": [],
            "competitor_scores": [],
            "reasoning": "Gemini is temporarily disabled for Phase 5.",
            "llm_response": None,
            "target_site": _target_site_relevance_fallback(
                url=url,
                question_text=str(question.get("text") or ""),
                ctx=target_context,
            ),
        }
    if normalized_provider == "openai":
        return await _analyze_single_question_openai(
            url=url,
            question=question,
            include_competitors=include_competitors,
            target_context=target_context,
        )
    if normalized_provider == "claude":
        return await _analyze_single_question_openai(
            url=url,
            question=question,
            include_competitors=include_competitors,
            provider_name="claude",
            target_context=target_context,
        )
    if normalized_provider == "perplexity":
        return await _analyze_single_question_perplexity(
            url=url,
            question=question,
            include_competitors=include_competitors,
            target_context=target_context,
        )

    client = get_client()

    domain_match = re.search(r"(?:https?://)?(?:www\.)?([^/]+)", url)
    raw_domain = domain_match.group(1).lower() if domain_match else url.lower()
    domain = _normalize_domain(raw_domain)
    target_context_block = _target_site_context_block(url, target_context)

    def score_from_position(pos: int | None) -> int:
        if not pos or pos < 1:
            return 0
        return max(0, 110 - (pos * 10))

    if not include_competitors:
        prompt = f"""
        You are an expert brand visibility evaluator. Use live Google Search for this query:
        '{question['text']}'

        Target brand domain: '{domain}'
        Target site context to check separately:
        {target_context_block}

        Return strict JSON only:
        {{
            "target": {{
                "mentioned": true or false,
                "position": <1-10 or null>,
                "source_domains": ["<domain>"]
            }},
            "references": ["<domain1.com>", "<domain2.com>", "... up to 20"],
            "idea_candidates": ["<potential-competitor-domain.com>", "... up to 8"],
            "reasoning": "1 short sentence",
            "concise_answer": "Natural user-facing answer to the query (90-180 words), mentioning top options and practical guidance.",
            "target_site_check": {{
                "status": "matched | partial | no_match",
                "matched_facts": ["facts from target site context that relate to the query"],
                "missing_facts": ["important facts missing from target site context"],
                "summary": "short explanation of whether the target site supports this query"
            }}
        }}

        Rules:
        - 'references' must contain real web domains from observed evidence.
        - 'idea_candidates' should contain possible direct competitor domains inferred from this query context.
        - target.mentioned must be true ONLY if the target appears in organic search evidence, not because target site context was provided.
        - Do not include target domain in idea_candidates.
        - Do not count the provided target site context as organic visibility.
        - Evaluate target_site_check only from the provided target site context.
        - Output raw JSON only. No markdown.
        - concise_answer must sound like a real assistant reply to the query (no analytics wording, no "mentioned/not mentioned" wording).
        - concise_answer should be compact but useful: top picks, brief why, and one practical tip.
        """
    elif PHASE5_FAST_MODE:
        prompt = f"""
        Use live Google Search and return JSON only.
        Query: '{question['text']}'
        Target domain: '{domain}'
        Target site context:
        {target_context_block}
        {{
            "target": {{"mentioned": true or false, "position": <1-10 or null>, "source_domains": ["domain.com"]}},
            "references": ["domain1.com", "domain2.com"],
            "idea_candidates": ["competitor.com"],
            "ranked_competitors": [{{"domain": "competitor.com", "position": 1, "evidence": "short reason"}}],
            "reasoning": "short sentence",
            "concise_answer": "Natural user-facing answer to the query (90-180 words), mentioning top options and practical guidance.",
            "target_site_check": {{"status": "matched | partial | no_match", "matched_facts": [], "missing_facts": [], "summary": "short"}}
        }}
        Rules: no markdown, no prose, max 5 competitors, do not include target in competitors.
        - target.mentioned must be true ONLY if the target appears in organic search evidence, not because target site context was provided.
        - Do not count the provided target site context as organic visibility.
        - Evaluate target_site_check only from the provided target site context.
        - concise_answer must sound like a real assistant reply to the query (no analytics wording, no "mentioned/not mentioned" wording).
        - concise_answer should be compact but useful: top picks, brief why, and one practical tip.
        """
    else:
        prompt = f"""
        You are an expert brand visibility evaluator. Use live Google Search to answer this query:
        '{question['text']}'

        Target brand domain: '{domain}' (brand token: '{domain.split('.')[0]}').
        Target site context to check separately:
        {target_context_block}
        Evaluate top search evidence and produce strict JSON only.

        JSON schema:
        {{
            "target": {{
                "mentioned": true or false,
                "position": <1-10 or null>,
                "source_domains": ["<domain>"]
            }},
            "references": ["<domain1.com>", "<domain2.com>", "<domain3.com>", "... up to 20"],
            "idea_candidates": ["<potential-competitor-domain.com>", "... up to 8"],
            "ranked_competitors": [
                {{"domain": "<competitor.com>", "position": <1-10>, "evidence": "short reason"}}
            ],
            "reasoning": "1 short sentence",
            "concise_answer": "Natural user-facing answer to the query (90-180 words), mentioning top options and practical guidance.",
            "target_site_check": {{
                "status": "matched | partial | no_match",
                "matched_facts": ["facts from target site context that relate to the query"],
                "missing_facts": ["important facts missing from target site context"],
                "summary": "short explanation of whether the target site supports this query"
            }}
        }}

        Rules:
        - 'references' must contain real web domains from observed evidence.
        - target.mentioned must be true ONLY if the target appears in organic search evidence, not because target site context was provided.
        - Do not include the target domain in ranked_competitors.
        - Do not count the provided target site context as organic visibility.
        - Evaluate target_site_check only from the provided target site context.
        - Keep ranked_competitors to max 5 entries.
        - Output raw JSON only. No markdown.
        - concise_answer must sound like a real assistant reply to the query (no analytics wording, no "mentioned/not mentioned" wording).
        - concise_answer should be compact but useful: top picks, brief why, and one practical tip.
        """

    try:
        response = await _call_gemini_with_retry(
            client,
            prompt,
            config=types.GenerateContentConfig(
                tools=[{"google_search": {}}],
                temperature=0.0,
                top_p=0.95,
                # 700 was silently truncating the JSON mid-object — Gemini's
                # "thinking" models spend part of this budget on hidden
                # reasoning before any visible text, and this prompt also
                # asks for a full concise_answer + ranked_competitors
                # payload. A truncated response fails _safe_json_parse and
                # falls through to generic fallback facts for every single
                # question, which is why target_site matched/missing facts
                # were identical across unrelated queries.
                max_output_tokens=4000,
            ),
        )
        data = _safe_json_parse((response.text or "") if response is not None else "")

        if not isinstance(data, dict):
            data = {}

        target = data.get("target", {}) if isinstance(data, dict) else {}
        target_mentioned = bool(target.get("mentioned", False)) if isinstance(target, dict) else False
        raw_position = target.get("position") if isinstance(target, dict) else None
        position = raw_position if isinstance(raw_position, int) and 1 <= raw_position <= 10 else None
        brand_parts = [p for p in domain.split(".") if p]
        brand_token = brand_parts[0] if brand_parts else domain

        raw_sources = target.get("source_domains", []) if isinstance(target, dict) else []
        if not isinstance(raw_sources, list):
            raw_sources = []

        clean_sources = []
        source_urls = []
        for s in raw_sources:
            s_str = _normalize_domain(s)
            if s_str and "vertexaisearch.cloud.google.com" not in s_str:
                clean_sources.append(s_str)

        raw_references = data.get("references", []) if isinstance(data, dict) else []
        if not isinstance(raw_references, list):
            raw_references = []
        clean_references = []
        for r in raw_references:
            r_str = _normalize_domain(r)
            if r_str and "vertexaisearch.cloud.google.com" not in r_str:
                clean_references.append(r_str)

        grounded_refs, grounded_urls, grounded_mentioned, grounded_pos = _extract_grounding_signals(
            response,
            domain,
        ) if response is not None else ([], [], False, None)
        clean_references.extend(grounded_refs)
        clean_sources.extend(grounded_refs)
        source_urls.extend(grounded_urls)
        if grounded_mentioned:
            target_mentioned = True
            if position is None:
                position = grounded_pos

        response_text = ((response.text or "") if response is not None else "").lower()
        gemini_answer = str(data.get("concise_answer", "") if isinstance(data, dict) else "").strip()
        named_in_answer, named_evidence = _named_in_answer(gemini_answer, domain, (target_context or {}).get("name"))
        cited_as_source = bool(grounded_mentioned or any(_is_target_domain_match(str(x), domain) for x in raw_sources + raw_references))
        # B1: counts only when the ANSWER names the business. (This used to
        # test the whole raw reply, whose JSON fields echo the target's own
        # domain - so Gemini "mentioned" the business on nearly every question.)
        if not target_mentioned and named_in_answer:
            target_mentioned = True
            if position is None:
                position = 8

        # Single-call answer only, matching the other three providers (each
        # makes exactly one API call per question) — no chained probe/verify/
        # chat-style fallback calls. Those made Gemini's own path 3-4x
        # sequential live-search round trips deep on "not mentioned"
        # questions, which is what made the whole 4-model batch bound on
        # Gemini's tail latency (see PHASE5_QUESTION_TIMEOUT_GEMINI_SEC=140s
        # vs ~40-45s for the others). Whatever the single main call above
        # found is now the final answer.

        clean_sources = list(dict.fromkeys(clean_sources))
        clean_references = list(dict.fromkeys(clean_references))
        source_urls = list(dict.fromkeys(source_urls))

        if not clean_sources and position is not None:
            clean_sources = [domain]

        raw_ideas = data.get("idea_candidates", []) if isinstance(data, dict) else []
        idea_candidates: list[str] = []
        if isinstance(raw_ideas, list):
            for item in raw_ideas:
                d = _normalize_domain(item)
                if d and d != domain and d not in NON_COMPETITOR_DOMAINS:
                    idea_candidates.append(d)
        for ref in clean_references:
            d = _normalize_domain(ref)
            if d and d != domain and d not in NON_COMPETITOR_DOMAINS:
                idea_candidates.append(d)
        idea_candidates = list(dict.fromkeys(idea_candidates))[:12]

        competitor_scores = []
        competitors = []
        if include_competitors:
            raw_ranked = data.get("ranked_competitors", []) if isinstance(data, dict) else []
            if not isinstance(raw_ranked, list):
                raw_ranked = []

            ranked_candidates = []
            for item in raw_ranked[:10]:
                if not isinstance(item, dict):
                    continue
                c_domain = _normalize_domain(item.get("domain", ""))
                c_pos = item.get("position")
                if (
                    not c_domain
                    or c_domain == domain
                    or c_domain in NON_COMPETITOR_DOMAINS
                    or not isinstance(c_pos, int)
                    or c_pos < 1
                    or c_pos > 10
                ):
                    continue
                ranked_candidates.append(
                    {
                        "domain": c_domain,
                        "position": c_pos,
                        "evidence": str(item.get("evidence", "")).strip()[:180],
                    }
                )

            candidate_domains = [c["domain"] for c in ranked_candidates]
            for ref in clean_references:
                if ref and ref != domain and ref not in NON_COMPETITOR_DOMAINS:
                    candidate_domains.append(ref)
            for src in clean_sources:
                if src and src != domain and src not in NON_COMPETITOR_DOMAINS:
                    candidate_domains.append(src)

            dedup_candidates = []
            seen_candidates = set()
            for cd in candidate_domains:
                if cd in seen_candidates:
                    continue
                seen_candidates.add(cd)
                dedup_candidates.append(cd)

            validated = []
            if PHASE5_VALIDATE_COMPETITORS:
                validated = await _validate_direct_competitors(
                    client=client,
                    query_text=question["text"],
                    target_domain=domain,
                    candidate_domains=dedup_candidates[:12],
                )

            if not validated:
                for idx, c in enumerate(ranked_candidates[:5]):
                    validated.append(
                        {
                            "domain": c["domain"],
                            "position": c["position"],
                            "category_overlap": 75,
                            "geo_overlap": 70,
                            "confidence": 75,
                            "reason": c.get("evidence") or "Detected in grounded ranked competitor results.",
                        }
                    )

            ranked_lookup = {c["domain"]: c for c in ranked_candidates}
            for idx, item in enumerate(validated[:5]):
                c_domain = item["domain"]
                c_position = item.get("position")
                raw_pos = ranked_lookup.get(c_domain, {}).get("position")
                position_value = c_position if isinstance(c_position, int) else raw_pos
                if not isinstance(position_value, int):
                    position_value = min(10, idx + 3)

                base_score = score_from_position(position_value)
                quality = round(
                    (
                        item.get("category_overlap", 0)
                        + item.get("geo_overlap", 0)
                        + item.get("confidence", 0)
                    ) / 3
                )
                blended_score = int(round((base_score * 0.7) + (quality * 0.3)))

                evidence = item.get("reason") or ranked_lookup.get(c_domain, {}).get("evidence") or "Validated as direct competitor for this query."
                competitor_scores.append(
                    {
                        "domain": c_domain,
                        "position": position_value,
                        "score": max(0, min(100, blended_score)),
                        "evidence": str(evidence)[:180],
                    }
                )

            competitors = [c["domain"] for c in competitor_scores]

        status = "Mentioned" if (target_mentioned or position is not None) else "Not Mentioned"
        reasoning = str(data.get("reasoning", "")).strip() if isinstance(data, dict) else ""
        if not reasoning and response is None:
            reasoning = "Primary structured request failed; fallback grounded probe used."

        if status == "Not Mentioned":
            print(
                f"[Phase5] Not Mentioned qid={question.get('id')} domain={domain} "
                f"refs={len(clean_references)} sources={len(clean_sources)} "
                f"response_none={response is None}"
            )

        concise_answer = str(data.get("concise_answer", "") if isinstance(data, dict) else "").strip()
        answer_text = (
            concise_answer[:4000]
            if concise_answer
            else _build_fallback_concise_answer(
                question_text=str(question.get("text") or "").strip(),
                status=status,
                position=position,
                references=clean_references,
                sources=clean_sources,
                competitor_scores=competitor_scores,
                domain=domain,
            )[:4000]
        )
        target_site = _target_site_relevance_fallback(
            url=url,
            question_text=str(question.get("text") or ""),
            ctx=target_context,
            raw_check=data.get("target_site_check") if isinstance(data, dict) else None,
        )

        result_payload = {
            "id": question["id"],
            "status": status,
            "position": position,
            "sources": clean_sources[:30],
            "source_urls": source_urls[:200],
            "references": clean_references[:30],
            "idea_candidates": idea_candidates,
            "competitors": competitors[:5],
            "competitor_scores": competitor_scores[:5],
            "reasoning": reasoning or None,
            "llm_response": answer_text,
            "target_site": target_site,
            "named_in_answer": named_in_answer,
            "cited_as_source": cited_as_source,
            "named_evidence": named_evidence,
        }
        return result_payload

    except Phase5RateLimitError:
        raise
    except Exception as e:
        print(f"Error parsing single question {question['id']}: {e}")
        return {
            "id": question["id"],
            "fetch_status": "failed",
            "error": f"Could not read the model's answer: {e}",
            "status": "Not Mentioned",
            "position": None,
            "sources": [],
            "source_urls": [],
            "references": [],
            "competitors": [],
            "competitor_scores": [],
            "reasoning": None,
            "llm_response": None,
            "target_site": _target_site_relevance_fallback(
                url=url,
                question_text=str(question.get("text") or ""),
                ctx=target_context,
            ),
        }


async def _run_with_backoff(
    url: str,
    question: dict,
    model_provider: str = "perplexity",
    include_competitors: bool = False,
    target_context: dict | None = None,
) -> dict:
    """Wrap analyze_single_question with exponential backoff for provider rate limits."""
    provider = str(model_provider or "perplexity").strip().lower()
    if target_context is None:
        target_context = await _fetch_page_context(url)
    if provider == "gemini" and not PHASE5_ENABLE_GEMINI:
        return {
            "id": question["id"],
            "fetch_status": "not_run",
            "status": "Not Mentioned",
            "position": None,
            "sources": [],
            "source_urls": [],
            "references": [],
            "idea_candidates": [],
            "competitors": [],
            "competitor_scores": [],
            "reasoning": "Gemini is temporarily disabled for Phase 5.",
            "llm_response": None,
            "target_site": _target_site_relevance_fallback(
                url=url,
                question_text=str(question.get("text") or ""),
                ctx=target_context,
            ),
        }
    if provider == "openai":
        retries = max(1, OPENAI_PHASE5_MAX_RETRIES)
    elif provider == "perplexity":
        retries = max(1, PERPLEXITY_PHASE5_MAX_RETRIES)
    elif provider == "claude":
        retries = max(1, ANTHROPIC_PHASE5_MAX_RETRIES)
    else:
        retries = max(1, MAX_RETRIES)
    for attempt in range(retries):
        try:
            return await analyze_single_question(
                url=url,
                question=question,
                model_provider=model_provider,
                include_competitors=include_competitors,
                target_context=target_context,
            )
        except Phase5RateLimitError:
            if attempt == retries - 1:
                raise
            wait = 2 ** (attempt + 1)
            print(
                f"[Phase5] rate limit for qid={question.get('id')} provider={provider} "
                f"attempt={attempt + 1}/{retries}; retrying in {wait}s"
            )
            await asyncio.sleep(wait)


def _empty_provider_result(error: str | None = None, detail: str | None = None) -> dict:
    payload = {
        "mentioned": False,
        "position": None,
        "sources": [],
        "source_urls": [],
        "references": [],
        "idea_candidates": [],
        "competitors": [],
        "competitor_scores": [],
        "reasoning": None,
        "llm_response": None,
        "target_site": None,
        "cited": False,
        "status": "Not Mentioned",
        # Both callers are failure paths (exception or non-dict result): this
        # is missing data, not a real "not mentioned" (A4).
        "fetch_status": "failed",
        "error": str(error) if error else "No usable response from model",
    }
    if detail:
        payload["error_detail"] = str(detail)[:300]      # what the provider really said
    return payload


def _safe_provider_result(raw: dict | Exception | None) -> dict:
    if isinstance(raw, Exception):
        return _empty_provider_result(str(raw), getattr(raw, "detail", None))
    if not isinstance(raw, dict):
        return _empty_provider_result()

    status = str(raw.get("status") or "Not Mentioned")
    position = raw.get("position") if isinstance(raw.get("position"), int) else None
    sources = raw.get("sources") if isinstance(raw.get("sources"), list) else []
    source_urls = raw.get("source_urls") if isinstance(raw.get("source_urls"), list) else []
    references = raw.get("references") if isinstance(raw.get("references"), list) else []
    idea_candidates = raw.get("idea_candidates") if isinstance(raw.get("idea_candidates"), list) else []
    competitors = raw.get("competitors") if isinstance(raw.get("competitors"), list) else []
    competitor_scores = raw.get("competitor_scores") if isinstance(raw.get("competitor_scores"), list) else []
    reasoning = raw.get("reasoning")
    llm_response = raw.get("llm_response")
    target_site = raw.get("target_site") if isinstance(raw.get("target_site"), dict) else None
    mentioned = status == "Mentioned" or position is not None
    cited = bool(sources or references or source_urls)
    raw_fetch_status = str(raw.get("fetch_status") or "ok").strip().lower()
    fetch_status = raw_fetch_status if raw_fetch_status in ("ok", "failed", "not_run") else "ok"

    result = {
        "fetch_status": fetch_status,
        "mentioned": bool(mentioned),
        "position": position,
        "sources": sources,
        "source_urls": source_urls,
        "references": references,
        "idea_candidates": idea_candidates,
        "competitors": competitors,
        "competitor_scores": competitor_scores,
        "reasoning": reasoning if isinstance(reasoning, str) and reasoning.strip() else None,
        "llm_response": llm_response if isinstance(llm_response, str) and llm_response.strip() else None,
        "target_site": target_site,
        "cited": cited,
        "status": "Mentioned" if mentioned else "Not Mentioned",
    }
    if fetch_status != "ok" and raw.get("error"):
        result["error"] = str(raw.get("error"))
    if fetch_status != "ok" and raw.get("error_detail"):
        result["error_detail"] = str(raw.get("error_detail"))[:300]
    # B2: the model answered but we captured no sources at all - a citation
    # capture failure (the search/grounding step returned nothing we could
    # read). Flagged so it is visible instead of silently scoring as "not cited".
    if fetch_status == "ok" and not (sources or references or source_urls):
        result["citations_missing"] = True
        excerpt = (result.get("llm_response") or "")[:600]
        result["raw_response_excerpt"] = excerpt
        print(f"[Phase5][CitationsMissing] answer had no sources/references. raw={excerpt!r}")
    # B1: why it counts as a mention - named in the answer, cited as a source, or both
    result["named_in_answer"] = bool(raw.get("named_in_answer"))
    result["cited_as_source"] = bool(raw.get("cited_as_source"))
    if raw.get("named_evidence"):
        result["named_evidence"] = str(raw.get("named_evidence"))[:240]
    # A5: how many of the repeated answers mentioned us - scoring and the
    # score margin both need it, so it must survive this normalisation.
    vote = raw.get("vote")
    if isinstance(vote, dict) and isinstance(vote.get("of"), int) and vote["of"] > 0:
        result["vote"] = {"mentioned": int(vote.get("mentioned") or 0), "of": vote["of"]}
    return result


def _sample_mentioned(result) -> bool:
    """Did this one answer name the business? Raw model answers carry
    `status` / `position`; `mentioned` is only derived later, in
    _safe_provider_result - so read whichever is there."""
    if not isinstance(result, dict):
        return False
    if "mentioned" in result:
        return bool(result["mentioned"])
    # same rule compute_mention_rate applies to a single answer (B1): named
    # in the answer, ranked, or cited as a source - all of which set status
    # "Mentioned". "Matched the site's facts" is relevance, not a mention.
    return str(result.get("status") or "") == "Mentioned" or isinstance(result.get("position"), int)


def _sample_ok(result) -> bool:
    return isinstance(result, dict) and str(result.get("fetch_status") or "ok") == "ok"


def pick_majority_sample(samples: list) -> dict | None:
    """A5: several answers to the same question from the same model -> the one
    to keep. The models search the live web, so the same question can be
    answered differently from one call to the next even at temperature 0;
    the majority view is what we score. Failed samples never vote. A tie
    (only possible with two voters) keeps the first answer."""
    ok = [r for r in samples if _sample_ok(r)]
    if not ok:
        return samples[0] if samples else None
    yes = [r for r in ok if _sample_mentioned(r)]
    no = [r for r in ok if not _sample_mentioned(r)]
    winners = yes if len(yes) > len(no) else no if len(no) > len(yes) else ok[:1]
    chosen = dict(winners[0])
    chosen["vote"] = {"mentioned": len(yes), "of": len(ok)}
    return chosen


async def analyze_single_question_multi(
    url: str,
    question: dict,
    include_competitors: bool = False,
    only_providers: set | None = None,
) -> dict:
    """Ask all four models (or just `only_providers`, named perplexity /
    chatgpt / claude / gemini - the rest come from a stored snapshot)."""
    target_context = await _fetch_page_context(url)

    async def _call_provider(provider: str):
        async def _once():
            return await _run_with_backoff(
                url=url,
                question=question,
                model_provider=provider,
                include_competitors=include_competitors,
                target_context=target_context,
            )

        # Retry each model once before it is recorded as failed (A4).
        try:
            result = await _once()
        except Exception:
            await asyncio.sleep(3)
            result = await _once()
        # A "swallowed" failure comes back as a dict marked failed rather
        # than an exception; give it the same single retry.
        if isinstance(result, dict) and str(result.get("fetch_status") or "") == "failed":
            await asyncio.sleep(3)
            result = await _once()
        if not PHASE5_VOTE_SAMPLING or not _sample_ok(result):
            return result

        # A5: ask again. Two answers that agree settle it; if they disagree a
        # third decides (majority of 3). Extra samples are best-effort - if
        # one fails, the answers we have still count.
        async def _extra():
            try:
                r = await _once()
                return r if _sample_ok(r) else None
            except Exception:
                return None

        # Branded questions ("What is <you>?") are the unstable ones - the
        # models flip on them from call to call - so they get up to 5 votes:
        # 3 up front, 2 more if those 3 are not unanimous. Everything else
        # gets 2, plus a tiebreaker if they disagree.
        branded = str(question.get("category") or question.get("type") or "").strip().lower() == "branded"
        samples = [result]
        for r in await asyncio.gather(*[_extra() for _ in range(2 if branded else 1)]):
            if r is not None:
                samples.append(r)
        if len({_sample_mentioned(r) for r in samples}) > 1:
            for r in await asyncio.gather(*[_extra() for _ in range(2 if branded else 1)]):
                if r is not None:
                    samples.append(r)
        return pick_majority_sample(samples)

    # result key -> name _run_with_backoff understands
    wanted = [(key, call) for key, call in (("perplexity", "perplexity"), ("chatgpt", "openai"), ("claude", "claude"), ("gemini", "gemini"))
              if only_providers is None or key in only_providers]
    raws = await asyncio.gather(*[_call_provider(call) for _, call in wanted], return_exceptions=True)
    providers = {key: _safe_provider_result(raw) for (key, _), raw in zip(wanted, raws)}

    return {
        "id": question.get("id"),
        "providers": providers,
    }


async def rank_brand_in_ai(url: str, questions: list) -> dict:
    """
    Uses Gemini model chain with Google Search Grounding enabled.
    Loops through questions, asks Gemini to answer them using live search data,
    and explicitly grades whether the target url or brand name appeared in the
    top 10 sources provided in the grounded response.
    Input: questions - list of dicts: [{'id': '...', 'text': '...'}]
    Returns: dict mapping question id to dict with status, position, sources.
    """
    client = get_client()
    results = {}

    domain_match = re.search(r"(?:https?://)?(?:www\.)?([^/]+)", url)
    domain = domain_match.group(1).lower() if domain_match else url.lower()

    async def _run_one(q: dict):
        try:
            prompt = f"Using Google Search, answer this question: '{q['text']}'. Provide a comprehensive answer with sources."

            response = await asyncio.wait_for(
                asyncio.to_thread(
                    generate_with_fallback,
                    client,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        tools=[{"google_search": {}}],
                        temperature=0.0,
                        max_output_tokens=2048,
                    ),
                ),
                timeout=PHASE5_MODEL_CALL_TIMEOUT_SEC,
            )

            sources = []
            mentioned = False
            position = None

            if response.candidates and response.candidates[0].grounding_metadata:
                metadata = response.candidates[0].grounding_metadata
                if hasattr(metadata, "grounding_chunks") and metadata.grounding_chunks:
                    for idx, chunk in enumerate(metadata.grounding_chunks):
                        if hasattr(chunk, "web") and chunk.web and hasattr(chunk.web, "uri"):
                            uri = chunk.web.uri
                            title = chunk.web.title if hasattr(chunk.web, "title") else uri
                            domain_only = re.search(r"(?:https?://)?(?:www\.)?([^/]+)", uri)
                            source_name = domain_only.group(1).capitalize() if domain_only else "Web"
                            if source_name not in sources:
                                sources.append(source_name)

                            if domain in uri.lower() or domain.split(".")[0] in title.lower():
                                mentioned = True
                                if position is None:
                                    position = idx + 1

            if not mentioned and domain.split(".")[0] in response.text.lower():
                mentioned = True
                if position is None:
                    position = 5

            status = "Mentioned" if mentioned else "Not Mentioned"

            results[q["id"]] = {
                "status": status,
                "position": position,
                "sources": sources[:3] if sources else ["Google"] if mentioned else [],
            }

        except Exception as e:
            print(f"Error processing question {q['id']}: {str(e)}")
            results[q["id"]] = {
                "status": "Not Mentioned",
                "position": None,
                "sources": [],
            }

        return None

    for i, q in enumerate(questions):
        await _run_one(q)
        if i < len(questions) - 1:
            await asyncio.sleep(1.0)
    return results
