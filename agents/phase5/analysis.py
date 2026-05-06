import asyncio
import re
from google import genai
from google.genai import types
from utils.gemini_utils import generate_with_fallback

from .config import (
    MAX_RETRIES,
    OPENAI_PHASE5_MAX_RETRIES,
    OPENAI_PHASE5_TIMEOUT_SEC,
    PERPLEXITY_PHASE5_MAX_RETRIES,
    PERPLEXITY_PHASE5_TIMEOUT_SEC,
    PHASE5_ENABLE_GEMINI,
    PHASE5_FAST_MODE,
    PHASE5_MODEL_CALL_TIMEOUT_SEC,
    PHASE5_VALIDATE_COMPETITORS,
    NON_COMPETITOR_DOMAINS,
)
from .helpers import (
    _align_reasoning_with_status,
    _build_fallback_concise_answer,
    _extract_domains_from_text,
    _is_non_competitor_domain,
    _is_target_domain_match,
    _normalize_domain,
    _safe_json_parse,
)
from .providers import (
    Phase5RateLimitError,
    _call_gemini_with_retry,
    _call_openai_with_retry,
    _call_perplexity_with_retry,
    _extract_grounding_signals,
    get_client,
)


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


async def _analyze_single_question_openai(url: str, question: dict, include_competitors: bool = False) -> dict:
    domain_match = re.search(r"(?:https?://)?(?:www\.)?([^/]+)", url)
    raw_domain = domain_match.group(1).lower() if domain_match else url.lower()
    domain = _normalize_domain(raw_domain)

    prompt = f"""
    Analyze this user-intent query for brand visibility using your general web/search understanding.

    Query: '{question['text']}'
    Target domain: '{domain}'

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
      "concise_answer": "Natural user-facing answer to the query (90-180 words), mentioning top options and practical guidance."
    }}

    Rules:
    - Keep lists short and realistic.
    - If target appears in results, set target.mentioned=true and include a realistic position.
    - It is okay to include the target domain once in references when it appears in results.
    - Sources/references must be third-party domains only.
    - Do not include target domain in competitors.
    - position must be 1..10 or null.
    - concise_answer must sound like a real assistant reply to the query (no analytics wording, no "mentioned/not mentioned" wording).
    - concise_answer should be compact but useful: top picks, brief why, and one practical tip.
    - JSON only.
    """

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

    target_evidence = bool(
        target_seen_in_sources
        or target_seen_in_references
        or (text_has_domain and text_positive and not text_negative)
    )

    if not target_evidence:
        target_mentioned = False
        position = None
    elif not target_mentioned:
        target_mentioned = True

    status = "Mentioned" if (target_mentioned or position is not None) else "Not Mentioned"
    reasoning_raw = str(data.get("reasoning", "")).strip() if isinstance(data, dict) else ""
    reasoning = _align_reasoning_with_status(
        reasoning=reasoning_raw,
        status=status,
        domain=domain,
        position=position,
        source_count=len(clean_sources),
        reference_count=len(clean_references),
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
    }


async def _analyze_single_question_perplexity(url: str, question: dict, include_competitors: bool = False) -> dict:
    domain_match = re.search(r"(?:https?://)?(?:www\.)?([^/]+)", url)
    raw_domain = domain_match.group(1).lower() if domain_match else url.lower()
    domain = _normalize_domain(raw_domain)

    prompt = f"""
    Analyze this user-intent query for brand visibility using web-aware reasoning.

    Query: '{question['text']}'
    Target domain: '{domain}'

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
      "concise_answer": "Natural user-facing answer to the query (90-180 words), mentioning top options and practical guidance."
    }}

    Rules:
    - Keep lists short and realistic.
    - Never use the target domain as a source or reference.
    - Sources/references must be third-party domains only.
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

    target_evidence = bool(
        target_seen_in_sources
        or target_seen_in_references
        or (text_has_domain and text_positive and not text_negative)
    )

    if not target_evidence:
        target_mentioned = False
        position = None
    elif not target_mentioned:
        target_mentioned = True

    status = "Mentioned" if (target_mentioned or position is not None) else "Not Mentioned"
    reasoning_raw = str(data.get("reasoning", "")).strip() if isinstance(data, dict) else ""
    reasoning = _align_reasoning_with_status(
        reasoning=reasoning_raw,
        status=status,
        domain=domain,
        position=position,
        source_count=len(clean_sources),
        reference_count=len(clean_references),
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
    }


async def analyze_single_question(
    url: str,
    question: dict,
    model_provider: str = "perplexity",
    include_competitors: bool = False,
) -> dict:
    """
    Analyzes a single question using Gemini with Google Search Grounding.
    Returns status, position, sources, and competitors dynamically.
    """
    normalized_provider = str(model_provider or "perplexity").strip().lower()
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
        }
    if normalized_provider == "openai":
        return await _analyze_single_question_openai(
            url=url,
            question=question,
            include_competitors=include_competitors,
        )
    if normalized_provider == "perplexity":
        return await _analyze_single_question_perplexity(
            url=url,
            question=question,
            include_competitors=include_competitors,
        )

    client = get_client()

    domain_match = re.search(r"(?:https?://)?(?:www\.)?([^/]+)", url)
    raw_domain = domain_match.group(1).lower() if domain_match else url.lower()
    domain = _normalize_domain(raw_domain)

    def score_from_position(pos: int | None) -> int:
        if not pos or pos < 1:
            return 0
        return max(0, 110 - (pos * 10))

    async def _run_grounded_probe() -> dict:
        probe_prompt = f"Use live Google Search for this query and list top evidence domains only: '{question['text']}'."
        probe_response = await _call_gemini_with_retry(
            client,
            probe_prompt,
            config=types.GenerateContentConfig(
                tools=[{"google_search": {}}],
                temperature=0.0,
                top_p=0.95,
                max_output_tokens=450,
            ),
        )
        if probe_response is None:
            return {
                "references": [],
                "source_urls": [],
                "target_mentioned": False,
                "position": None,
                "response_text": "",
            }

        probe_refs, probe_urls, probe_mentioned, probe_pos = _extract_grounding_signals(
            probe_response,
            domain,
        )
        probe_text = (getattr(probe_response, "text", "") or "").strip()
        text_domains = _extract_domains_from_text(probe_text)
        for d in text_domains:
            if d not in probe_refs:
                probe_refs.append(d)
            if _is_target_domain_match(d, domain):
                probe_mentioned = True
                if probe_pos is None:
                    probe_pos = 8

        if not probe_mentioned and domain in probe_text.lower():
            probe_mentioned = True
            if probe_pos is None:
                probe_pos = 8

        return {
            "references": list(dict.fromkeys(probe_refs)),
            "source_urls": list(dict.fromkeys(probe_urls)),
            "target_mentioned": probe_mentioned,
            "position": probe_pos,
            "response_text": probe_text,
        }

    async def _run_target_verification() -> tuple[bool, int | None, list[str], str]:
        verify_prompt = f"""
        Use live Google Search for this exact query and check if the target domain appears in top results.
        Query: '{question['text']}'
        Target domain: '{domain}'

        Return strict JSON only:
        {{
          "mentioned": true or false,
          "position": <1-10 or null>,
          "sources": ["domain.com"]
        }}
        """
        verify_response = await _call_gemini_with_retry(
            client,
            verify_prompt,
            config=types.GenerateContentConfig(
                tools=[{"google_search": {}}],
                temperature=0.0,
                top_p=0.95,
                max_output_tokens=350,
            ),
        )
        if verify_response is None:
            return False, None, [], ""

        parsed = _safe_json_parse(verify_response.text or "")
        mentioned = bool(parsed.get("mentioned", False)) if isinstance(parsed, dict) else False
        raw_pos = parsed.get("position") if isinstance(parsed, dict) else None
        position_value = raw_pos if isinstance(raw_pos, int) and 1 <= raw_pos <= 10 else None
        raw_sources = parsed.get("sources", []) if isinstance(parsed, dict) else []
        verify_sources = []
        if isinstance(raw_sources, list):
            for item in raw_sources:
                d = _normalize_domain(item)
                if d:
                    verify_sources.append(d)

        grounded_refs, _, grounded_mentioned, grounded_pos = _extract_grounding_signals(
            verify_response,
            domain,
        )
        verify_sources.extend(grounded_refs)

        text = (verify_response.text or "").lower()
        if not mentioned and (domain in text or domain.split(".")[0] in text):
            mentioned = True
            if position_value is None:
                position_value = 8

        if grounded_mentioned:
            mentioned = True
            if position_value is None:
                position_value = grounded_pos

        return mentioned, position_value, list(dict.fromkeys(verify_sources)), (verify_response.text or "")

    async def _run_chat_style_verification() -> tuple[bool, int | None, list[str], str]:
        chat_prompt = f"""
        Query: '{question['text']}'
        Target domain: '{domain}'

        Return JSON only:
        {{
          "mentioned": true or false,
          "position": <1-10 or null>,
          "references": ["domain.com"],
          "reason": "short reason"
        }}

        Rules:
        - Use plain reasoning like Gemini chat style.
        - No markdown.
        - If uncertain, set mentioned=false.
        """
        chat_response = await _call_gemini_with_retry(
            client,
            chat_prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,
                top_p=0.95,
                max_output_tokens=300,
            ),
        )
        if chat_response is None:
            return False, None, [], ""

        parsed = _safe_json_parse(chat_response.text or "")
        mentioned = bool(parsed.get("mentioned", False)) if isinstance(parsed, dict) else False
        raw_pos = parsed.get("position") if isinstance(parsed, dict) else None
        position_value = raw_pos if isinstance(raw_pos, int) and 1 <= raw_pos <= 10 else None
        refs = parsed.get("references", []) if isinstance(parsed, dict) else []
        out_refs = []
        if isinstance(refs, list):
            for item in refs:
                d = _normalize_domain(item)
                if d:
                    out_refs.append(d)

        txt = (chat_response.text or "").lower()
        if not mentioned and (domain in txt or domain.split(".")[0] in txt):
            mentioned = True
            if position_value is None:
                position_value = 8

        for d in out_refs:
            if _is_target_domain_match(d, domain):
                mentioned = True
                if position_value is None:
                    position_value = 8
                break

        return mentioned, position_value, list(dict.fromkeys(out_refs)), (chat_response.text or "")

    if not include_competitors:
        prompt = f"""
        You are an expert brand visibility evaluator. Use live Google Search for this query:
        '{question['text']}'

        Target brand domain: '{domain}'

        Return strict JSON only:
        {{
            "target": {{
                "mentioned": true or false,
                "position": <1-10 or null>,
                "source_domains": ["<domain>"]
            }},
            "references": ["<domain1.com>", "<domain2.com>", "... up to 20"],
            "idea_candidates": ["<potential-competitor-domain.com>", "... up to 8"],
            "reasoning": "1 short sentence"
        }}

        Rules:
        - 'references' must contain real web domains from observed evidence.
        - 'idea_candidates' should contain possible direct competitor domains inferred from this query context.
        - Do not include target domain in idea_candidates.
        - Output raw JSON only. No markdown.
        """
    elif PHASE5_FAST_MODE:
        prompt = f"""
        Use live Google Search and return JSON only.
        Query: '{question['text']}'
        Target domain: '{domain}'
        {{
            "target": {{"mentioned": true or false, "position": <1-10 or null>, "source_domains": ["domain.com"]}},
            "references": ["domain1.com", "domain2.com"],
            "idea_candidates": ["competitor.com"],
            "ranked_competitors": [{{"domain": "competitor.com", "position": 1, "evidence": "short reason"}}],
            "reasoning": "short sentence"
        }}
        Rules: no markdown, no prose, max 5 competitors, do not include target in competitors.
        """
    else:
        prompt = f"""
        You are an expert brand visibility evaluator. Use live Google Search to answer this query:
        '{question['text']}'

        Target brand domain: '{domain}' (brand token: '{domain.split('.')[0]}').
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
            "reasoning": "1 short sentence"
        }}

        Rules:
        - 'references' must contain real web domains from observed evidence.
        - Do not include the target domain in ranked_competitors.
        - Keep ranked_competitors to max 5 entries.
        - Output raw JSON only. No markdown.
        """

    try:
        response = await _call_gemini_with_retry(
            client,
            prompt,
            config=types.GenerateContentConfig(
                tools=[{"google_search": {}}],
                temperature=0.0,
                top_p=0.95,
                max_output_tokens=700,
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
        if not target_mentioned and (domain in response_text or brand_token in response_text):
            target_mentioned = True
            if position is None:
                position = 8

        should_probe = (
            response is None
            or (not clean_references and not clean_sources and not target_mentioned and position is None)
            or (not isinstance(data, dict) or not data)
        )
        if should_probe:
            probe = await _run_grounded_probe()
            clean_references.extend(probe["references"])
            clean_sources.extend(probe["references"])
            source_urls.extend(probe["source_urls"])

            if probe["target_mentioned"]:
                target_mentioned = True
                if position is None:
                    position = probe["position"]

            probe_text = (probe.get("response_text") or "").lower()
            if not target_mentioned and (domain in probe_text or brand_token in probe_text):
                target_mentioned = True
                if position is None:
                    position = 8

        if not target_mentioned and position is None:
            verified, verified_position, verified_sources, _ = await _run_target_verification()
            if verified:
                target_mentioned = True
                position = verified_position if isinstance(verified_position, int) else 8
                clean_sources.extend(verified_sources)
                clean_references.extend(verified_sources)

        if not target_mentioned and position is None:
            chat_verified, chat_position, chat_sources, _ = await _run_chat_style_verification()
            if chat_verified:
                target_mentioned = True
                position = chat_position if isinstance(chat_position, int) else 8
                clean_sources.extend(chat_sources)
                clean_references.extend(chat_sources)

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
                f"response_none={response is None} probed={should_probe}"
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
            "llm_response": ((response.text or "")[:4000] if response is not None and getattr(response, "text", None) else None),
        }
        return result_payload

    except Phase5RateLimitError:
        raise
    except Exception as e:
        print(f"Error parsing single question {question['id']}: {e}")
        return {
            "id": question["id"],
            "status": "Not Mentioned",
            "position": None,
            "sources": [],
            "source_urls": [],
            "references": [],
            "competitors": [],
            "competitor_scores": [],
            "reasoning": None,
            "llm_response": None,
        }


async def _run_with_backoff(
    url: str,
    question: dict,
    model_provider: str = "perplexity",
    include_competitors: bool = False,
) -> dict:
    """Wrap analyze_single_question with exponential backoff for provider rate limits."""
    provider = str(model_provider or "perplexity").strip().lower()
    if provider == "gemini" and not PHASE5_ENABLE_GEMINI:
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
        }
    if provider == "openai":
        retries = max(1, OPENAI_PHASE5_MAX_RETRIES)
    elif provider == "perplexity":
        retries = max(1, PERPLEXITY_PHASE5_MAX_RETRIES)
    else:
        retries = max(1, MAX_RETRIES)
    for attempt in range(retries):
        try:
            return await analyze_single_question(
                url=url,
                question=question,
                model_provider=model_provider,
                include_competitors=include_competitors,
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


def _empty_provider_result(error: str | None = None) -> dict:
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
        "cited": False,
        "status": "Not Mentioned",
    }
    if error:
        payload["error"] = str(error)
    return payload


def _safe_provider_result(raw: dict | Exception | None) -> dict:
    if isinstance(raw, Exception):
        return _empty_provider_result(str(raw))
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
    mentioned = status == "Mentioned" or position is not None
    cited = bool(sources or references or source_urls)

    return {
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
        "cited": cited,
        "status": "Mentioned" if mentioned else "Not Mentioned",
    }


async def analyze_single_question_multi(
    url: str,
    question: dict,
    include_competitors: bool = False,
) -> dict:
    async def _call_provider(provider: str):
        return await _run_with_backoff(
            url=url,
            question=question,
            model_provider=provider,
            include_competitors=include_competitors,
        )

    gemini_call = _call_provider("gemini")
    perplexity_call = _call_provider("perplexity")
    openai_call = _call_provider("openai")

    gemini_raw, perplexity_raw, openai_raw = await asyncio.gather(
        gemini_call,
        perplexity_call,
        openai_call,
        return_exceptions=True,
    )

    providers = {
        "gemini": _safe_provider_result(gemini_raw),
        "perplexity": _safe_provider_result(perplexity_raw),
        "chatgpt": _safe_provider_result(openai_raw),
    }

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
