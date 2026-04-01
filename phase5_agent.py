import os
import re
import json
import asyncio
from google import genai
from google.genai import types
from gemini_utils import generate_with_fallback


NON_COMPETITOR_DOMAINS = {
    "google.com",
    "bing.com",
    "youtube.com",
    "facebook.com",
    "instagram.com",
    "reddit.com",
    "wikipedia.org",
    "yelp.com",
    "tripadvisor.com",
    "opentable.com",
    "booking.com",
    "expedia.com",
    "kayak.com",
    "airbnb.com",
    "maps.google.com",
}

PHASE5_VALIDATE_COMPETITORS = False
PHASE5_FAST_MODE = False


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


def _extract_grounding_signals(response, target_domain: str) -> tuple[list[str], list[str], bool, int | None]:
    references: list[str] = []
    source_urls: list[str] = []
    target_mentioned = False
    position: int | None = None

    try:
        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            return references, source_urls, target_mentioned, position

        metadata = getattr(candidates[0], "grounding_metadata", None)
        chunks = getattr(metadata, "grounding_chunks", []) if metadata else []

        for idx, chunk in enumerate(chunks):
            web = getattr(chunk, "web", None)
            uri = getattr(web, "uri", "") if web else ""
            if not uri or "vertexaisearch.cloud.google.com" in uri:
                continue

            source_urls.append(uri)
            d = _normalize_domain(uri)
            if d:
                references.append(d)
                if _is_target_domain_match(d, target_domain):
                    target_mentioned = True
                    if position is None:
                        position = min(10, idx + 1)
    except Exception:
        pass

    return (
        list(dict.fromkeys(references)),
        list(dict.fromkeys(source_urls)),
        target_mentioned,
        position,
    )


async def _call_gemini_with_timeout(client: genai.Client, contents: str, config: types.GenerateContentConfig):
    try:
        return await asyncio.to_thread(
            generate_with_fallback,
            client,
            contents=contents,
            config=config,
        )
    except Exception as e:
        print(f"[Phase5][Gemini] call failed: {type(e).__name__}: {e}")
        return None


async def _call_gemini_with_retry(
    client: genai.Client,
    contents: str,
    config: types.GenerateContentConfig,
):
    response = await _call_gemini_with_timeout(client, contents, config)
    if response is not None:
        return response

    # One retry to reduce transient provider failures under load.
    try:
        return await asyncio.to_thread(
            generate_with_fallback,
            client,
            contents=contents,
            config=config,
        )
    except Exception as e:
        print(f"[Phase5][Gemini] retry failed: {type(e).__name__}: {e}")
        return None


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
    Candidate domains: {json.dumps(candidate_domains)}

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

def get_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")
    return genai.Client(api_key=api_key)

async def generate_brand_questions(url: str) -> list[str]:
    """
    Generate 20 realistic user-like search questions for a target domain.
    """
    client = get_client()
    prompt = f"""
    You are a senior search-intent strategist.
    Target website: '{url}'. Infer business type, customer intent, and local/non-local purchase journeys.

    Generate exactly 20 natural user search queries that real users would type.

    Requirements:
    - No generic boilerplate prompts.
    - No internal/technical jargon.
    - Include intent diversity: discovery, comparison, trust, pricing, service fit, urgency.
    - Mix short and long-tail phrasing.
    - Queries must be realistic and conversational.
    - Do not repeat the same wording pattern.
    - Avoid questions that directly include the exact domain unless natural.

    Return ONLY valid JSON with this schema:
    {{
      "queries": ["query1", "query2", ... exactly 20 items]
    }}
    """

    response = await _call_gemini_with_retry(
        client,
        prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.2,
            top_p=0.95,
            max_output_tokens=2500,
        ),
    )
    if response is None:
        return [
            "What are the best options in this area?",
            "Which place has the best reviews nearby?",
            "Where should I go for quality service in this area?",
            "What is the top-rated choice near me?",
            "Which business offers the best value here?",
            "Where can I find trusted local recommendations?",
            "What is the most popular option in this city?",
            "Which place is best for first-time visitors?",
            "What option is best for price and quality?",
            "Which local business is most recommended right now?",
        ]

    try:
        cleaned_text = response.text.strip()
        if cleaned_text.startswith("```json"):
            cleaned_text = cleaned_text[7:]
        if cleaned_text.endswith("```"):
            cleaned_text = cleaned_text[:-3]

        parsed = json.loads(cleaned_text.strip())
        if isinstance(parsed, dict) and isinstance(parsed.get("queries"), list):
            questions = parsed.get("queries", [])
        elif isinstance(parsed, list):
            questions = parsed
        else:
            questions = [str(parsed)]

        cleaned: list[str] = []
        seen = set()
        for q in questions:
            text = re.sub(r"\s+", " ", str(q).strip())
            if text and text[0].isalpha():
                text = text[0].upper() + text[1:]
            key = text.lower().rstrip("?.!")
            if len(text) < 12 or key in seen:
                continue
            seen.add(key)
            cleaned.append(text if text.endswith("?") else f"{text}?")

        return cleaned[:20]
    except Exception as e:
        print(f"Error parsing Gemini response: {e}, Response: {response.text}")
        q_list = [
            line.strip("- *1234567890. ")
            for line in response.text.split("\n")
            if len(line) > 10 and "?" in line
        ]
        return q_list[:20]


async def generate_brand_perception_summary(
    url: str,
    questions: list[dict],
    results: dict,
) -> str:
    """
    Generate a detailed business-style brand summary from completed Phase 5 analysis.
    """
    try:
        client = get_client()
        domain = _normalize_domain(url)

        compact_items = []
        for q in questions[:30]:
            qid = q.get("id")
            r = results.get(qid, {}) if isinstance(results, dict) else {}
            compact_items.append(
                {
                    "question": q.get("text", ""),
                    "status": r.get("status"),
                    "position": r.get("position"),
                    "references": (r.get("references") or [])[:5],
                    "competitors": (r.get("competitors") or [])[:5],
                    "reasoning": r.get("reasoning"),
                }
            )

        prompt = f"""
        You are a senior brand copywriter.
        Write ONE clear, human-friendly paragraph that describes what this business feels like to a normal customer.

        Target domain: {domain}
        Analysis data: {json.dumps(compact_items)}

        Requirements:
        - 95 to 140 words.
        - Plain English. Easy to understand. No technical language.
        - Natural narrative paragraph (not bullets).
        - Explain: what kind of business it appears to be, where it seems to operate, what style/tone it presents, and who it is likely for.
        - Mention trust/value signals in simple words (for example: clear menu, modern feel, professional tone, local relevance).
        - Make it sound like a concise profile someone can read in 10 seconds.
        - Do not fabricate exact facts not implied by data; if uncertain, use cautious wording like "appears to".
        - Do not mention internal fields, rankings, percentages, or prompt metrics.
        - Output plain text only.
        """

        response = await _call_gemini_with_retry(
            client,
            prompt,
            config=types.GenerateContentConfig(
                temperature=0.25,
                top_p=0.95,
                max_output_tokens=512,
            ),
        )
        if response is None:
            return f"{domain} appears to present a clear and focused business identity. The brand comes across as professional and customer-oriented, with messaging that suggests quality service and a defined audience. Its online presence feels credible and well-structured, and it appears positioned to attract people who are comparing options and looking for a reliable choice in its category."
        text = (response.text or "").strip()
        text = re.sub(r"\s+", " ", text)
        if not text:
            return f"{domain} appears to present a clear and focused business identity. The brand comes across as professional and customer-oriented, with messaging that suggests quality service and a defined audience. Its online presence feels credible and well-structured, and it appears positioned to attract people who are comparing options and looking for a reliable choice in its category."
        return text
    except Exception:
        domain = _normalize_domain(url)
        return f"{domain} appears to present a clear and focused business identity. The brand comes across as professional and customer-oriented, with messaging that suggests quality service and a defined audience. Its online presence feels credible and well-structured, and it appears positioned to attract people who are comparing options and looking for a reliable choice in its category."

async def analyze_single_question(url: str, question: dict) -> dict:
    """
    Analyzes a single question using Gemini with Google Search Grounding.
    Returns status, position, sources, and competitors dynamically.
    """
    client = get_client()

    domain_match = re.search(r'(?:https?://)?(?:www\.)?([^/]+)', url)
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

    # Create a prompt that requests structured ranking evidence
    prompt = ""
    if PHASE5_FAST_MODE:
        prompt = f"""
        Use live Google Search and return JSON only.
        Query: '{question['text']}'
        Target domain: '{domain}'
        {{
            "target": {{"mentioned": true or false, "position": <1-10 or null>, "source_domains": ["domain.com"]}},
            "references": ["domain1.com", "domain2.com"],
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

        # Grounding fallback: extract reference domains and raw URLs from metadata.
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

        # Text fallback: sometimes model returns valid prose with mention signals but no structured target block.
        response_text = ((response.text or "") if response is not None else "").lower()
        if not target_mentioned and (domain in response_text or brand_token in response_text):
            target_mentioned = True
            if position is None:
                position = 8

        # Reliability fallback: if structured parse failed or evidence is empty, run a second grounded probe
        # before deciding this is Not Mentioned.
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
        competitor_scores = []
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
            "competitors": competitors[:5],
            "competitor_scores": competitor_scores[:5],
            "reasoning": reasoning or None,
        }
        return result_payload

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
    
    # Simple extraction of domain to use as target
    domain_match = re.search(r'(?:https?://)?(?:www\.)?([^/]+)', url)
    domain = domain_match.group(1).lower() if domain_match else url.lower()
    
    for i, q in enumerate(questions):
        try:
            prompt = f"Using Google Search, answer this question: '{q['text']}'. Provide a comprehensive answer with sources."
            
            # Grounding configuration
            response = generate_with_fallback(
                client,
                contents=prompt,
                config=types.GenerateContentConfig(
                    tools=[{"google_search": {}}],
                    temperature=0.0,
                    max_output_tokens=2048,
                ),
            )
            
            # Extract sources if search grounding was used
            sources = []
            mentioned = False
            position = None
            
            # genai library structure for grounding:
            # response.candidates[0].grounding_metadata.grounding_chunks[i].web.uri / title
            if response.candidates and response.candidates[0].grounding_metadata:
                metadata = response.candidates[0].grounding_metadata
                if hasattr(metadata, "grounding_chunks") and metadata.grounding_chunks:
                    for idx, chunk in enumerate(metadata.grounding_chunks):
                        if hasattr(chunk, "web") and chunk.web and hasattr(chunk.web, "uri"):
                            uri = chunk.web.uri
                            title = chunk.web.title if hasattr(chunk.web, "title") else uri
                            domain_only = re.search(r'(?:https?://)?(?:www\.)?([^/]+)', uri)
                            source_name = domain_only.group(1).capitalize() if domain_only else "Web"
                            if source_name not in sources:
                                sources.append(source_name)
                            
                            # Check if the brand's domain is in the URI
                            if domain in uri.lower() or domain.split('.')[0] in title.lower():
                                mentioned = True
                                if position is None:
                                    position = idx + 1
            
            # Fallback text search if metadata chunks didn't yield sources but we still want to check response text
            if not mentioned and domain.split('.')[0] in response.text.lower():
                mentioned = True
                if position is None:
                    position = 5  # Estimated
            
            status = "Mentioned" if mentioned else "Not Mentioned"
            
            results[q['id']] = {
                "status": status,
                "position": position,
                "sources": sources[:3] if sources else ["Google"] if mentioned else []
            }
            
        except Exception as e:
            print(f"Error processing question {q['id']}: {str(e)}")
            results[q['id']] = {
                "status": "Not Mentioned",
                "position": None,
                "sources": []
            }
            
        # Small delay to avoid aggressive rate limiting
        if i < len(questions) - 1:
            await asyncio.sleep(1.0)
            
    return results
