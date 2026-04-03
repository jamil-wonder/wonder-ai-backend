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
PHASE5_MODEL_CALL_TIMEOUT_SEC = int(os.getenv("PHASE5_MODEL_CALL_TIMEOUT_SEC", "35"))


class Phase5RateLimitError(RuntimeError):
    pass


def _is_rate_limited_error(exc: Exception) -> bool:
    message = str(exc or "").lower()
    return (
        "429" in message
        or "resource_exhausted" in message
        or "rate limit" in message
        or "too many requests" in message
        or "quota" in message
    )


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


async def _call_gemini_with_timeout(
    client: genai.Client,
    contents: str,
    config: types.GenerateContentConfig,
    timeout_sec: int | None = None,
):
    effective_timeout = max(8, timeout_sec if isinstance(timeout_sec, int) else PHASE5_MODEL_CALL_TIMEOUT_SEC)
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(
                generate_with_fallback,
                client,
                contents=contents,
                config=config,
            ),
            timeout=effective_timeout,
        )
    except asyncio.TimeoutError:
        print(f"[Phase5][Gemini] call timed out after {effective_timeout}s")
        return None
    except Exception as e:
        if _is_rate_limited_error(e):
            raise Phase5RateLimitError("Gemini rate limit reached. Please retry shortly.") from e
        print(f"[Phase5][Gemini] call failed: {type(e).__name__}: {e}")
        return None


async def _call_gemini_with_retry(
    client: genai.Client,
    contents: str,
    config: types.GenerateContentConfig,
    retry_once: bool = True,
    timeout_sec: int | None = None,
):
    response = await _call_gemini_with_timeout(client, contents, config, timeout_sec=timeout_sec)
    if response is not None:
        return response

    if not retry_once:
        return None

    # One retry to reduce transient provider failures under load.
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(
                generate_with_fallback,
                client,
                contents=contents,
                config=config,
            ),
            timeout=max(8, timeout_sec if isinstance(timeout_sec, int) else PHASE5_MODEL_CALL_TIMEOUT_SEC),
        )
    except asyncio.TimeoutError:
        print(f"[Phase5][Gemini] retry timed out")
        return None
    except Exception as e:
        if _is_rate_limited_error(e):
            raise Phase5RateLimitError("Gemini rate limit reached. Please retry shortly.") from e
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
    base_prompt = f"""
    You are a senior search-intent strategist.
    Target website: '{url}'. Infer business type, core services, and typical customer intent.

    Generate exactly 20 natural user search queries that real users would type.

    Requirements:
    - Every query must be clearly about the target's business category and services inferred from the URL.
    - Each query should be plausible for a customer who might reasonably be shown the target brand.
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

    response = None
    for _ in range(3):
        response = await _call_gemini_with_retry(
            client,
            base_prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.2,
                top_p=0.95,
                max_output_tokens=2500,
            ),
        )
        if response is not None:
            break
        await asyncio.sleep(0.5)

    if response is None:
        raise RuntimeError("Failed to generate brand questions from Gemini. Please retry.")

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

        if len(cleaned) < 12:
            raise ValueError("Gemini returned too few valid, relevant queries.")

        return cleaned[:20]
    except Exception as e:
        print(f"Error parsing Gemini response: {e}, Response: {response.text}")
        q_list = [
            line.strip("- *1234567890. ")
            for line in response.text.split("\n")
            if len(line) > 10 and "?" in line
        ]
        if len(q_list) < 12:
            raise ValueError("Gemini response parsing failed; insufficient relevant queries.")
        return q_list[:20]


async def generate_brand_perception_summary(
    url: str,
    questions: list[dict],
    results: dict,
) -> str:
    """
    Generate a detailed business-style brand summary from completed Phase 5 analysis.
    """
    def _build_quick_summary() -> str:
        domain = _normalize_domain(url)
        rows = []
        for q in questions[:30]:
            qid = q.get("id")
            rows.append(results.get(qid, {}) if isinstance(results, dict) else {})

        total = len(rows) if rows else 1
        mentioned = 0
        positions: list[int] = []
        ref_counts: dict[str, int] = {}

        for r in rows:
            status = str(r.get("status", ""))
            if status == "Mentioned":
                mentioned += 1
            pos = r.get("position")
            if isinstance(pos, int) and 1 <= pos <= 10:
                positions.append(pos)

            refs = r.get("references", []) if isinstance(r, dict) else []
            srcs = r.get("sources", []) if isinstance(r, dict) else []
            urls = r.get("source_urls", []) if isinstance(r, dict) else []
            for item in list(refs) + list(srcs) + list(urls):
                d = _normalize_domain(item)
                if d and d != domain and d not in NON_COMPETITOR_DOMAINS:
                    ref_counts[d] = ref_counts.get(d, 0) + 1

        mention_rate = int(round((mentioned / max(1, total)) * 100))
        avg_rank = round(sum(positions) / len(positions), 1) if positions else None
        top_refs = [d for d, _ in sorted(ref_counts.items(), key=lambda kv: kv[1], reverse=True)[:2]]

        if mention_rate >= 65:
            visibility_line = f"{domain} shows strong visibility across customer search intent, with mention coverage around {mention_rate}%"
        elif mention_rate >= 35:
            visibility_line = f"{domain} shows moderate visibility across customer search intent, with mention coverage around {mention_rate}%"
        else:
            visibility_line = f"{domain} currently has limited visibility across customer search intent, with mention coverage around {mention_rate}%"

        rank_line = (
            f" and an average appearance near rank {avg_rank}."
            if avg_rank is not None
            else "."
        )

        if top_refs:
            ref_line = f" Frequent reference context came from {', '.join(top_refs)}, which indicates where comparison traffic is happening most."
        else:
            ref_line = " Reference spread is still narrow, so content clarity and stronger intent-matching pages should be prioritized."

        return visibility_line + rank_line + ref_line

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
                max_output_tokens=320,
            ),
            retry_once=False,
            timeout_sec=8,
        )
        if response is None:
            return _build_quick_summary()
        text = (response.text or "").strip()
        text = re.sub(r"\s+", " ", text)
        if not text:
            return _build_quick_summary()
        return text
    except Exception:
        return _build_quick_summary()


async def generate_deep_competitor_scores(
    url: str,
    questions: list[dict],
    seed_results: dict,
) -> list[dict]:
    """Generate competitor scores in one aggregate pass using ideas collected during core analysis."""
    client = get_client()
    domain = _normalize_domain(url)

    candidate_counts: dict[str, int] = {}
    for q in questions:
        qid = q.get("id")
        r = seed_results.get(qid, {}) if isinstance(seed_results, dict) else {}

        idea_candidates = r.get("idea_candidates", []) if isinstance(r, dict) else []
        if isinstance(idea_candidates, list):
            for item in idea_candidates:
                if isinstance(item, dict):
                    d = _normalize_domain(item.get("domain", ""))
                else:
                    d = _normalize_domain(str(item))
                if d and d != domain and d not in NON_COMPETITOR_DOMAINS:
                    candidate_counts[d] = candidate_counts.get(d, 0) + 2

        refs = r.get("references", []) if isinstance(r, dict) else []
        if isinstance(refs, list):
            for ref in refs:
                d = _normalize_domain(ref)
                if d and d != domain and d not in NON_COMPETITOR_DOMAINS:
                    candidate_counts[d] = candidate_counts.get(d, 0) + 1

        srcs = r.get("sources", []) if isinstance(r, dict) else []
        if isinstance(srcs, list):
            for src in srcs:
                d = _normalize_domain(src)
                if d and d != domain and d not in NON_COMPETITOR_DOMAINS:
                    candidate_counts[d] = candidate_counts.get(d, 0) + 1

        source_urls = r.get("source_urls", []) if isinstance(r, dict) else []
        if isinstance(source_urls, list):
            for u in source_urls:
                d = _normalize_domain(u)
                if d and d != domain and d not in NON_COMPETITOR_DOMAINS:
                    candidate_counts[d] = candidate_counts.get(d, 0) + 1

    top_candidates = [d for d, _ in sorted(candidate_counts.items(), key=lambda kv: kv[1], reverse=True)[:12]]
    if not top_candidates:
        # Quick grounded probe fallback to avoid empty competitor lists.
        probe_text = " ; ".join([q.get("text", "") for q in questions[:5]])
        probe_prompt = (
            f"Use Google Search and list domains that appear for these intents: {probe_text}. "
            f"Target domain is {domain}. Return short text with domains only."
        )
        probe_response = await _call_gemini_with_retry(
            client,
            probe_prompt,
            config=types.GenerateContentConfig(
                tools=[{"google_search": {}}],
                temperature=0.0,
                top_p=0.9,
                max_output_tokens=180,
            ),
            retry_once=False,
            timeout_sec=8,
        )
        probe_domains = []
        if probe_response is not None:
            probe_domains.extend(_extract_domains_from_text(probe_response.text or ""))
            refs, _, _, _ = _extract_grounding_signals(probe_response, domain)
            probe_domains.extend(refs)
        for d in probe_domains:
            nd = _normalize_domain(d)
            if nd and nd != domain and nd not in NON_COMPETITOR_DOMAINS:
                candidate_counts[nd] = candidate_counts.get(nd, 0) + 1

        top_candidates = [d for d, _ in sorted(candidate_counts.items(), key=lambda kv: kv[1], reverse=True)[:12]]
        if not top_candidates:
            return []

    # Fast deterministic fallback from observed first-pass signals.
    heuristic_scores: list[dict] = []
    max_hits = max(candidate_counts.values()) if candidate_counts else 1
    for d, hits in sorted(candidate_counts.items(), key=lambda kv: kv[1], reverse=True):
        if d in NON_COMPETITOR_DOMAINS or d == domain:
            continue
        score = int(round((hits / max_hits) * 100)) if max_hits > 0 else 50
        heuristic_scores.append(
            {
                "domain": d,
                "position": None,
                "score": max(35, min(95, score)),
                "evidence": "Ranked from repeated first-pass competitor context and source overlap.",
            }
        )
        if len(heuristic_scores) >= 5:
            break

    compact_questions = [q.get("text", "") for q in questions[:20]]
    prompt = f"""
    You are a market intelligence analyst.
    Use live Google Search and these user-intent queries to identify direct competitors for the target.

    Target domain: '{domain}'
    Queries: {json.dumps(compact_questions)}
    Candidate domains from first-pass visibility analysis: {json.dumps(top_candidates)}

    Return JSON only:
    {{
      "competitors": [
        {{
          "domain": "example.com",
          "position": 1,
          "score": 0,
          "evidence": "short reason"
        }}
      ]
    }}

    Rules:
    - Max 5 competitors.
    - Direct competitors only (same intent/category overlap).
    - Exclude aggregator/listing/general platform sites.
    - Exclude target domain.
    - score must be integer 0..100.
    - position must be 1..10 or null.
    - JSON only.
    """

    response = await _call_gemini_with_retry(
        client,
        prompt,
        config=types.GenerateContentConfig(
            tools=[{"google_search": {}}],
            temperature=0.0,
            top_p=0.9,
            max_output_tokens=420,
        ),
        retry_once=False,
        timeout_sec=10,
    )

    if response is None:
        return heuristic_scores

    parsed = _safe_json_parse(response.text or "")
    raw = parsed.get("competitors", []) if isinstance(parsed, dict) else []
    if not isinstance(raw, list):
        return heuristic_scores

    out: list[dict] = []
    seen = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        d = _normalize_domain(item.get("domain", ""))
        if (
            not d
            or d == domain
            or d in NON_COMPETITOR_DOMAINS
            or d in seen
        ):
            continue
        seen.add(d)
        pos = item.get("position")
        if not isinstance(pos, int) or pos < 1 or pos > 10:
            pos = None
        score = item.get("score")
        if not isinstance(score, int):
            score = 55
        score = max(0, min(100, score))
        out.append(
            {
                "domain": d,
                "position": pos,
                "score": score,
                "evidence": str(item.get("evidence", "")).strip()[:180],
            }
        )
        if len(out) >= 5:
            break

    return out if out else heuristic_scores

async def analyze_single_question(url: str, question: dict, include_competitors: bool = False) -> dict:
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
