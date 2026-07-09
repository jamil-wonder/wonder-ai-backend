import json
import re

from .config import NON_COMPETITOR_DOMAINS
from .context import _fetch_page_context
from .helpers import (
    _estimate_target_visibility_score,
    _extract_domains_from_text,
    _flatten_multi_result,
    _is_non_competitor_domain,
    _looks_like_platform_domain,
    _normalize_domain,
    _safe_json_parse,
)
from .providers import _call_perplexity_with_retry
from .providers import _call_claude_web_search_with_retry


def _canonical_site_url(value: str) -> str:
    domain = _normalize_domain(value)
    return f"https://{domain}/" if domain else ""


def _clean_business_name(value: str, domain: str) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip())
    text = re.sub(r"\s+[|-]\s+.*$", "", text).strip()
    if text and len(text) <= 80 and "." not in text:
        return text
    root = str(domain or "").split(".")[0].replace("-", " ").strip()
    return root.title() if root else domain


def _normalize_competitor_item(item: dict, target_domain: str) -> dict | None:
    if not isinstance(item, dict):
        return None

    raw_url = str(item.get("url") or item.get("homepage_url") or item.get("source_url") or "").strip()
    raw_domain = str(item.get("domain") or raw_url or "").strip()
    domain = _normalize_domain(raw_domain)
    if (
        not domain
        or domain == target_domain
        or _is_non_competitor_domain(domain)
        or _looks_like_platform_domain(domain)
    ):
        return None

    url = raw_url if raw_url.startswith(("http://", "https://")) else _canonical_site_url(domain)
    url_domain = _normalize_domain(url)
    if url_domain and url_domain != domain:
        url = _canonical_site_url(domain)

    pos = item.get("position")
    if not isinstance(pos, int) or pos < 1 or pos > 10:
        pos = None

    score = item.get("score")
    if not isinstance(score, int):
        score = item.get("confidence_score")
    if not isinstance(score, int):
        score = 60

    name = _clean_business_name(str(item.get("name") or item.get("business_name") or ""), domain)
    evidence = str(item.get("evidence") or item.get("reason") or "").strip()

    return {
        "domain": domain,
        "url": url or _canonical_site_url(domain),
        "name": name,
        "position": pos,
        "score": max(0, min(100, score)),
        "evidence": evidence[:220],
        "confidence": str(item.get("confidence") or "high").strip().lower(),
    }


async def _validate_same_niche_competitors_claude(
    *,
    target_domain: str,
    target_context: dict,
    query_texts: list[str],
    candidates: list[str],
) -> list[dict]:
    if not candidates:
        return []

    compact_context = {
        "category": str(target_context.get("category") or "").strip(),
        "location": str(target_context.get("location") or "").strip(),
        "services": [str(s).strip() for s in (target_context.get("services") or []) if str(s).strip()][:6],
        "description": str(target_context.get("description") or "").strip()[:180],
    }

    prompt = f"""
        You are a strict competitor validation analyst.
        Use live web search and validate only TRUE direct competitors.

    Target domain: {target_domain}
    Target context: {json.dumps(compact_context)}
    Search intents: {json.dumps(query_texts[:20])}
    Candidate domains: {json.dumps(candidates[:20])}

    Return JSON only:
    {{
      "competitors": [
        {{
          "domain": "example.com",
          "url": "https://example.com/",
          "name": "Business name",
          "is_direct_competitor": true,
          "competitor_type": "independent_business",
          "niche_match": 0,
          "business_model_match": 0,
          "score": 0,
          "position": null,
          "evidence": "short reason"
        }}
      ]
    }}

    Rules:
        - Include only same-niche, same customer-intent competitors.
        - A valid competitor must have both:
            1) niche_match >= 80
            2) business_model_match >= 80
        - competitor_type must be one of:
            independent_business, marketplace, directory, social_profile, review_listing, media, other
        - Prefer independent_business.
        - Reject platforms/profiles/listings when they are not actual competing businesses.
        - Use search results to verify the actual business homepage and business name.
        - url must be the official homepage or best official business page, not a directory/profile route.
    - Exclude the target domain.
    - Max 5 domains.
        - is_direct_competitor must be true only for real same-niche alternatives.
        - niche_match and business_model_match must be integers 0..100.
    - score must be integer 0..100.
    - position must be 1..10 or null.
    - JSON only.
    """

    response = await _call_claude_web_search_with_retry(
        prompt,
        retry_once=False,
        timeout_sec=28,
        max_uses=5,
    )
    if not isinstance(response, dict):
        return []

    parsed = response
    if not isinstance(parsed.get("competitors"), list):
        maybe = _safe_json_parse(str(parsed.get("_meta_response_text") or ""))
        if isinstance(maybe, dict):
            parsed = maybe

    raw = parsed.get("competitors", []) if isinstance(parsed, dict) else []
    if not isinstance(raw, list):
        return []

    out: list[dict] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        normalized = _normalize_competitor_item(item, target_domain)
        if not normalized:
            continue
        d = normalized["domain"]
        is_direct = bool(item.get("is_direct_competitor", False))
        competitor_type = str(item.get("competitor_type", "")).strip().lower()
        niche_match = item.get("niche_match") if isinstance(item.get("niche_match"), int) else 0
        model_match = item.get("business_model_match") if isinstance(item.get("business_model_match"), int) else 0
        if d in seen:
            continue
        if not is_direct:
            continue
        if competitor_type != "independent_business":
            continue
        if niche_match < 70 or model_match < 70:
            continue
        normalized["score"] = max(normalized["score"], 55)
        normalized["confidence"] = "verified"
        out.append(normalized)
        seen.add(d)
        if len(out) >= 5:
            break
    return out


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

        for raw_r in rows:
            r = _flatten_multi_result(raw_r)
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
        domain = _normalize_domain(url)

        compact_items = []
        for q in questions[:30]:
            qid = q.get("id")
            raw_r = results.get(qid, {}) if isinstance(results, dict) else {}
            r = _flatten_multi_result(raw_r)
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

        response = await _call_perplexity_with_retry(
            prompt,
            retry_once=False,
            timeout_sec=8,
        )
        if response is None:
            return _build_quick_summary()
        text = str((response.get("_meta_response_text") if isinstance(response, dict) else "") or "").strip()
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
    domain = _normalize_domain(url)

    candidate_counts: dict[str, int] = {}
    for q in questions:
        qid = q.get("id")
        raw_r = seed_results.get(qid, {}) if isinstance(seed_results, dict) else {}
        r = _flatten_multi_result(raw_r)

        idea_candidates = r.get("idea_candidates", []) if isinstance(r, dict) else []
        if isinstance(idea_candidates, list):
            for item in idea_candidates:
                if isinstance(item, dict):
                    d = _normalize_domain(item.get("domain", ""))
                else:
                    d = _normalize_domain(str(item))
                if d and d != domain and d not in NON_COMPETITOR_DOMAINS and not _looks_like_platform_domain(d):
                    candidate_counts[d] = candidate_counts.get(d, 0) + 2

        refs = r.get("references", []) if isinstance(r, dict) else []
        if isinstance(refs, list):
            for ref in refs:
                d = _normalize_domain(ref)
                if d and d != domain and d not in NON_COMPETITOR_DOMAINS and not _looks_like_platform_domain(d):
                    candidate_counts[d] = candidate_counts.get(d, 0) + 1

        srcs = r.get("sources", []) if isinstance(r, dict) else []
        if isinstance(srcs, list):
            for src in srcs:
                d = _normalize_domain(src)
                if d and d != domain and d not in NON_COMPETITOR_DOMAINS and not _looks_like_platform_domain(d):
                    candidate_counts[d] = candidate_counts.get(d, 0) + 1

        source_urls = r.get("source_urls", []) if isinstance(r, dict) else []
        if isinstance(source_urls, list):
            for u in source_urls:
                d = _normalize_domain(u)
                if d and d != domain and d not in NON_COMPETITOR_DOMAINS and not _looks_like_platform_domain(d):
                    candidate_counts[d] = candidate_counts.get(d, 0) + 1

    sorted_candidates = sorted(candidate_counts.items(), key=lambda kv: kv[1], reverse=True)
    top_candidates = [d for d, hits in sorted_candidates if hits >= 3 and not _is_non_competitor_domain(d)][:20]
    if not top_candidates:
        top_candidates = [d for d, hits in sorted_candidates if hits >= 2 and not _is_non_competitor_domain(d)][:20]
    if not top_candidates:
        top_candidates = [d for d, _ in sorted_candidates if not _is_non_competitor_domain(d) and not _looks_like_platform_domain(d)][:20]

    if len(top_candidates) < 3:
        probe_text = " ; ".join([q.get("text", "") for q in questions[:5]])
        probe_prompt = (
            f"Use web search and list official business domains that are direct competitors for these intents: {probe_text}. "
            f"Target domain is {domain}. Return JSON only: "
            '{"competitors":[{"domain":"example.com","url":"https://example.com/","name":"Business name","evidence":"short reason"}]}'
        )
        probe_response = await _call_claude_web_search_with_retry(
            probe_prompt,
            retry_once=False,
            timeout_sec=24,
            max_uses=4,
        )
        probe_domains = []
        if probe_response is not None:
            if isinstance(probe_response, dict):
                probe_domains.extend(probe_response.get("_meta_source_domains", []) or [])
                probe_domains.extend(_extract_domains_from_text(str(probe_response.get("_meta_response_text") or "")))
        for d in probe_domains:
            nd = _normalize_domain(d)
            if nd and nd != domain and not _is_non_competitor_domain(nd) and not _looks_like_platform_domain(nd):
                candidate_counts[nd] = candidate_counts.get(nd, 0) + 1

        top_candidates = [d for d, _ in sorted(candidate_counts.items(), key=lambda kv: kv[1], reverse=True)[:15]]

    heuristic_scores: list[dict] = []
    max_hits = max(candidate_counts.values()) if candidate_counts else 1
    for d, hits in sorted(candidate_counts.items(), key=lambda kv: kv[1], reverse=True):
        if _is_non_competitor_domain(d) or d == domain or _looks_like_platform_domain(d):
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
    Use live web search and these user-intent queries to identify direct competitors for the target.

    Target domain: '{domain}'
    Queries: {json.dumps(compact_questions)}
    Candidate domains from first-pass visibility analysis: {json.dumps(top_candidates)}

    Return JSON only:
    {{
      "competitors": [
        {{
          "name": "Business name",
          "domain": "example.com",
          "url": "https://example.com/",
          "position": 1,
          "score": 0,
          "confidence": "high",
          "evidence": "short reason"
        }}
      ]
    }}

    Rules:
    - Max 5 competitors.
    - Direct competitors only (same intent/category overlap).
    - Prefer independent business websites that users can choose instead of the target.
    - Use the official business homepage or best official business URL.
    - Do not return review/listing/profile/article URLs as competitor URLs.
    - Avoid platform/profile/listing style domains when they are not actual competing businesses.
    - Exclude target domain.
    - name must be the competitor's public business name, not a route title.
    - url must match the returned domain.
    - score must be integer 0..100.
    - position must be 1..10 or null.
    - JSON only.
    """

    response = await _call_claude_web_search_with_retry(
        prompt,
        retry_once=False,
        timeout_sec=30,
        max_uses=6,
    )

    def _with_target(items: list[dict]) -> list[dict]:
        target_score = _estimate_target_visibility_score(seed_results)
        final = [
            item for item in items
            if isinstance(item, dict) and _normalize_domain(item.get("domain", "")) != domain
        ][:4]
        final.append(
            {
                "domain": domain,
                "url": _canonical_site_url(domain),
                "name": _clean_business_name("", domain),
                "position": None,
                "score": target_score,
                "evidence": "Your site score from visibility and rank consistency across analyzed prompts.",
                "confidence": "target",
            }
        )
        return final

    if response is None:
        return _with_target(heuristic_scores)

    parsed = response if isinstance(response, dict) else {}
    if not isinstance(parsed.get("competitors"), list):
        maybe = _safe_json_parse(str(parsed.get("_meta_response_text") or "")) if isinstance(parsed, dict) else {}
        if isinstance(maybe, dict):
            parsed = maybe
    raw = parsed.get("competitors", []) if isinstance(parsed, dict) else []
    if not isinstance(raw, list):
        return _with_target(heuristic_scores)

    out: list[dict] = []
    seen = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        normalized = _normalize_competitor_item(item, domain)
        if not normalized:
            continue
        d = normalized["domain"]
        if d in seen:
            continue
        seen.add(d)
        out.append(normalized)
        if len(out) >= 5:
            break

    combined_pool = list(dict.fromkeys([*top_candidates, *[c.get("domain", "") for c in out], *[h.get("domain", "") for h in heuristic_scores]]))
    combined_pool = [d for d in combined_pool if isinstance(d, str) and d and d != domain and not _is_non_competitor_domain(d) and not _looks_like_platform_domain(d)][:30]

    validated = await _validate_same_niche_competitors_claude(
        target_domain=domain,
        target_context=await _fetch_page_context(url),
        query_texts=compact_questions,
        candidates=combined_pool,
    )

    validated_filtered: list[dict] = []
    if isinstance(validated, list):
        for v in validated:
            normalized = _normalize_competitor_item(v, domain)
            if normalized:
                normalized["confidence"] = v.get("confidence") or "verified"
                validated_filtered.append(normalized)

    desired_external = 4
    final_competitors: list[dict] = []

    if validated_filtered:
        final_competitors = validated_filtered[:desired_external]
    else:
        for item in out:
            if not isinstance(item, dict):
                continue
            normalized = _normalize_competitor_item(item, domain)
            if not normalized:
                continue
            normalized["confidence"] = normalized.get("confidence") or "claude-discovered"
            if normalized["domain"] in [c.get("domain") for c in final_competitors]:
                continue
            final_competitors.append(normalized)
            if len(final_competitors) >= desired_external:
                break

    final_competitors = final_competitors[:desired_external]

    target_score = _estimate_target_visibility_score(seed_results)
    final_competitors.append(
        {
            "domain": domain,
            "url": _canonical_site_url(domain),
            "name": _clean_business_name("", domain),
            "position": None,
            "score": target_score,
            "evidence": "Your site score from visibility and rank consistency across analyzed prompts.",
            "confidence": "target",
        }
    )

    return final_competitors
