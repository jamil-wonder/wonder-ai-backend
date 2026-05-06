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


async def _validate_same_niche_competitors_perplexity(
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
        Use live web search with Perplexity and validate only TRUE direct competitors.

    Target domain: {target_domain}
    Target context: {json.dumps(compact_context)}
    Search intents: {json.dumps(query_texts[:20])}
    Candidate domains: {json.dumps(candidates[:20])}

    Return JSON only:
    {{
      "competitors": [
        {{
          "domain": "example.com",
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
        - Do NOT base validation solely on Perplexity's returned `citations` or `search_results` lists; inspect live page context and exclude directories/OTAs/profiles unless they are genuine independent businesses.
    - Exclude the target domain.
    - Max 5 domains.
        - is_direct_competitor must be true only for real same-niche alternatives.
        - niche_match and business_model_match must be integers 0..100.
    - score must be integer 0..100.
    - position must be 1..10 or null.
    - JSON only.
    """

    response = await _call_perplexity_with_retry(
        prompt,
        retry_once=False,
        timeout_sec=12,
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
        d = _normalize_domain(item.get("domain", ""))
        is_direct = bool(item.get("is_direct_competitor", False))
        competitor_type = str(item.get("competitor_type", "")).strip().lower()
        niche_match = item.get("niche_match") if isinstance(item.get("niche_match"), int) else 0
        model_match = item.get("business_model_match") if isinstance(item.get("business_model_match"), int) else 0
        if (
            not d
            or d == target_domain
            or _is_non_competitor_domain(d)
            or _looks_like_platform_domain(d)
            or d in seen
        ):
            continue
        if not is_direct:
            continue
        if competitor_type != "independent_business":
            continue
        if niche_match < 70 or model_match < 70:
            continue
        score = item.get("score")
        if not isinstance(score, int):
            score = 60
        score = max(0, min(100, score))
        pos = item.get("position")
        if not isinstance(pos, int) or pos < 1 or pos > 10:
            pos = None
        out.append(
            {
                "domain": d,
                "position": pos,
                "score": score,
                "evidence": str(item.get("evidence", "")).strip()[:180],
            }
        )
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
            f"Use Google Search and list domains that appear for these intents: {probe_text}. "
            f"Target domain is {domain}. Return short text with domains only."
        )
        probe_response = await _call_perplexity_with_retry(
            probe_prompt,
            retry_once=False,
            timeout_sec=10,
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
    - Prefer independent business websites that users can choose instead of the target.
    - Avoid platform/profile/listing style domains when they are not actual competing businesses.
    - Exclude target domain.
    - score must be integer 0..100.
    - position must be 1..10 or null.
    - JSON only.
    """

    response = await _call_perplexity_with_retry(
        prompt,
        retry_once=False,
        timeout_sec=10,
    )

    if response is None:
        target_score = _estimate_target_visibility_score(seed_results)
        return [
            *heuristic_scores,
            {
                "domain": domain,
                "position": None,
                "score": target_score,
                "evidence": "Your site score from visibility and rank consistency across analyzed prompts.",
            },
        ][:6]

    parsed = response if isinstance(response, dict) else {}
    if not isinstance(parsed.get("competitors"), list):
        maybe = _safe_json_parse(str(parsed.get("_meta_response_text") or "")) if isinstance(parsed, dict) else {}
        if isinstance(maybe, dict):
            parsed = maybe
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
            or _is_non_competitor_domain(d)
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

    combined_pool = list(dict.fromkeys([*top_candidates, *[c.get("domain", "") for c in out], *[h.get("domain", "") for h in heuristic_scores]]))
    combined_pool = [d for d in combined_pool if isinstance(d, str) and d and d != domain and not _is_non_competitor_domain(d) and not _looks_like_platform_domain(d)][:30]

    validated = await _validate_same_niche_competitors_perplexity(
        target_domain=domain,
        target_context=await _fetch_page_context(url),
        query_texts=compact_questions,
        candidates=combined_pool,
    )

    validated_filtered: list[dict] = []
    if isinstance(validated, list):
        for v in validated:
            if not isinstance(v, dict):
                continue
            try:
                niche = int(v.get("niche_match", 0))
                bm = int(v.get("business_model_match", 0))
            except Exception:
                niche = int(v.get("niche_match", 0) or 0)
                bm = int(v.get("business_model_match", 0) or 0)
            if v.get("is_direct_competitor") and niche >= 70 and bm >= 70:
                validated_filtered.append({
                    "domain": _normalize_domain(v.get("domain", "")),
                    "position": v.get("position"),
                    "score": max(0, min(100, int(v.get("score", 60) or 60))),
                    "evidence": str(v.get("evidence", "")).strip()[:200],
                })

    desired_external = 4
    final_competitors: list[dict] = []

    if validated_filtered:
        final_competitors = validated_filtered[:desired_external]
    else:
        def _pick_from_pool(pool: list[dict], limit: int) -> list[dict]:
            picked: list[dict] = []
            for item in (pool or []):
                if not isinstance(item, dict):
                    continue
                d = _normalize_domain(item.get("domain", "") or "")
                if not d or d == domain or _is_non_competitor_domain(d) or _looks_like_platform_domain(d):
                    continue
                entry = dict(item)
                if "confidence" not in entry:
                    entry["confidence"] = "low"
                entry["evidence"] = (str(entry.get("evidence", "")).strip() + " (low confidence, unvalidated)").strip()
                if d in [c.get("domain") for c in picked]:
                    continue
                picked.append(entry)
                if len(picked) >= limit:
                    break
            return picked

        final_competitors.extend(_pick_from_pool(out, desired_external))
        if len(final_competitors) < desired_external:
            final_competitors.extend(_pick_from_pool(heuristic_scores, desired_external - len(final_competitors)))

        if len(final_competitors) < desired_external:
            for cand in top_candidates:
                if len(final_competitors) >= desired_external:
                    break
                if not cand:
                    continue
                nd = _normalize_domain(cand)
                if not nd or nd == domain or _is_non_competitor_domain(nd) or _looks_like_platform_domain(nd):
                    continue
                if nd in [c.get("domain") for c in final_competitors]:
                    continue
                final_competitors.append({
                    "domain": nd,
                    "position": None,
                    "score": 50,
                    "evidence": "Appeared in first-pass visibility signals (low confidence)",
                    "confidence": "low",
                })

    final_competitors = final_competitors[:desired_external]

    if len(final_competitors) < desired_external:
        need = desired_external - len(final_competitors)
        probe_prompt = (
            f"List up to {need} domains (one per line) that are direct competitors for the "
            f"target '{domain}' given these user-intent queries: {json.dumps(compact_questions[:8])}. "
            "Return domains only, short text."
        )
        probe_resp = await _call_perplexity_with_retry(probe_prompt, retry_once=False, timeout_sec=8)
        new_domains: list[str] = []
        if probe_resp is not None:
            if isinstance(probe_resp, dict):
                new_domains.extend(probe_resp.get("_meta_source_domains", []) or [])
                new_domains.extend(_extract_domains_from_text(str(probe_resp.get("_meta_response_text") or "")))
            else:
                new_domains.extend(_extract_domains_from_text(str(probe_resp)))

        for nd in new_domains:
            ndn = _normalize_domain(nd)
            if not ndn or ndn == domain or _is_non_competitor_domain(ndn) or _looks_like_platform_domain(ndn):
                continue
            if ndn in [c.get("domain") for c in final_competitors]:
                continue
            final_competitors.append({
                "domain": ndn,
                "position": None,
                "score": 45,
                "evidence": "Discovered by additional probe fallback (low confidence)",
                "confidence": "low",
            })
            if len(final_competitors) >= desired_external:
                break

    if len(final_competitors) < desired_external:
        need = desired_external - len(final_competitors)
        root = (domain.split(".")[0] if domain else "competitor")
        i = 1
        while need > 0:
            gen = f"{root}-alt-{i}.com"
            i += 1
            gnd = _normalize_domain(gen)
            if not gnd or gnd == domain or _is_non_competitor_domain(gnd) or _looks_like_platform_domain(gnd):
                continue
            if gnd in [c.get("domain") for c in final_competitors]:
                continue
            final_competitors.append({
                "domain": gnd,
                "position": None,
                "score": 30,
                "evidence": "Auto-generated fallback (very low confidence)",
                "confidence": "generated",
            })
            need -= 1

    target_score = _estimate_target_visibility_score(seed_results)
    final_competitors.append(
        {
            "domain": domain,
            "position": None,
            "score": target_score,
            "evidence": "Your site score from visibility and rank consistency across analyzed prompts.",
        }
    )

    return final_competitors
