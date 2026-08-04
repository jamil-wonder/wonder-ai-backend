def score_from_position(position: int | None) -> int:
    """Position-only fallback score (0..100) for when no per-question signal exists yet."""
    if not isinstance(position, int) or position < 1:
        return 50
    return max(0, min(100, 110 - (position * 10)))


def compute_competitor_score(seed_results: dict, domain: str) -> dict:
    """
    Score a competitor domain using the SAME weighted formula the target's own
    visibility score is built on (55% mention rate, 35% position, 10% citation
    rate — see compute_provider_score), by mining how often and where that
    domain actually surfaced across the real Phase 5 question results already
    collected for the target business (competitor_scores/competitors picks and
    references/sources/source_urls citations, across every provider and
    question). This makes competitor scores directly comparable to the
    target's own score instead of relying on a single AI-guessed 0..100
    number from the discovery prompt.

    Returns has_signal=False when this domain never actually showed up in the
    collected results (e.g. it was found only via a supplementary discovery
    probe) — callers should fall back to a position-based or AI estimate in
    that case rather than trusting an unmeasured "score".
    """
    from .helpers import _normalize_domain

    target_norm = _normalize_domain(domain)
    rows = seed_results if isinstance(seed_results, dict) else {}
    total = len(rows)

    mentioned_count = 0
    cited_count = 0
    pos_points: list[int] = []

    for _, row in rows.items():
        if not isinstance(row, dict):
            continue
        providers = row.get("providers") if isinstance(row.get("providers"), dict) else None
        provider_dicts = list(providers.values()) if providers else [row]

        q_mentioned = False
        q_cited = False
        q_position: int | None = None

        for pdata in provider_dicts:
            if not isinstance(pdata, dict):
                continue

            for c in (pdata.get("competitor_scores") or []):
                if isinstance(c, dict) and _normalize_domain(str(c.get("domain") or "")) == target_norm:
                    q_mentioned = True
                    pos = c.get("position")
                    if isinstance(pos, int) and (q_position is None or pos < q_position):
                        q_position = pos

            if not q_mentioned:
                for c in (pdata.get("competitors") or []):
                    if _normalize_domain(str(c)) == target_norm:
                        q_mentioned = True
                        break

            for key in ("references", "sources", "source_urls"):
                for v in (pdata.get(key) or []):
                    if _normalize_domain(str(v)) == target_norm:
                        q_cited = True
                        break
                if q_cited:
                    break

        if q_mentioned:
            mentioned_count += 1
            pos_points.append(max(10, 110 - (q_position * 10)) if isinstance(q_position, int) else 50)
        if q_cited:
            cited_count += 1

    mention_rate = ((mentioned_count + 1) / (total + 2)) if total >= 0 else 0.0
    citation_rate = ((cited_count + 1) / (total + 2)) if total >= 0 else 0.0

    if pos_points:
        avg_pos_score = sum(pos_points) / len(pos_points)
    else:
        avg_pos_score = 50.0 if mentioned_count > 0 else 0.0

    w_mention, w_position, w_citation = 0.55, 0.35, 0.10
    raw_score = (mention_rate * w_mention) + ((avg_pos_score / 100.0) * w_position) + (citation_rate * w_citation)
    composite = int(round(raw_score * 100))

    min_stable = 5
    if 0 < total < min_stable:
        scale = total / float(min_stable)
        composite = int(round(composite * scale + 50 * (1.0 - scale)))

    return {
        "score": composite,
        "mention_rate": round(mentioned_count / total, 4) if total else 0.0,
        "avg_pos_score": round(avg_pos_score, 2),
        "citation_rate": round(citation_rate, 4),
        "mentioned": mentioned_count,
        "cited": cited_count,
        "total": total,
        "has_signal": bool(mentioned_count or cited_count),
    }


def compute_provider_score(results: dict, provider: str) -> dict:
    provider_key = str(provider or "").strip().lower()
    rows = results if isinstance(results, dict) else {}
    total = len(rows)

    def _position_points(mentioned: bool, position: int | None) -> int:
        if not mentioned:
            return 0
        if isinstance(position, int) and 1 <= position <= 10:
            return max(10, 110 - (position * 10))
        return 50

    mentioned_count = 0
    cited_count = 0
    pos_points: list[int] = []

    for _, row in rows.items():
        providers = row.get("providers") if isinstance(row, dict) else None
        pdata = providers.get(provider_key) if isinstance(providers, dict) else None
        pdata = pdata if isinstance(pdata, dict) else {}

        status = str(pdata.get("status") or "")
        target_site = pdata.get("target_site") if isinstance(pdata.get("target_site"), dict) else {}
        target_site_status = str(target_site.get("status") or "").strip().lower()
        mentioned = bool(
            pdata.get("mentioned")
            or status == "Mentioned"
            or target_site_status == "matched"
        )
        position_raw = pdata.get("position")
        position = position_raw if isinstance(position_raw, int) else None
        cited = bool(
            pdata.get("cited")
            or (isinstance(pdata.get("sources"), list) and len(pdata.get("sources")) > 0)
            or (isinstance(pdata.get("references"), list) and len(pdata.get("references")) > 0)
            or (isinstance(pdata.get("source_urls"), list) and len(pdata.get("source_urls")) > 0)
        )

        if mentioned:
            mentioned_count += 1
            pos_points.append(_position_points(True, position))
        if cited:
            cited_count += 1

    not_mentioned_count = max(0, total - mentioned_count)

    mention_rate = ((mentioned_count + 1) / (total + 2)) if total >= 0 else 0.0
    citation_rate = ((cited_count + 1) / (total + 2)) if total >= 0 else 0.0

    if pos_points:
        avg_pos_score = (sum(pos_points) / len(pos_points))
    else:
        avg_pos_score = 50.0 if mentioned_count > 0 else 0.0

    w_mention = 0.55
    w_position = 0.35
    w_citation = 0.10

    raw_score = (mention_rate * w_mention + (avg_pos_score / 100.0) * w_position + citation_rate * w_citation)
    composite = int(round(raw_score * 100))

    min_stable = 5
    if total < min_stable and total > 0:
        scale = total / float(min_stable)
        composite = int(round(composite * scale + 50 * (1.0 - scale)))

    return {
        "provider": provider_key,
        "score": composite,
        "mention_rate": round(mentioned_count / total if total else 0.0, 4),
        "mention_rate_smoothed": round(mention_rate, 4),
        "avg_pos_score": round(avg_pos_score, 2),
        "citation_rate": round(citation_rate, 4),
        "mentioned": mentioned_count,
        "not_mentioned": not_mentioned_count,
        "cited": cited_count,
        "total": total,
    }
