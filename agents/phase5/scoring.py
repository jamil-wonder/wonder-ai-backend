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
        mentioned = bool(pdata.get("mentioned") or status == "Mentioned")
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
