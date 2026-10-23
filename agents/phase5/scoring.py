import os


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


PROVIDER_NAMES = ("perplexity", "chatgpt", "claude", "gemini")

# A4 (decided): each model is judged on its OWN answers. A model counts only if it answered at least
# 75% of the questions it was asked (it failed no more than this share). A run is scored from the models
# that count, provided at least MIN_USABLE_MODELS of them do; with fewer it is "incomplete" and no score is
# written (J6). The failures are on our side (credits, rate limits, timeouts), so they must never be
# reported to the client as zero.
INCOMPLETE_FAILED_FRACTION = float(os.getenv("PHASE5_INCOMPLETE_FAILED_FRACTION", "0.25"))
MIN_USABLE_MODELS = int(os.getenv("PHASE5_MIN_USABLE_MODELS", "3"))


def provider_counts(results: dict, provider: str) -> tuple:
    """(answered ok, failed) for one model across a run; not-run answers are neither."""
    ok = failed = 0
    for row in (results if isinstance(results, dict) else {}).values():
        providers = row.get("providers") if isinstance(row, dict) else None
        pdata = providers.get(provider) if isinstance(providers, dict) else None
        if not isinstance(pdata, dict):
            continue
        state = provider_result_state(pdata)
        if state == "ok":
            ok += 1
        elif state == "failed":
            failed += 1
    return ok, failed


def provider_usable(ok: int, failed: int) -> bool:
    """A model counts when it answered something and failed no more than 25% of what it was asked."""
    attempted = ok + failed
    return ok > 0 and (failed / attempted) <= INCOMPLETE_FAILED_FRACTION
# B2: more than this share of returned answers with no captured sources = citation capture failed
CITATIONS_MISSING_FRACTION = 0.5


def provider_result_state(pdata) -> str:
    """'ok' | 'failed' | 'not_run' for one model's answer to one question.

    New results carry an explicit `fetch_status`. Results stored before that
    field existed are treated as failed when they carry an `error` string, so
    historical rows are read the same way as new ones.
    """
    if not isinstance(pdata, dict):
        return "failed"
    explicit = str(pdata.get("fetch_status") or "").strip().lower()
    if explicit in ("ok", "failed", "not_run"):
        return explicit
    return "failed" if pdata.get("error") else "ok"


# ---------------------------------------------------------------------------
# A1: the ONE Wonder Score formula. Weights are fixed from launch (decided
# 29 Sept): every historical point must stay comparable, so don't change them.
# ---------------------------------------------------------------------------
WONDER_WEIGHT_MENTION = 0.60
WONDER_WEIGHT_WEBSITE = 0.30
WONDER_WEIGHT_MARKET = 0.10


def question_weight(category) -> int:
    """Branded questions count 1; Non-branded, Local SEO and Broad SEO count 2.
    So a business that only appears when asked by name scores far lower than
    one AI recommends unprompted. Unknown/missing category counts 2."""
    return 1 if str(category or "").strip().lower() == "branded" else 2


def compute_mention_rate(results: dict, questions: list | None) -> float | None:
    """Weighted share (0-100) of questions where the business was mentioned,
    averaged over the models whose answers actually came back (A4: failed and
    not-run answers are missing data, so they are left out). None when no
    model answered anything."""
    rows = results if isinstance(results, dict) else {}
    weight_by_qid = {}
    for q in (questions or []):
        if isinstance(q, dict):
            weight_by_qid[str(q.get("id"))] = question_weight(q.get("category") or q.get("type"))

    per_model_rates = []
    for provider in PROVIDER_NAMES:
        if not provider_usable(*provider_counts(rows, provider)):
            continue  # A4: a model that failed too many of its own answers is left out, not scored as zero
        mentioned_w = 0
        answered_w = 0
        for qid, row in rows.items():
            providers = row.get("providers") if isinstance(row, dict) else None
            pdata = providers.get(provider) if isinstance(providers, dict) else None
            if not isinstance(pdata, dict) or provider_result_state(pdata) != "ok":
                continue
            w = weight_by_qid.get(str(qid), 2)
            answered_w += w
            vote = pdata.get("vote") if isinstance(pdata.get("vote"), dict) else None
            if vote and (vote.get("of") or 0) > 1:
                # A5: the model was asked several times - count the share of
                # answers that mentioned us, not a coin-flip yes/no, so a
                # borderline question moves the score by a fraction.
                mentioned_w += w * min(1.0, max(0.0, vote.get("mentioned", 0) / vote["of"]))
            elif pdata.get("mentioned") or str(pdata.get("status") or "") == "Mentioned":
                mentioned_w += w
        if answered_w > 0:
            per_model_rates.append(100.0 * mentioned_w / answered_w)
    if not per_model_rates:
        return None
    return round(sum(per_model_rates) / len(per_model_rates), 2)


def compute_mention_rate_margin(results: dict, questions: list | None) -> float | None:
    """How much the mention rate (0-100 points) can move from one run to the
    next purely because the models answer differently each time (A5).

    Each answer is a vote share k/n; its sampling variance is p(1-p)/n with p
    smoothed away from 0 and 1 (a unanimous 2-of-2 is not proof of certainty).
    Variances are combined with the same weights as the mention rate itself;
    the margin is two standard errors. None when no model answered."""
    rows = results if isinstance(results, dict) else {}
    weight_by_qid = {}
    for q in (questions or []):
        if isinstance(q, dict):
            weight_by_qid[str(q.get("id"))] = question_weight(q.get("category") or q.get("type"))

    model_vars = []
    for provider in PROVIDER_NAMES:
        if not provider_usable(*provider_counts(rows, provider)):
            continue
        total_w = 0
        cells = []
        for qid, row in rows.items():
            providers = row.get("providers") if isinstance(row, dict) else None
            pdata = providers.get(provider) if isinstance(providers, dict) else None
            if not isinstance(pdata, dict) or provider_result_state(pdata) != "ok":
                continue
            w = weight_by_qid.get(str(qid), 2)
            vote = pdata.get("vote") if isinstance(pdata.get("vote"), dict) else None
            if vote and (vote.get("of") or 0) > 0:
                k, n = vote.get("mentioned", 0), vote["of"]
            else:
                k, n = (1 if pdata.get("mentioned") or str(pdata.get("status") or "") == "Mentioned" else 0), 1
            p = (k + 0.1) / (n + 0.2)
            cells.append((w, p * (1 - p) / n))
            total_w += w
        if total_w > 0:
            model_vars.append(sum((w / total_w) ** 2 * v for w, v in cells))
    if not model_vars:
        return None
    variance = sum(model_vars) / (len(model_vars) ** 2)
    return round(2 * (variance ** 0.5) * 100, 1)


def wonder_margin(mention_rate_margin) -> float | None:
    """The mention rate carries 60% of the Wonder Score, so its noise does too."""
    if not isinstance(mention_rate_margin, (int, float)):
        return None
    return round(0.6 * float(mention_rate_margin), 1)


def compute_wonder_score(mention_rate, technical_score, market_position) -> float | None:
    """Wonder Score = 60% mention rate + 30% website health + 10% market
    position, each input 0-100. Returns None (no score) when mention rate or
    website health is missing: a number built from half the inputs would not
    be comparable with next week's (J6/L1). Market position is 0 until A2
    measures it."""
    def _num(v):
        return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None

    mr, tech = _num(mention_rate), _num(technical_score)
    if mr is None or tech is None:
        return None
    market = _num(market_position) or 0.0
    clamp = lambda v: max(0.0, min(100.0, v))
    # Whole-number maths (hundredths of a point) with round-half-up, so the
    # frontend popover - which does the identical integer maths - can never
    # land on a different last digit than the stored score. (Float rounding
    # differs between Python and JavaScript on exact ties like 26.55.)
    mr_h, tech_h, market_h = (int(round(clamp(v) * 100)) for v in (mr, tech, market))
    total = mr_h * 60 + tech_h * 30 + market_h * 10      # units of 1/10000
    return ((total + 500) // 1000) / 10


def compute_run_health(results: dict) -> dict:
    """Count ok / failed / not_run model results across a whole multi-model run."""
    counts = {"ok": 0, "failed": 0, "not_run": 0}
    failed_by_provider: dict[str, int] = {}
    citations_missing = 0
    citations_missing_by_provider: dict[str, int] = {}
    rows = results if isinstance(results, dict) else {}
    for row in rows.values():
        providers = row.get("providers") if isinstance(row, dict) else None
        if not isinstance(providers, dict):
            continue
        # (single-provider rows have no "providers" and are not counted here)
        for name, pdata in providers.items():
            state = provider_result_state(pdata)
            counts[state] += 1
            if state == "failed":
                failed_by_provider[name] = failed_by_provider.get(name, 0) + 1
            elif state == "ok" and isinstance(pdata, dict) and pdata.get("citations_missing"):
                citations_missing += 1
                citations_missing_by_provider[name] = citations_missing_by_provider.get(name, 0) + 1
    # not_run (e.g. Gemini switched off on purpose) is neither a success nor
    # a failure, so it stays out of the failure rate.
    attempted = counts["ok"] + counts["failed"]
    fraction = (counts["failed"] / attempted) if attempted else 0.0
    # B2: when more than half of the answers that DID come back have no
    # captured sources, citation capture is broken for this run - the cited
    # half of "mentioned" cannot be trusted, so the run is incomplete too.
    cm_fraction = (citations_missing / counts["ok"]) if counts["ok"] else 0.0
    cm_majority = counts["ok"] > 0 and cm_fraction > CITATIONS_MISSING_FRACTION
    # A4: judge each model on its own answers; the run is scored from the models that count
    per_model = {name: provider_counts(rows, name) for name in PROVIDER_NAMES}
    asked = [name for name, (ok, failed) in per_model.items() if ok + failed > 0]
    usable = [name for name in asked if provider_usable(*per_model[name])]
    unusable = [name for name in asked if name not in usable]
    needed = min(MIN_USABLE_MODELS, len(asked))
    too_many_failed = bool(asked) and len(usable) < needed
    return {
        **counts,
        "failed_fraction": round(fraction, 4),
        "failed_by_provider": failed_by_provider,
        "models_asked": asked,
        "models_usable": usable,
        "models_unusable": unusable,
        "citations_missing": citations_missing,
        "citations_missing_fraction": round(cm_fraction, 4),
        "citations_missing_by_provider": citations_missing_by_provider,
        "citations_missing_majority": cm_majority,
        "incomplete": too_many_failed or cm_majority,
        "incomplete_reason": "failed_answers" if too_many_failed else ("citations_missing" if cm_majority else None),
    }


def compute_provider_score(results: dict, provider: str) -> dict:
    provider_key = str(provider or "").strip().lower()
    all_rows = results if isinstance(results, dict) else {}
    # Only answers the model actually gave count. A failed or not-run fetch
    # is missing data, not "not mentioned" (A4).
    rows = {}
    failed_count = 0
    not_run_count = 0
    for qid, row in all_rows.items():
        providers = row.get("providers") if isinstance(row, dict) else None
        pdata = providers.get(provider_key) if isinstance(providers, dict) else None
        state = provider_result_state(pdata) if isinstance(providers, dict) else "ok"
        if state == "failed":
            failed_count += 1
        elif state == "not_run":
            not_run_count += 1
        else:
            rows[qid] = row
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
        # B1: a mention is being named in the answer or cited as a source
        # (both already set status "Mentioned"). "The answer matched the
        # site's facts" is relevance, not a mention, and no longer counts.
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
        "failed": failed_count,
        "not_run": not_run_count,
        # False when this model gave no usable answer at all: show
        # "Unavailable", not a 0 score.
        "available": total > 0,
    }
