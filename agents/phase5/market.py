"""A2: competitor scores measured the same way as the customer's.

Every competitor is judged on the SAME questions (the ones open to anyone - not the branded ones, which name the business) and the SAME stored AI answers as
the business, by the same rule: a business counts as mentioned in an answer when it
is named in the answer text or cited as a source (B1). The score is the weighted
share of questions where it was mentioned, averaged over the models that answered
(same weights and same averaging as the business's own mention rate). No extra AI
calls - the answers a competitor would be judged on are the ones already collected,
because the 20 questions do not change per competitor.
"""
from __future__ import annotations

from .analysis import _named_in_answer
from .helpers import _normalize_domain
from .scoring import PROVIDER_NAMES, provider_result_state, question_weight


def _domains(pdata: dict) -> set[str]:
    out: set[str] = set()
    for key in ("sources", "references", "source_urls", "idea_candidates"):
        for item in pdata.get(key) or []:
            value = item.get("domain") if isinstance(item, dict) else item
            d = _normalize_domain(str(value or ""))
            if d:
                out.add(d)
    return out


def competitor_mentioned(pdata: dict, domain: str, name: str | None) -> bool:
    """Named in the answer text, or cited / suggested as a source."""
    d = _normalize_domain(domain)
    if not d:
        return False
    if _named_in_answer(pdata.get("llm_response") or "", d, name)[0]:
        return True
    return any(s == d or s.endswith("." + d) for s in _domains(pdata))


def target_mentioned(pdata: dict) -> bool:
    return bool(pdata.get("mentioned")) or str(pdata.get("status") or "") == "Mentioned"


def measure_business(results: dict, questions: list | None, is_mentioned) -> dict:
    """{score (0-100 int | None), appeared, answered} for one business.
    `is_mentioned(provider_data) -> bool`."""
    rows = results if isinstance(results, dict) else {}
    weight_by_qid = {}
    for q in questions or []:
        if isinstance(q, dict):
            weight_by_qid[str(q.get("id"))] = question_weight(q.get("category") or q.get("type"))

    per_model: list[float] = []
    for provider in PROVIDER_NAMES:
        hit_w = answered_w = 0
        for qid, row in rows.items():
            providers = row.get("providers") if isinstance(row, dict) else None
            pdata = providers.get(provider) if isinstance(providers, dict) else None
            if not isinstance(pdata, dict) or provider_result_state(pdata) != "ok":
                continue
            w = weight_by_qid.get(str(qid), 2)
            answered_w += w
            if is_mentioned(pdata):
                hit_w += w
        if answered_w:
            per_model.append(100.0 * hit_w / answered_w)

    appeared = answered = 0
    for row in rows.values():
        providers = row.get("providers") if isinstance(row, dict) else None
        ok = [p for p in (providers or {}).values() if isinstance(p, dict) and provider_result_state(p) == "ok"]
        if not ok:
            continue
        answered += 1
        if any(is_mentioned(p) for p in ok):
            appeared += 1
    score = round(sum(per_model) / len(per_model)) if per_model else None
    return {"score": score, "appeared": appeared, "answered": answered}


def build_ladder(results: dict, questions: list | None, target_domain: str, target_name: str | None, competitors: list[dict]) -> dict:
    """The measured market ladder. rows are best-first, each with score, appeared/answered
    counts and `score_basis: "measured"`. `position` is the business's market position
    (0-100: 100 = top of the ladder, 0 = bottom) or None when there is nobody to compare with."""
    # Branded questions name the business, so no competitor could appear in them: the
    # comparison uses the questions that are open to anyone (non-branded, local, broad).
    open_questions = [q for q in (questions or []) if isinstance(q, dict) and question_weight(q.get("category") or q.get("type")) != 1]
    open_ids = {str(q.get("id")) for q in open_questions}
    results = {qid: row for qid, row in (results or {}).items() if str(qid) in open_ids}
    questions = open_questions
    target = measure_business(results, questions, target_mentioned)
    rows = []
    if target["score"] is not None:
        rows.append({"domain": _normalize_domain(target_domain), "name": target_name, "is_user": True, **target})
    for c in competitors or []:
        d = _normalize_domain(str((c or {}).get("domain") or ""))
        if not d or d == _normalize_domain(target_domain):
            continue
        m = measure_business(results, questions, lambda p, d=d, n=c.get("name"): competitor_mentioned(p, d, n))
        if m["score"] is not None:
            rows.append({"domain": d, "name": c.get("name"), "is_user": False, **m})
    rows.sort(key=lambda r: r["score"], reverse=True)
    position = None
    others = [r for r in rows if not r["is_user"]]
    me = next((r for r in rows if r["is_user"]), None)
    if me and others:
        ahead = sum(1 for r in others if r["score"] > me["score"])
        position = round(100.0 * (len(others) - ahead) / len(others))
    return {"rows": rows, "position": position, "measured": bool(me and others)}


def evidence_text(row: dict) -> str:
    return f"Named or cited in {row['appeared']} of {row['answered']} questions, across the AI models that answered."
