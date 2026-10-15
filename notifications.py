"""J10: in-app notifications. Pure builders here; storing and the API are in main_parts.

Four events: a scan finished, a Wonder Score moved by more than 5 points, a
competitor overtook the business, and the blog drafts are ready. The bell badge is
the number of unread ones - never a fixed number.
"""

SCORE_MOVE_THRESHOLD = 5      # "moved more than 5"
LIST_LIMIT = 10


def scan_complete(business_name: str, score) -> dict:
    name = business_name or "Your business"
    if isinstance(score, (int, float)):
        return {"type": "scan_complete", "title": "Scan complete",
                "body": f"{name}'s Wonder Score is {round(score)}/100.", "link": "/overview"}
    return {"type": "scan_complete", "title": "Website scan complete",
            "body": f"The website scan for {name} finished.", "link": "/analyser"}


def score_moved(business_name: str, previous, current):
    """None unless the score moved by MORE than the threshold."""
    if not isinstance(previous, (int, float)) or not isinstance(current, (int, float)):
        return None
    delta = round(current) - round(previous)
    if abs(delta) <= SCORE_MOVE_THRESHOLD:
        return None
    name = business_name or "Your business"
    word = "up" if delta > 0 else "down"
    return {"type": "score_moved", "title": f"Wonder Score {word} {abs(delta)} points",
            "body": f"{name} went from {round(previous)} to {round(current)}.", "link": "/overview"}


def _score_rows(rows):
    out = []
    for r in rows or []:
        if not isinstance(r, dict):
            continue
        score = r.get("score")
        domain = str(r.get("domain") or r.get("url") or "").strip().lower()
        if not isinstance(score, (int, float)) or not domain:
            continue
        is_user = bool(r.get("isUser") or r.get("is_user") or r.get("confidence") == "target")
        out.append({"domain": domain, "name": r.get("name") or domain, "score": score, "is_user": is_user})
    return out


def overtakers(previous_rows, current_rows) -> list[str]:
    """Names of competitors that were NOT ahead of the business last time and are now."""
    prev, cur = _score_rows(previous_rows), _score_rows(current_rows)
    prev_user = next((r for r in prev if r["is_user"]), None)
    cur_user = next((r for r in cur if r["is_user"]), None)
    if not prev_user or not cur_user:
        return []
    before = {r["domain"]: r["score"] for r in prev if not r["is_user"]}
    names = []
    for r in cur:
        if r["is_user"] or r["domain"] not in before:
            continue
        if before[r["domain"]] <= prev_user["score"] and r["score"] > cur_user["score"]:
            names.append(str(r["name"]))
    return names


def competitor_overtook(business_name: str, names: list[str]):
    if not names:
        return None
    who = names[0] if len(names) == 1 else f"{names[0]} and {len(names) - 1} more"
    return {"type": "competitor_overtook", "title": "A competitor overtook you",
            "body": f"{who} now scores higher than {business_name or 'your business'}.", "link": "/query"}


def blogs_ready(business_name: str, count: int) -> dict:
    return {"type": "blogs_ready", "title": "Your articles are ready",
            "body": f"{count} new draft{'s' if count != 1 else ''} for {business_name or 'your business'} to review.", "link": "/plan"}
