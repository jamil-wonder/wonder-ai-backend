"""E1: which stored runs a shared report may compare against.

The public report must show exactly what the app shows, built from stored
COMPLETE scan runs. Two rules keep it honest:
  - "what moved" needs a second complete run with its own Search Tracker data;
  - "vs last week" needs a complete run at least 6 days older than the latest.
Anything else is left off the report rather than guessed."""
from datetime import datetime, timedelta

MIN_DAYS_FOR_WEEK_COMPARISON = 6


def parse_iso(value):
    try:
        return datetime.fromisoformat(str(value or "").replace("Z", "")).replace(tzinfo=None)
    except Exception:
        return None


def pick_previous_score(history, latest_run_id, latest_at, min_days: int = MIN_DAYS_FOR_WEEK_COMPARISON):
    """The most recent complete run that is at least `min_days` older than the
    latest one, as {"score", "at", "run_id"}; None when there is not one."""
    latest = parse_iso(latest_at)
    if latest is None or not isinstance(history, list):
        return None
    cutoff = latest - timedelta(days=min_days)
    best = None
    for entry in history:
        if not isinstance(entry, dict) or entry.get("run_id") == latest_run_id:
            continue
        score = entry.get("score")
        at = parse_iso(entry.get("created_at"))
        if not isinstance(score, (int, float)) or at is None or at > cutoff:
            continue
        if best is None or at > best[0]:
            best = (at, {"score": score, "at": entry.get("created_at"), "run_id": entry.get("run_id")})
    return best[1] if best else None


def pick_previous_tracker_job_id(runs, latest_run_id, latest_job_id):
    """`runs`: complete scan runs, newest first. The Search Tracker run of the
    newest OLDER complete run whose tracker data differs from the latest's."""
    seen_latest = False
    for run in runs or []:
        if not isinstance(run, dict):
            continue
        if run.get("run_id") == latest_run_id:
            seen_latest = True
            continue
        if not seen_latest:
            continue  # newer than the latest complete run: not a "previous" one
        job_id = run.get("tracker_job_id")
        if job_id and job_id != latest_job_id:
            return job_id
    return None
