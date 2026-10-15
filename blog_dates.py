"""L3: blogs are written with today's date, never an old year.

A model that is not told the date writes "best restaurants of 2023". The prompt
now carries today's date, and a finished draft is checked: a recent past year in
the title, meta title, slug or a heading is replaced with the current year, and
a past year anywhere else is listed for the owner to confirm.
"""
import re
from datetime import datetime

RECENT_YEARS = 6  # a year within this many years back is treated as "meant to be now"

_YEAR = re.compile(r"(?<![\d-])(20\d\d)(?![\d-])")


def _now(now=None) -> datetime:
    return now or datetime.now()


def date_prompt_block(now=None) -> str:
    n = _now(now)
    return (
        f"\n    DATE: Today is {n.day} {n.strftime('%B')} {n.year}. The current year is {n.year}. "
        f"If a year belongs in the title or text, it is {n.year}. Never write an earlier year as if it were current, "
        f"and do not mention any other year at all unless it appears in the business facts above.\n"
    )


def _is_stale(year: int, now) -> bool:
    n = _now(now)
    return n.year - RECENT_YEARS <= year < n.year


def fix_year_in_text(text: str, now=None) -> tuple[str, list[str]]:
    """Replace recent past years with the current year. Returns (text, the years replaced)."""
    n = _now(now)
    replaced: list[str] = []

    def swap(m):
        y = int(m.group(1))
        if _is_stale(y, n):
            replaced.append(m.group(1))
            return str(n.year)
        return m.group(1)

    return _YEAR.sub(swap, str(text or "")), replaced


def past_years_in(text: str, now=None) -> list[str]:
    n = _now(now)
    return sorted({m.group(1) for m in _YEAR.finditer(str(text or "")) if int(m.group(1)) < n.year})


def fix_blog_dates(title: str, meta_title: str, slug: str, sections: list[dict], now=None) -> dict:
    """Fix the places where a stale year misleads (title, meta title, slug, headings);
    list any past year left in the body for the owner."""
    n = _now(now)
    fixed_title, a = fix_year_in_text(title, n)
    fixed_meta, b = fix_year_in_text(meta_title, n)
    fixed_slug = slug
    for y in sorted({*a, *b}):
        fixed_slug = fixed_slug.replace(y, str(n.year))
    fixed_slug, c = fix_year_in_text(fixed_slug, n)
    out_sections: list[dict] = []
    fixed_headings: list[str] = []
    for s in sections:
        s = dict(s)
        for key in ("heading", "label"):
            if key in s:
                s[key], d = fix_year_in_text(s[key], n)
                fixed_headings += d
        out_sections.append(s)
    notes: list[str] = []
    changed = sorted({*a, *b, *c, *fixed_headings})
    if changed:
        notes.append(f"An out-of-date year ({', '.join(changed)}) was changed to {n.year} in the title or headings.")
    body_years = sorted({y for s in out_sections for y in past_years_in(s.get("content", ""), n)})
    if body_years:
        notes.append(f"The text mentions {', '.join(body_years)}. Confirm that is still accurate before publishing.")
    return {"title": fixed_title, "metaTitle": fixed_meta, "slug": fixed_slug, "sections": out_sections, "notes": notes}
