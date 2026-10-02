from __future__ import annotations

import html as _html
import re
from typing import Iterable, Optional


def clean_text(value: Optional[str]) -> str:
    """Unescape HTML entities and collapse whitespace.

    Names and descriptions often arrive from JSON-LD/meta with entities still
    escaped ("GAIL&#39;s") or with layout whitespace ("\\n      GAIL's ...").
    """
    if not value:
        return ""
    text = _html.unescape(str(value)).replace(" ", " ")
    # unescape twice: some sites double-escape ("&amp;#39;")
    text = _html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _alnum(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _prefix_matching_label(name: str, label: str) -> str:
    """The leading part of `name` that spells `label` ignoring spaces/punctuation
    ("Cafe Rouge Bristol" + "caferouge" -> "Cafe Rouge"), or "" if there isn't
    one on a clean word boundary."""
    target = _alnum(label)
    if len(target) < 3:
        return ""
    built = ""
    for index, char in enumerate(name):
        if char.isalnum():
            built += char.lower()
            if built == target:
                end = index + 1
                return name[:end].strip() if end >= len(name) or not name[end].isalnum() else ""
            if not target.startswith(built):
                return ""
    return ""


def prefer_brand_name(
    name: str,
    site_name: Optional[str],
    cities: Iterable[str] = (),
    multi_location: bool = False,
    domain_label: str = "",
) -> str:
    """On a multi-location site the first JSON-LD entry is usually one branch
    ("Hawksmoor Lakeland Guest House Windermere"), not the brand. Two signals
    can recover the brand, each only when the leftover words look like a branch:

    * the site's own og:site_name is a strict prefix of the name and the rest
      mentions a detected city OR the site is multi-location;
    * the site's domain label ("hawksmoor" from hawksmoor.com) spells the start
      of the name and the rest mentions a detected city (stricter: no
      multi-location shortcut, since a domain is a weaker brand signal).

    Anything ambiguous keeps the original name.
    """
    name = clean_text(name)
    if not name:
        return name
    city_list = [c for c in cities if c]

    def _branch_like(remainder: str, allow_multi: bool) -> bool:
        remainder = remainder.strip(" -|,:").lower()
        if not remainder:
            return False
        if any(re.search(rf"\b{re.escape(c.lower())}\b", remainder) for c in city_list):
            return True
        return allow_multi and multi_location

    brand = clean_text(site_name)
    if len(brand) >= 3 and name.lower() != brand.lower() and name.lower().startswith(brand.lower()):
        rest = name[len(brand):]
        # "Dishoomery" must not be treated as "Dishoom" + "ery"
        if not rest[:1].isalnum() and _branch_like(rest, allow_multi=True):
            return brand

    derived = _prefix_matching_label(name, domain_label) if domain_label else ""
    if derived and derived.lower() != name.lower() and _branch_like(name[len(derived):], allow_multi=False):
        return derived

    return name
