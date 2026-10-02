"""Deterministic location extraction for a crawled business website.

Replaces "take addresses[0] from a Python set" (arbitrary order, raw junk
strings, full street addresses where the product needs a city) with ranked,
validated, city-level locations plus a multi-location flag.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import parse_qs, unquote, urlparse

_UK_PC = re.compile(r"\b([A-Z]{1,2}\d[A-Z\d]?)\s*(\d[ABD-HJLNP-UW-Z]{2})\b", re.I)
_US_ZIP = re.compile(r"\b([A-Z]{2})\s+(\d{5})(?:-\d{4})?\b")
_AU_PC = re.compile(r"\b(NSW|VIC|QLD|WA|SA|TAS|ACT|NT)\s+(\d{4})\b")
_CA_PC = re.compile(r"\b([ABCEGHJ-NPRSTVXY]\d[ABCEGHJ-NPRSTV-Z])\s*(\d[ABCEGHJ-NPRSTV-Z]\d)\b", re.I)
_LATLNG = re.compile(r"^\s*-?\d{1,3}(?:\.\d+)?\s*[,;]\s*-?\d{1,3}(?:\.\d+)?\s*$")
_PHONE = re.compile(r"\+?\d[\d\s().\-]{8,}\d")
_CUT_MARKERS = re.compile(
    r"\b(?:tel(?:ephone)?|phone|mobile|fax|e-?mail|call us|opening (?:hours|times)|open (?:daily|today|now))\b",
    re.I,
)
_LEGAL_ADDRESS = re.compile(r"registered (?:office|address)|company (?:no|number)|vat (?:no|number|reg)", re.I)
_LEADING_LABEL = re.compile(
    r"^(?:our\s+)?(?:address|find us|visit us|location|locations|where to find us|headquarters|"
    r"head office|registered office|registered address|contact us)\s*[:\-–]?\s*",
    re.I,
)

_US_STATES = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA",
    "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK",
    "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY", "DC",
}

_COUNTRY_ALIASES = {
    "uk": "UK", "gb": "UK", "gbr": "UK", "united kingdom": "UK", "great britain": "UK", "england": "UK",
    "scotland": "UK", "wales": "UK", "northern ireland": "UK",
    "us": "US", "usa": "US", "united states": "US", "united states of america": "US", "u.s.": "US", "u.s.a.": "US",
    "ca": "Canada", "canada": "Canada", "au": "Australia", "australia": "Australia",
    "ie": "Ireland", "ireland": "Ireland", "de": "Germany", "germany": "Germany", "fr": "France", "france": "France",
    "es": "Spain", "spain": "Spain", "it": "Italy", "italy": "Italy", "nl": "Netherlands",
    "netherlands": "Netherlands", "the netherlands": "Netherlands", "be": "Belgium", "belgium": "Belgium",
    "pt": "Portugal", "portugal": "Portugal", "ch": "Switzerland", "switzerland": "Switzerland",
    "at": "Austria", "austria": "Austria", "se": "Sweden", "sweden": "Sweden", "dk": "Denmark",
    "denmark": "Denmark", "no": "Norway", "norway": "Norway", "fi": "Finland", "finland": "Finland",
    "pl": "Poland", "poland": "Poland", "ae": "UAE", "uae": "UAE", "united arab emirates": "UAE",
    "in": "India", "india": "India", "sg": "Singapore", "singapore": "Singapore", "nz": "New Zealand",
    "new zealand": "New Zealand", "za": "South Africa", "south africa": "South Africa", "bd": "Bangladesh",
    "bangladesh": "Bangladesh", "pk": "Pakistan", "pakistan": "Pakistan", "my": "Malaysia",
    "malaysia": "Malaysia", "mx": "Mexico", "mexico": "Mexico", "br": "Brazil", "brazil": "Brazil",
    "jp": "Japan", "japan": "Japan", "tr": "Turkey", "turkey": "Turkey", "gr": "Greece", "greece": "Greece",
}
# Bare two-letter codes are only trusted as a country when they arrive from
# structured data (addressCountry) — in free text "IN"/"NO"/"CA" are far more
# likely a state abbreviation or a stray word.
_TWO_LETTER_FREE_TEXT_OK = {"uk", "gb", "us"}

_TLD_COUNTRY = {
    ".co.uk": "UK", ".uk": "UK", ".ie": "Ireland", ".ca": "Canada", ".com.au": "Australia", ".au": "Australia",
    ".de": "Germany", ".fr": "France", ".es": "Spain", ".it": "Italy", ".nl": "Netherlands", ".be": "Belgium",
    ".pt": "Portugal", ".ch": "Switzerland", ".at": "Austria", ".se": "Sweden", ".dk": "Denmark",
    ".no": "Norway", ".fi": "Finland", ".pl": "Poland", ".ae": "UAE", ".in": "India", ".co.in": "India",
    ".sg": "Singapore", ".nz": "New Zealand", ".co.nz": "New Zealand", ".za": "South Africa",
    ".co.za": "South Africa", ".bd": "Bangladesh", ".com.bd": "Bangladesh",
}
_PHONE_PREFIX_COUNTRY = (
    ("+44", "UK"), ("+353", "Ireland"), ("+61", "Australia"), ("+64", "New Zealand"), ("+971", "UAE"),
    ("+91", "India"), ("+65", "Singapore"), ("+880", "Bangladesh"), ("+27", "South Africa"),
)

_STREET_SUFFIXES = {
    "street", "st", "road", "rd", "avenue", "ave", "lane", "ln", "drive", "dr", "way", "square", "sq", "plaza",
    "place", "pl", "gardens", "gdns", "court", "ct", "crescent", "cres", "terrace", "terr", "boulevard", "blvd",
    "highway", "hwy", "parade", "row", "walk", "mews", "close", "grove", "quay", "wharf", "passage", "arcade",
    "yard", "centre", "center", "mall", "building", "house", "tower", "strasse", "straße", "platz", "laan",
    "straat", "weg", "gade",
}
_LEADING_UNIT_WORDS = {"unit", "suite", "ste", "flat", "floor", "level", "shop", "room", "apt", "apartment", "block"}
_JUNK_WORDS = {
    "opening", "hours", "open", "closed", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
    "sunday", "mon", "tue", "tues", "wed", "thu", "thur", "thurs", "fri", "sat", "sun", "book", "booking",
    "reserve", "reservation", "reservations", "menu", "menus", "contact", "call", "tel", "phone", "email",
    "copyright", "rights", "reserved", "privacy", "cookie", "cookies", "policy", "terms", "directions", "map",
    "maps", "view", "get", "visit", "find", "follow", "subscribe", "newsletter", "login", "sign", "cart",
    "basket", "delivery", "order", "gift", "careers", "jobs", "press", "faq", "home", "about", "us", "our",
    "location", "locations", "address", "welcome", "click", "here", "more", "info", "information",
    # sentence fragments ("Rules is London", "Established in 1798") are never a city
    "is", "are", "was", "were", "has", "have", "had", "we", "you", "your", "established", "since", "founded",
    "oldest", "best", "famous", "restaurant", "restaurants", "hotel", "cafe", "bar", "pub", "kitchen", "grill",
    "rest", "team", "story", "history",
}
_UK_COUNTIES = {
    "cumbria", "devon", "cornwall", "kent", "essex", "surrey", "sussex", "east sussex", "west sussex",
    "hampshire", "dorset", "somerset", "norfolk", "suffolk", "lincolnshire", "yorkshire", "north yorkshire",
    "west yorkshire", "south yorkshire", "east yorkshire", "lancashire", "cheshire", "derbyshire",
    "nottinghamshire", "leicestershire", "northamptonshire", "warwickshire", "worcestershire", "herefordshire",
    "shropshire", "staffordshire", "gloucestershire", "oxfordshire", "buckinghamshire", "bedfordshire",
    "hertfordshire", "cambridgeshire", "berkshire", "wiltshire", "northumberland", "durham", "county durham",
    "tyne and wear", "merseyside", "west midlands", "isle of wight", "rutland", "powys", "gwynedd", "fife",
    "highland", "aberdeenshire", "perthshire", "lothian", "ayrshire", "lanarkshire", "dyfed", "gwent", "clwyd",
}
_NOT_A_CITY = {
    "uk", "united kingdom", "england", "scotland", "wales", "northern ireland", "great britain", "usa",
    "united states", "us", "canada", "australia", "ireland", "europe", "worldwide", "global", "online",
}
_CITY_ALIASES = {"greater london": "London", "city of london": "London", "greater manchester": "Manchester"}
_LOWER_CONNECTORS = {"on", "upon", "under", "the", "of", "le", "la", "de", "sur", "en", "in", "by", "next", "cum", "super"}

_SKIP_JSONLD_TYPES = {
    "person", "review", "event", "jobposting", "product", "offer", "article", "blogposting", "newsarticle",
    "webpage", "website", "imageobject", "recipe", "question", "answer", "comment", "videoobject",
    "aggregaterating", "rating",
}
_SKIP_JSONLD_KEYS = {"@context", "geo", "image", "logo", "sameAs", "potentialAction", "aggregateRating", "review"}

_SOURCE_RANK = {"jsonld": 0, "microdata": 0, "address_tag": 1, "maps": 2, "meta": 3, "text": 4, "extra": 5}
_CONFIDENCE_RANK = {"high": 0, "medium": 1, "low": 2}


def _collapse(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value).replace(" ", " ").replace("​", "")).strip()


def _normalize_country(value: Any, *, structured: bool) -> str:
    raw = _collapse(value)
    if not raw:
        return ""
    key = raw.lower().strip(". ")
    if len(key) <= 3 and not structured and key not in _TWO_LETTER_FREE_TEXT_OK:
        return ""
    if key in _COUNTRY_ALIASES:
        return _COUNTRY_ALIASES[key]
    if structured and 2 < len(raw) <= 30 and all(ch.isalpha() or ch in " -'." for ch in raw):
        return _smart_title(raw)
    return ""


def _smart_title(value: str) -> str:
    text = _collapse(value)
    if not text:
        return ""
    if not (text.isupper() or text.islower()):
        return text
    lowered = text.lower()
    first = True

    def _cap(match: re.Match) -> str:
        nonlocal first
        word = match.group(0)
        if not first and word in _LOWER_CONNECTORS:
            return word
        first = False
        return word.capitalize()

    return re.sub(r"(?<![’'])[^\W\d_]+", _cap, lowered)


def _clean_city(value: Any) -> str:
    text = _collapse(value)
    if not text:
        return ""
    text = re.sub(r"\b[A-Z]{1,2}\d[A-Z\d]?\b\s*$", "", text).strip(" .,-")
    if not text or any(ch.isdigit() for ch in text):
        return ""
    lowered = text.lower()
    if lowered in _CITY_ALIASES:
        return _CITY_ALIASES[lowered]
    if lowered in _NOT_A_CITY:
        return ""
    tokens = [t for t in re.split(r"[\s\-]+", lowered) if t]
    if not tokens or len(tokens) > 6 or not (2 <= len(text) <= 45):
        return ""
    if any(t.strip(".") in _JUNK_WORDS for t in tokens):
        return ""
    if tokens[0].strip(".") in _LEADING_UNIT_WORDS:
        return ""
    if tokens[-1].strip(".") in _STREET_SUFFIXES:
        return ""
    if not all(ch.isalpha() or ch in " -'’.&" for ch in text):
        return ""
    if sum(1 for ch in text if ch.isalpha()) < 2:
        return ""
    return _smart_title(text)


def _clean_address_string(value: Any) -> str:
    text = _collapse(value)
    if not text:
        return ""
    text = _LEADING_LABEL.sub("", text)
    cut = _CUT_MARKERS.search(text)
    if cut:
        text = text[: cut.start()]
    phone = _PHONE.search(text)
    if phone:
        text = text[: phone.start()]
    text = text.strip(" ,;|·•-–—:")
    if not (8 <= len(text) <= 200):
        return ""
    lowered = text.lower()
    if "@" in text or "http" in lowered or "www." in lowered or _LATLNG.match(text):
        return ""
    letters = sum(1 for ch in text if ch.isalpha())
    if letters < 4 or letters < len(text) * 0.35:
        return ""
    return text


def _format_label(city: str, region: str, country: str) -> str:
    if country == "US" and region and len(region) == 2:
        return f"{city}, {region.upper()}"
    if country:
        return f"{city}, {country}"
    return city


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "location"


def _norm_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _candidate_key(candidate: Dict[str, Any]) -> str:
    # Same branch seen via JSON-LD ("..., GB") and an <address> tag must collapse.
    street, postal = _norm_key(candidate.get("street") or ""), _norm_key(candidate.get("postal") or "")
    if street and postal:
        return f"{street}|{postal}"
    return _norm_key(candidate.get("full") or "")


def _strip_match(text: str, match: re.Match) -> str:
    return (text[: match.start()] + ", " + text[match.end():]).strip()


def _parse_free_text(raw: Any) -> Optional[Dict[str, str]]:
    text = _clean_address_string(raw)
    if not text:
        return None

    postal = region = country = ""
    match = _UK_PC.search(text)
    if match:
        postal = f"{match.group(1).upper()} {match.group(2).upper()}"
        country = "UK"
        text = _strip_match(text, match)
    else:
        match = _US_ZIP.search(text)
        if match and match.group(1) in _US_STATES:
            region, postal, country = match.group(1), match.group(2), "US"
            text = _strip_match(text, match)
        else:
            match = _AU_PC.search(text)
            if match:
                region, postal, country = match.group(1), match.group(2), "Australia"
                text = _strip_match(text, match)
            else:
                match = _CA_PC.search(text)
                if match:
                    postal, country = f"{match.group(1).upper()} {match.group(2).upper()}", "Canada"
                    text = _strip_match(text, match)

    parts = [p.strip(" .") for p in re.split(r"\s*[,;|•·\n]\s*|\s+[–—-]\s+", text) if p.strip(" .")]

    while parts:
        tail = parts[-1]
        detected = _normalize_country(tail, structured=False)
        if detected:
            country = country or detected
            parts.pop()
            continue
        words = tail.rsplit(" ", 1)
        if len(words) == 2:
            trailing = _normalize_country(words[1], structured=False)
            if trailing and _clean_city(words[0]):
                country = country or trailing
                parts[-1] = words[0]
        break

    if region and parts and parts[-1].upper() == region:
        parts.pop()

    city = ""
    city_index = -1
    for index in range(len(parts) - 1, -1, -1):
        part = parts[index]
        lead = re.match(r"^\d{4,6}\s+(.+)$", part)
        trail = re.match(r"^(.+?)\s+\d{4,6}$", part)
        if lead:
            postal = postal or part[: part.index(" ")]
            part = lead.group(1)
        elif trail:
            part = trail.group(1)
        candidate = _clean_city(part)
        if candidate:
            city, city_index = candidate, index
            break

    if city and city.lower() in _UK_COUNTIES:
        # "Windermere, Cumbria LA23 2EQ": the county is the last part, the town is the one before it
        for index in range(city_index - 1, -1, -1):
            earlier = _clean_city(parts[index])
            if earlier and earlier.lower() not in _UK_COUNTIES:
                city, city_index = earlier, index
                break

    if not city:
        return None
    street = ", ".join(parts[:city_index]).strip(" ,")
    return {"street": street, "city": city, "region": region, "postal": postal, "country": country, "full": _clean_address_string(raw)}


def _has_evidence(parsed: Dict[str, str]) -> bool:
    street = parsed.get("street", "")
    return bool(parsed.get("postal") or parsed.get("region") or street)


def _types(node: Dict[str, Any]) -> set:
    raw = node.get("@type")
    values = raw if isinstance(raw, list) else [raw]
    return {_collapse(v).lower() for v in values if v}


def _postal_to_candidate(addr: Dict[str, Any], name: str) -> Optional[Dict[str, Any]]:
    street = _collapse(addr.get("streetAddress"))
    locality = _collapse(addr.get("addressLocality"))
    region_raw = _collapse(addr.get("addressRegion"))
    postal = _collapse(addr.get("postalCode"))
    country_raw = addr.get("addressCountry")
    if isinstance(country_raw, list):
        country_raw = country_raw[0] if country_raw else ""
    if isinstance(country_raw, dict):
        country_raw = country_raw.get("name") or country_raw.get("@id") or ""
    country = _normalize_country(country_raw, structured=True)
    region = region_raw.upper() if len(region_raw) == 2 and region_raw.upper() in _US_STATES else ""

    full_parts = [p for p in (street, locality, region_raw, postal, _collapse(country_raw)) if p]
    full = _clean_address_string(", ".join(full_parts))
    city = _clean_city(locality)

    if not city:
        parsed = _parse_free_text(", ".join(p for p in (street, locality, region_raw, postal) if p)) if (street or locality) else None
        if not parsed:
            return {"full": full, "city": "", "source": "jsonld", "confidence": "low", "name": name} if full else None
        city = parsed["city"]
        postal = postal or parsed["postal"]
        country = country or parsed["country"]
        region = region or parsed["region"]
        confidence = "medium"
    else:
        confidence = "high" if (street or postal) else "medium"
        if not country:
            m = _UK_PC.search(postal or "")
            if m:
                country = "UK"
    return {
        "full": full or _clean_address_string(f"{street}, {city}"),
        "city": city, "region": region, "postal": postal, "country": country,
        "street": street, "source": "jsonld", "confidence": confidence, "name": name,
    }


def _walk_jsonld(node: Any, out: List[Dict[str, Any]], depth: int, budget: List[int]) -> None:
    if depth > 7 or budget[0] <= 0:
        return
    budget[0] -= 1
    if isinstance(node, list):
        for item in node:
            _walk_jsonld(item, out, depth + 1, budget)
        return
    if not isinstance(node, dict):
        return
    types = _types(node)
    if types & _SKIP_JSONLD_TYPES:
        return
    name = _collapse(node.get("name"))
    address = node.get("address")
    if "postaladdress" in types:
        candidate = _postal_to_candidate(node, "")
        if candidate:
            out.append(candidate)
    elif address:
        for addr in address if isinstance(address, list) else [address]:
            if isinstance(addr, dict):
                candidate = _postal_to_candidate(addr, name)
            else:
                parsed = _parse_free_text(addr)
                candidate = (
                    {**parsed, "source": "jsonld", "confidence": "medium" if _has_evidence(parsed) else "low", "name": name}
                    if parsed else None
                )
            if candidate:
                out.append(candidate)
    for key, value in node.items():
        if key == "address" or key in _SKIP_JSONLD_KEYS:
            continue
        if isinstance(value, (dict, list)):
            _walk_jsonld(value, out, depth + 1, budget)


def _microdata_candidates(soups: Iterable[Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for soup in soups:
        try:
            nodes = list(soup.find_all(attrs={"itemtype": re.compile(r"PostalAddress", re.I)})) + list(soup.select(".adr"))
        except Exception:
            continue
        for node in nodes[:20]:
            def _field(*names: str) -> str:
                for name in names:
                    found = node.find(attrs={"itemprop": name}) or node.select_one(f".{name}")
                    if found:
                        return _collapse(found.get("content") or found.get_text(" ", strip=True))
                return ""

            candidate = _postal_to_candidate(
                {
                    "streetAddress": _field("streetAddress", "street-address"),
                    "addressLocality": _field("addressLocality", "locality"),
                    "addressRegion": _field("addressRegion", "region"),
                    "postalCode": _field("postalCode", "postal-code"),
                    "addressCountry": _field("addressCountry", "country-name"),
                },
                "",
            )
            if candidate and candidate.get("city"):
                candidate["source"] = "microdata"
                out.append(candidate)
    return out


def _address_tag_candidates(soups: Iterable[Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for soup in soups:
        try:
            tags = soup.find_all("address")
        except Exception:
            continue
        for tag in tags[:30]:
            own_text = tag.get_text(", ", strip=True)
            sibling = tag.find_previous_sibling()
            context = f"{sibling.get_text(' ', strip=True)[:120] if sibling else ''} {own_text}"
            if _LEGAL_ADDRESS.search(context):
                continue
            parsed = _parse_free_text(own_text)
            if parsed:
                out.append({**parsed, "source": "address_tag", "confidence": "medium" if _has_evidence(parsed) else "low", "name": ""})
    return out


def _maps_candidates(soups: Iterable[Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for soup in soups:
        try:
            nodes = soup.find_all(["a", "iframe"])
        except Exception:
            continue
        for node in nodes[:400]:
            href = node.get("href") or node.get("src") or ""
            low = href.lower()
            if not any(x in low for x in ("maps.google", "goo.gl/maps", "google.com/maps")):
                continue
            query = ""
            try:
                q = parse_qs(urlparse(href).query).get("q")
                if q:
                    query = q[0]
                elif "place/" in href:
                    query = unquote(href.split("place/")[1].split("/")[0]).replace("+", " ")
            except Exception:
                continue
            parsed = _parse_free_text(query)
            if parsed and _has_evidence(parsed):
                out.append({**parsed, "source": "maps", "confidence": "medium", "name": ""})
    return out


def _meta_candidates(raw_meta: Dict[str, str]) -> List[Dict[str, Any]]:
    def _get(*keys: str) -> str:
        for key in keys:
            if raw_meta.get(key):
                return _collapse(raw_meta[key])
        return ""

    locality = _get("business:contact_data:locality", "og:locality", "place:location:locality", "geo.placename")
    city = _clean_city(locality)
    if not city:
        return []
    country = _normalize_country(
        _get("business:contact_data:country_name", "og:country-name", "place:location:country_name"), structured=True
    )
    street = _get("business:contact_data:street_address", "og:street-address")
    postal = _get("business:contact_data:postal_code", "og:postal-code")
    full = _clean_address_string(", ".join(p for p in (street, locality, postal) if p)) or city
    return [{
        "full": full, "city": city, "region": "", "postal": postal, "country": country, "street": street,
        "source": "meta", "confidence": "medium" if (street or postal) else "low", "name": "",
    }]


_COVERAGE_LEAD = re.compile(
    r"\b(?:in|across|throughout|serving|around|within|covering)\s+"
    r"((?:[A-Z][\w'’.\-]*(?:\s+[A-Z][\w'’.\-]*){0,2})"
    r"(?:\s*(?:,|&|and)\s*(?:[A-Z][\w'’.\-]*(?:\s+[A-Z][\w'’.\-]*){0,2}))*)"
)
_COVERAGE_NOT_PLACES = {
    "stock", "seconds", "minutes", "bulk", "style", "touch", "one", "love", "action", "business", "town", "season",
    "store", "stores", "person", "demand", "progress", "motion", "real", "time", "full", "total", "partnership",
    "minute", "hours", "days", "weeks", "months", "years", "need", "case", "order", "advance", "general", "particular",
    "private", "public", "house", "home", "line", "app", "cart", "bag", "front", "good", "great", "brief", "short",
    "trust", "control", "charge", "place", "use", "reach", "sync", "week", "month", "year", "today", "tonight",
}


def _coverage_candidates(texts: Iterable[str], raw_meta: Dict[str, str]) -> List[Dict[str, Any]]:
    """Cities named in a page title/description ("Online Grocery in Dhaka, Chattogram & Sylhet").

    A last-resort, low-confidence source for sites that publish no address at
    all (JS-rendered store locators, marketplaces) but still say where they
    operate in their headline. Only used when nothing better exists."""
    country = _normalize_country(
        raw_meta.get("og:country_name") or raw_meta.get("og:country-name") or raw_meta.get("place:location:country_name"),
        structured=True,
    )
    found: List[str] = []
    for text in texts:
        for match in _COVERAGE_LEAD.finditer(_collapse(text)):
            for part in re.split(r"\s*(?:,|&|\band\b)\s*", match.group(1)):
                part = part.strip(" .,-")
                if not part or part.lower() in _COVERAGE_NOT_PLACES:
                    continue
                if any(w.lower() in _COVERAGE_NOT_PLACES for w in part.split()):
                    continue
                city = _clean_city(part)
                if city and city.lower() not in {c.lower() for c in found}:
                    found.append(city)
    if not found or len(found) > 8:
        return []
    return [
        {"full": city, "city": city, "region": "", "postal": "", "country": country, "street": "",
         "source": "meta", "confidence": "low", "name": ""}
        for city in found
    ]


def _text_candidates(soups: Iterable[Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for soup in soups:
        try:
            lines = [ln.strip() for ln in soup.get_text("\n").splitlines() if ln.strip()]
        except Exception:
            continue
        for index, line in enumerate(lines):
            if len(line) > 160 or not (_UK_PC.search(line) or _US_ZIP.search(line) or _AU_PC.search(line) or _CA_PC.search(line)):
                continue
            if _LEGAL_ADDRESS.search(" ".join(lines[max(0, index - 2): index + 1])):
                continue
            previous = lines[index - 1] if index > 0 else ""
            combined = line
            if previous and len(previous) <= 90 and (previous[:1].isdigit() or previous.split(" ")[-1].lower().strip(".,") in _STREET_SUFFIXES):
                combined = f"{previous}, {line}"
            parsed = _parse_free_text(combined)
            if parsed and parsed["postal"]:
                out.append({**parsed, "source": "text", "confidence": "low", "name": ""})
            if len(out) >= 12:
                return out
    return out


def _extra_candidates(extra_addresses: Iterable[str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for raw in extra_addresses:
        parsed = _parse_free_text(raw)
        if parsed and parsed["postal"]:
            out.append({**parsed, "source": "extra", "confidence": "low", "name": ""})
    return out


def _infer_country(candidates: List[Dict[str, Any]], page_url: str, phones: Iterable[str]) -> str:
    known = {c["country"] for c in candidates if c.get("country")}
    if len(known) == 1:
        return next(iter(known))
    if known:
        return ""
    try:
        host = (urlparse(page_url if "//" in page_url else f"//{page_url}").hostname or "").lower()
    except Exception:
        host = ""
    for suffix in sorted(_TLD_COUNTRY, key=len, reverse=True):
        if host.endswith(suffix):
            return _TLD_COUNTRY[suffix]
    for phone in phones or []:
        compact = re.sub(r"[^\d+]", "", str(phone))
        for prefix, country in _PHONE_PREFIX_COUNTRY:
            if compact.startswith(prefix):
                return country
    return ""


def build_location_info(
    *,
    schemas: Iterable[Any],
    soups: Iterable[Any],
    raw_meta: Optional[Dict[str, str]] = None,
    extra_addresses: Iterable[str] = (),
    page_url: str = "",
    phones: Iterable[str] = (),
    headline_texts: Iterable[str] = (),
) -> Dict[str, Any]:
    soups = list(soups)
    candidates: List[Dict[str, Any]] = []

    jsonld: List[Dict[str, Any]] = []
    _walk_jsonld(list(schemas), jsonld, 0, [900])
    candidates.extend(jsonld)
    candidates.extend(_microdata_candidates(soups))
    candidates.extend(_address_tag_candidates(soups))
    candidates.extend(_maps_candidates(soups))
    candidates.extend(_meta_candidates(raw_meta or {}))

    def _has_city(items: List[Dict[str, Any]]) -> bool:
        return any(c.get("city") for c in items)

    # Weaker sources only ever fill a gap — they must never add a city next
    # to structured data, or a stray AI/legal address makes a one-location
    # business look multi-location.
    if not _has_city(candidates):
        candidates.extend(_text_candidates(soups))
    if not _has_city(candidates):
        candidates.extend(_extra_candidates(extra_addresses))

    if not _has_city(candidates):
        heading_texts: List[str] = list(headline_texts)
        for soup in soups[:1]:
            try:
                heading_texts.extend(t.get_text(" ", strip=True)[:160] for t in soup.find_all(["h1", "h2"], limit=4))
            except Exception:
                pass
        candidates.extend(_coverage_candidates(heading_texts, raw_meta or {}))

    inferred = _infer_country([c for c in candidates if c.get("city")], page_url, phones)
    for candidate in candidates:
        if candidate.get("city") and not candidate.get("country") and inferred:
            candidate["country"] = inferred

    groups: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    for index, candidate in enumerate(candidates):
        city = candidate.get("city")
        if not city:
            continue
        key = f"{city.lower()}|{(candidate.get('country') or '').lower()}"
        group = groups.get(key)
        if group is None:
            group = groups[key] = {
                "city": city, "region": candidate.get("region") or "", "country": candidate.get("country") or "",
                "members": [], "first": index,
            }
            order.append(key)
        group["members"].append(candidate)

    locations: List[Dict[str, Any]] = []
    for key in order:
        group = groups[key]
        members = group["members"]
        best = min(members, key=lambda c: (_CONFIDENCE_RANK[c["confidence"]], _SOURCE_RANK.get(c["source"], 9)))
        addresses = []
        seen = set()
        for member in members:
            norm = _candidate_key(member)
            if member.get("full") and norm not in seen:
                seen.add(norm)
                addresses.append(member["full"])
        region = next((m.get("region") for m in members if m.get("region")), "")
        postal = next((m.get("postal") for m in members if m.get("postal")), "")
        label = _format_label(group["city"], region, group["country"])
        locations.append({
            "id": _slug(label),
            "label": label,
            "city": group["city"],
            "region": region,
            "country": group["country"],
            "address": addresses[0] if addresses else "",
            "postalCode": postal,
            "branchCount": max(1, len(addresses)),
            "source": best["source"],
            "confidence": best["confidence"],
            "_key": key,
            "_rank": (_CONFIDENCE_RANK[best["confidence"]], _SOURCE_RANK.get(best["source"], 9), group["first"]),
        })

    locations.sort(key=lambda loc: loc["_rank"])
    locations = locations[:12]

    ordered: List[str] = []
    seen_addresses = set()

    def _push(value: str, norm: str) -> None:
        if value and norm and norm not in seen_addresses:
            seen_addresses.add(norm)
            ordered.append(value)

    for loc in locations:
        for member in groups[loc["_key"]]["members"]:
            _push(member.get("full") or "", _candidate_key(member))
    for candidate in candidates:
        _push(candidate.get("full") or "", _candidate_key(candidate))
    if not ordered:
        # Nothing parseable at all: keep the raw leftovers (stable order) so
        # "address found" never flips to "not found" versus the old behaviour.
        for raw in sorted({_clean_address_string(a) for a in extra_addresses}):
            _push(raw, _norm_key(raw))

    for loc in locations:
        loc.pop("_rank", None)
        loc.pop("_key", None)

    overall = "none"
    if locations:
        overall = min((loc["confidence"] for loc in locations), key=lambda c: _CONFIDENCE_RANK[c])
    return {
        "locations": locations,
        "isMultiLocation": len(locations) > 1,
        "locationConfidence": overall,
        "orderedAddresses": ordered[:25],
    }
