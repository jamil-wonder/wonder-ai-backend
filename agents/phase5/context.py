import asyncio
import json
import httpx
from bs4 import BeautifulSoup

from .config import PHASE5_CONTEXT_CACHE_TTL_SEC, PHASE5_CONTEXT_FETCH_TIMEOUT_SEC
from .helpers import _extract_text_value, _normalize_url

_PAGE_CONTEXT_CACHE: dict[str, dict] = {}


async def _fetch_page_context(url: str) -> dict:
    """Fast lightweight HTTP fetch to extract business context for question generation."""
    normalized_url = _normalize_url(url)
    empty = {
        "name": "",
        "description": "",
        "category": "",
        "location": "",
        "services": [],
    }
    if not normalized_url:
        return empty

    now = asyncio.get_running_loop().time()
    cached = _PAGE_CONTEXT_CACHE.get(normalized_url)
    if cached and isinstance(cached.get("expires_at"), (int, float)) and cached["expires_at"] > now:
        return dict(cached.get("ctx") or empty)

    ctx = dict(empty)
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; WonderBot/1.0)"}
        timeout = max(3.0, PHASE5_CONTEXT_FETCH_TIMEOUT_SEC)
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            resp = await client.get(normalized_url, headers=headers)
            if resp.status_code >= 400:
                _PAGE_CONTEXT_CACHE[normalized_url] = {
                    "ctx": ctx,
                    "expires_at": now + PHASE5_CONTEXT_CACHE_TTL_SEC,
                }
                return ctx

        soup = BeautifulSoup(resp.text, "html.parser")

        title = soup.find("title")
        if title:
            ctx["name"] = title.get_text(strip=True)[:100]

        og_site = soup.find("meta", property="og:site_name")
        if og_site and og_site.get("content"):
            ctx["name"] = str(og_site.get("content", "")).strip()[:100]

        meta_desc = soup.find("meta", attrs={"name": "description"})
        if meta_desc and meta_desc.get("content"):
            ctx["description"] = str(meta_desc.get("content", "")).strip()[:300]

        for script in soup.find_all("script", type="application/ld+json"):
            raw = (script.string or script.get_text() or "").strip()
            if not raw:
                continue
            try:
                data = json.loads(raw)
            except Exception:
                continue

            json_objects = data if isinstance(data, list) else [data]
            for item in json_objects:
                if not isinstance(item, dict):
                    continue

                if not ctx["name"]:
                    ctx["name"] = _extract_text_value(item.get("name"))[:100]
                if not ctx["description"]:
                    ctx["description"] = _extract_text_value(item.get("description"))[:300]
                if not ctx["category"]:
                    ctx["category"] = _extract_text_value(item.get("@type"))[:60]

                if not ctx["location"]:
                    address = item.get("address", {})
                    if isinstance(address, dict):
                        locality = _extract_text_value(address.get("addressLocality"))
                        region = _extract_text_value(address.get("addressRegion"))
                        ctx["location"] = (locality or region)[:60]

                if not ctx["services"]:
                    services: list[str] = []
                    offer_catalog = item.get("hasOfferCatalog", {})
                    if isinstance(offer_catalog, dict):
                        elems = offer_catalog.get("itemListElement", [])
                        if isinstance(elems, list):
                            for elem in elems[:12]:
                                if isinstance(elem, dict):
                                    service_name = _extract_text_value(elem.get("name"))
                                    if service_name:
                                        services.append(service_name)

                    makes_offer = item.get("makesOffer", [])
                    if isinstance(makes_offer, list):
                        for offer in makes_offer[:12]:
                            if isinstance(offer, dict):
                                offered = offer.get("itemOffered")
                                name = _extract_text_value(offered if isinstance(offered, dict) else offer)
                                if name:
                                    services.append(name)

                    seen = set()
                    clean_services = []
                    for service in services:
                        token = service.strip()
                        key = token.lower()
                        if token and key not in seen:
                            seen.add(key)
                            clean_services.append(token[:60])
                    ctx["services"] = clean_services[:8]

        if not ctx["name"]:
            h1 = soup.find("h1")
            if h1:
                ctx["name"] = h1.get_text(" ", strip=True)[:100]

        if not ctx["description"]:
            p = soup.find("p")
            if p:
                ctx["description"] = p.get_text(" ", strip=True)[:300]

        if not ctx["services"]:
            headings = []
            for tag in soup.find_all(["h2", "h3"], limit=20):
                text = tag.get_text(" ", strip=True)
                if text and 3 <= len(text) <= 60:
                    headings.append(text)
            ctx["services"] = list(dict.fromkeys(headings))[:6]

    except Exception as e:
        print(f"[Phase5] context fetch failed for {normalized_url}: {e}")

    _PAGE_CONTEXT_CACHE[normalized_url] = {
        "ctx": ctx,
        "expires_at": now + PHASE5_CONTEXT_CACHE_TTL_SEC,
    }
    return ctx
