import json
import re
import asyncio
import sys
import os
from typing import Dict, List, Set, Any, Tuple
from urllib.parse import urlparse, unquote, urljoin, parse_qs
from bs4 import BeautifulSoup

try:
    from playwright.async_api import async_playwright
    PLAYWRIGHT_AVAILABLE = True
except (ImportError, Exception):
    PLAYWRIGHT_AVAILABLE = False
    print("WARNING: Playwright/Greenlet DLLs not found. Falling back to HTTPX (non-JS) scraping.")

import httpx
from models import ScrapeResult, Scores, ScoreBreakdown
import ai_agent

SOCIAL_PATTERNS = {
    'facebook':  re.compile(r'facebook\.com/(?!sharer|share|login|dialog)([\w.%-]+)', re.I),
    'instagram': re.compile(r'instagram\.com/([\w.%-]+)', re.I),
    'twitter':   re.compile(r'(?:twitter|x)\.com/([\w.%-]+)', re.I),
    'linkedin':  re.compile(r'linkedin\.com/(?:company|in)/([\w.%-]+)', re.I),
    'youtube':   re.compile(r'youtube\.com/(?:channel|c|user|@)([\w.%-]+)', re.I),
    'tiktok':    re.compile(r'tiktok\.com/@?([\w.%-]+)', re.I),
    'pinterest': re.compile(r'pinterest\.com/([\w.%-]+)', re.I),
}

AI_ENRICHMENT_CONFIDENCE_THRESHOLDS = {
    "emails": 72,
    "phones": 72,
    "addresses": 70,
    "openingHours": 70,
    "socialLinks": 68,
    "bookingPath": 68,
}

PHASE1_AI_VISION_TIMEOUT_SECONDS = int(os.getenv("PHASE1_AI_VISION_TIMEOUT_SECONDS", "45"))
PHASE1_AI_CONTACT_TIMEOUT_SECONDS = int(os.getenv("PHASE1_AI_CONTACT_TIMEOUT_SECONDS", "90"))
PHASE1_AI_ENRICH_TIMEOUT_SECONDS = int(os.getenv("PHASE1_AI_ENRICH_TIMEOUT_SECONDS", "180"))
PHASE1_ENABLE_CONTACT_FALLBACK = os.getenv("PHASE1_ENABLE_CONTACT_FALLBACK", "0").strip() == "1"

PHASE1_AI_DEBUG = True

PHONE_REGEX = re.compile(r'(?:\+?\d{1,3}[\s\-.]?)?(?:\(?\d{2,4}\)?[\s\-.]?)?\d{3,4}[\s\-.]?\d{3,4}(?:[\s\-.]?\d{2,4})?')
EMAIL_REGEX = re.compile(r'([a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,})')


def extract_emails_from_text(text: str) -> List[str]:
    if not text:
        return []
    normalized = str(text)
    # Handle common obfuscations like name [at] domain [dot] com
    normalized = re.sub(r'\s*\(at\)\s*|\s*\[at\]\s*|\s+at\s+', '@', normalized, flags=re.I)
    normalized = re.sub(r'\s*\(dot\)\s*|\s*\[dot\]\s*|\s+dot\s+', '.', normalized, flags=re.I)
    normalized = normalized.replace('mailto:', ' ')
    matches = EMAIL_REGEX.findall(normalized)
    return [m.strip().lower() for m in matches if m and m.strip()]

def clean_set(s: Set[str]) -> List[str]:
    return [x.strip() for x in s if x and x.strip()]

def normalise_phone(raw: str) -> str:
    return re.sub(r'\s+', ' ', raw).strip()

def get_grade(score: int) -> str:
    if score >= 90: return 'A+'
    if score >= 80: return 'A'
    if score >= 70: return 'B'
    if score >= 60: return 'C'
    if score >= 50: return 'D'
    return 'F'

def can_use_playwright_on_current_loop() -> bool:
    """
    Playwright launches a subprocess. On Windows this requires a loop that
    supports subprocess transports (typically ProactorEventLoop).
    """
    if not PLAYWRIGHT_AVAILABLE:
        return False
    if sys.platform != "win32":
        return True
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return True

    proactor_cls = getattr(asyncio, "ProactorEventLoop", None)
    if proactor_cls is None:
        return True
    return isinstance(loop, proactor_cls)


async def _fetch_with_playwright(target_url: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "content": "",
        "final_url": target_url,
        "used_playwright": False,
        "timed_out": False,
        "screenshot_bytes": None,
        "warnings": [],
    }

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, args=[
            '--disable-blink-features=AutomationControlled',
            '--disable-gpu',
            '--no-sandbox',
            '--disable-dev-shm-usage',
            '--disable-extensions',
        ])
        context = await browser.new_context(viewport={'width': 1280, 'height': 1600})
        await context.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")

        page = await context.new_page()

        async def route_intercept(route):
            if route.request.resource_type in ['media', 'font']:
                await route.abort()
            else:
                await route.continue_()

        await page.route("**/*", route_intercept)

        try:
            await page.goto(target_url, wait_until='domcontentloaded', timeout=25000)
        except Exception as e:
            if 'Timeout' in str(type(e)):
                result["timed_out"] = True
                result["warnings"].append('Page load timed out — results may be incomplete.')
            else:
                await browser.close()
                raise e

        if not result["timed_out"]:
            await asyncio.sleep(2.5)

        try:
            result["content"] = await page.content()
        except Exception:
            result["content"] = ""

        content_lower = result["content"].lower()
        cf_triggers = ['just a moment', 'cloudflare', '403 forbidden', '<title>403</title>', 'access denied', 'please enable cookies', 'cf-browser-verification', 'security check']

        if any(trigger in content_lower for trigger in cf_triggers):
            result["warnings"].append("Cloudflare / 403 block detected. Dropping Playwright context.")
            result["used_playwright"] = False
        else:
            result["final_url"] = page.url
            result["used_playwright"] = True
            # Screenshot capture disabled because vision extraction is not used.
            result["screenshot_bytes"] = None
            if result["final_url"] != target_url:
                result["warnings"].append(f"Redirected to {result['final_url']}")

        await browser.close()

    return result


def _run_playwright_in_thread(target_url: str) -> Dict[str, Any]:
    # Dedicated thread runner to avoid incompatible server loop restrictions on Windows.
    if sys.platform == "win32":
        try:
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        except Exception:
            pass
    return asyncio.run(_fetch_with_playwright(target_url))

async def head_check(url: str, timeout: float = 5.0) -> bool:
    try:
        async with httpx.AsyncClient(verify=False) as client:
            res = await client.head(url, timeout=timeout, follow_redirects=True)
            return res.status_code < 400
    except Exception:
        return False

async def fetch_with_httpx(url: str) -> Tuple[str, str]:
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5'
    }
    async with httpx.AsyncClient(verify=False, follow_redirects=True, timeout=30.0) as client:
        response = await client.get(url, headers=headers)
        if response.status_code in [403, 401, 429]:
            # Some WAFs target simulated browser UAs. Re-try natively.
            response = await client.get(url)
        return response.text, str(response.url)

async def scrape_website(url: str, enable_ai: bool = True, enable_deep_crawl: bool = True) -> dict:
    warnings = []

    target_url = url.strip()
    if not re.match(r'^https?://', target_url, re.I):
        target_url = f"https://{target_url}"

    has_ssl = target_url.startswith('https://')
    if not has_ssl:
        warnings.append('Site is not using HTTPS — this affects trust & SEO.')

    content = ""
    final_url = target_url
    timed_out = False
    used_playwright = False
    screenshot_bytes = None

    playwright_enabled = can_use_playwright_on_current_loop()
    if PLAYWRIGHT_AVAILABLE and not playwright_enabled:
        warnings.append("Playwright disabled on current Windows event loop. Falling back to HTTP fetch.")

    if playwright_enabled:
        try:
            pw_result = await _fetch_with_playwright(target_url)
            content = pw_result["content"]
            final_url = pw_result["final_url"]
            used_playwright = pw_result["used_playwright"]
            timed_out = pw_result["timed_out"]
            screenshot_bytes = pw_result["screenshot_bytes"]
            warnings.extend(pw_result["warnings"])
        except Exception as e:
            warnings.append(f"Playwright error: {str(e)[:100]}. Falling back to standard fetch.")
            print(f"Playwright Runtime Error: {e}")
    elif PLAYWRIGHT_AVAILABLE:
        # Try Playwright in a dedicated thread with its own Proactor loop on Windows.
        try:
            pw_result = await asyncio.to_thread(_run_playwright_in_thread, target_url)
            content = pw_result["content"]
            final_url = pw_result["final_url"]
            used_playwright = pw_result["used_playwright"]
            timed_out = pw_result["timed_out"]
            screenshot_bytes = pw_result["screenshot_bytes"]
            warnings.extend(pw_result["warnings"])
            if used_playwright:
                warnings.append("Playwright executed in dedicated worker thread mode.")
        except Exception as e:
            warnings.append(f"Playwright thread mode error: {str(e)[:100]}. Falling back to standard fetch.")
            print(f"Playwright Thread Runtime Error: {e}")

    if not used_playwright:
        try:
            content, final_url = await fetch_with_httpx(target_url)
            warnings.append("Note: JS-rendering was disabled or blocked. Dynamic content missing.")
        except Exception as e:
            raise Exception(f"Failed to fetch {target_url}: {str(e)}")

    is_protected_site = any("Cloudflare / 403 block detected" in str(w) for w in warnings)
    force_full_ai = os.getenv("PHASE1_FORCE_FULL_AI", "1").strip() == "1"
    if timed_out:
        warnings.append("Primary page fetch timed out; running reduced analysis mode.")
    if is_protected_site:
        warnings.append("Protected/WAF site detected; skipping deep crawl and AI enrichment to avoid long timeout.")

    soup = BeautifulSoup(content, 'html.parser')

    # Shallow Crawl for Contact/About Subpages
    subpage_contents = []
    crawl_queue = []
    base_domain = urlparse(final_url).netloc
    
    for a in soup.find_all('a', href=True):
        href = a['href']
        low = href.lower()
        if any(x in low for x in ['contact', 'about', 'location', 'find-us']):
            full_link = urljoin(final_url, href)
            try:
                if urlparse(full_link).netloc == base_domain and full_link not in crawl_queue and full_link != final_url:
                    crawl_queue.append(full_link)
            except: pass
            
    crawl_queue = crawl_queue[:3]
    if crawl_queue and enable_deep_crawl and not is_protected_site and not timed_out:
        async def fetch_sub(u):
            try:
                # First attempt with httpx (fast)
                txt, _ = await fetch_with_httpx(u)
                
                # Check if we hit a WAF
                low_txt = txt.lower()
                triggers = ['403 - forbidden', 'access denied', 'cloudflare', 'just a moment', 'security check', 'enable cookies']
                if any(t in low_txt for t in triggers):
                    if can_use_playwright_on_current_loop():
                        async with async_playwright() as pw:
                            b = await pw.chromium.launch(headless=True, args=['--disable-gpu', '--no-sandbox'])
                            p = await b.new_page()
                            await p.goto(u, wait_until='domcontentloaded', timeout=20000)
                            txt = await p.content()
                            await b.close()
                return txt
            except Exception as sub_e:
                print(f"Sub-crawl error for {u}: {sub_e}")
                return ""
        results = await asyncio.gather(*(fetch_sub(u) for u in crawl_queue))
        for r in results:
            if r:
                subpage_contents.append(BeautifulSoup(r, 'html.parser'))

    all_soups = [soup] + subpage_contents

    schemas = []
    for script in soup.find_all('script', type='application/ld+json'):
        try:
            parsed = json.loads(script.string or script.get_text() or '')
            if isinstance(parsed, list):
                schemas.extend(parsed)
            else:
                schemas.append(parsed)
        except:
            warnings.append('Could not parse a JSON-LD block — may be malformed.')

    flat_schemas = []
    for s in schemas:
        if isinstance(s, dict) and '@graph' in s and isinstance(s['@graph'], list):
            flat_schemas.extend(s['@graph'])
        else:
            flat_schemas.append(s)
    schemas = [s for s in flat_schemas if isinstance(s, dict)]

    raw_meta = {}
    for meta in soup.find_all('meta'):
        name = meta.get('name') or meta.get('property') or meta.get('itemprop') or ''
        content_attr = meta.get('content') or ''
        if name and content_attr:
            raw_meta[name] = content_attr

    schema_name = None
    for s in schemas:
        if isinstance(s, dict):
            if s.get('name'): schema_name = s.get('name'); break
            if s.get('legalName'): schema_name = s.get('legalName'); break

    title_text = soup.title.string if soup.title else None
    h1_tag = soup.find('h1')
    h1_text = h1_tag.get_text() if h1_tag else None

    business_name = (
        schema_name or 
        raw_meta.get('og:site_name') or 
        raw_meta.get('og:title') or 
        title_text or 
        h1_text or ""
    )
    business_name = re.sub(r'\s*[|\-–—]\s*.+$', '', business_name)
    business_name = re.sub(r'\s+(home|welcome|official site|official website)$', '', business_name, flags=re.I).strip()

    description = ""
    for s in schemas:
        if isinstance(s, dict) and s.get('description'):
            description = s.get('description')
            break
    if not description:
        description = raw_meta.get('description') or raw_meta.get('og:description') or raw_meta.get('twitter:description') or ''

    phones = set()
    for s_doc in all_soups:
        for a in s_doc.find_all('a', href=re.compile(r'^tel:')):
            raw = unquote(a['href'].replace('tel:', '')).strip()
            if raw: phones.add(normalise_phone(raw))

    for s in schemas:
        if isinstance(s, dict) and s.get('telephone'):
            t = s['telephone']
            if isinstance(t, list):
                for p in t: phones.add(normalise_phone(p))
            else:
                phones.add(normalise_phone(t))

    if len(phones) < 2:
        for s_doc in all_soups:
            for tag in s_doc.find_all(['footer', 'header', 'address']) + s_doc.select('.contact, [class*="contact"], [id*="contact"]'):
                text = tag.get_text()
                matches = PHONE_REGEX.findall(text)
                for m in matches:
                    clean = m.strip()
                    if len(re.sub(r'\D', '', clean)) >= 7:
                        phones.add(normalise_phone(clean))

    emails = set()
    for s_doc in all_soups:
        for a in s_doc.find_all('a', href=re.compile(r'^mailto:')):
            raw = unquote(a['href'].replace('mailto:', '')).split('?')[0].strip().lower()
            if raw: emails.add(raw)

    for s in schemas:
        if isinstance(s, dict) and s.get('email'):
            emails.add(str(s['email']).lower())

    if len(emails) < 2:
        for s_doc in all_soups:
            for tag in s_doc.find_all(['footer', 'header', 'address']) + s_doc.select('.contact, [class*="contact"], [id*="contact"]'):
                text = tag.get_text()
                matches = EMAIL_REGEX.findall(text)
                for m in matches:
                    emails.add(m.strip().lower())

    # Global fallback scan: some sites place email in overlays/menus that are outside contact blocks.
    if len(emails) == 0:
        for s_doc in all_soups:
            all_text = s_doc.get_text(" ", strip=True)
            for m in extract_emails_from_text(all_text):
                emails.add(m)

            for a in s_doc.find_all('a', href=True):
                href = a.get('href') or ''
                for m in extract_emails_from_text(href):
                    emails.add(m)

    addresses = set()
    for s in schemas:
        if not isinstance(s, dict): continue
        addr = s.get('address')
        if not addr: continue
        if isinstance(addr, str):
            addresses.add(addr.strip())
        elif isinstance(addr, dict):
            parts = [
                addr.get('streetAddress'),
                addr.get('addressLocality'),
                addr.get('addressRegion'),
                addr.get('postalCode'),
                addr.get('addressCountry')
            ]
            parts = [p for p in parts if p and isinstance(p, str)]
            if parts: addresses.add(', '.join(parts))

    for s_doc in all_soups:
        for tag in s_doc.find_all('address'):
            addresses.add(tag.get_text().strip())

        for a in s_doc.find_all(['a', 'iframe']):
            href = a.get('href') or a.get('src') or ''
            if any(x in href.lower() for x in ['maps.google', 'goo.gl/maps', 'google.com/maps']):
                try:
                    q = parse_qs(urlparse(href).query).get('q')
                    if q: 
                        addresses.add(unquote(q[0]))
                    elif 'place/' in href:
                        addr = href.split('place/')[1].split('/')[0]
                        addresses.add(unquote(addr).replace('+', ' '))
                except: pass

    # Deep Text Regex Fallback for Addresses (especially for UK/US formats)
    if not addresses:
        ADDRESS_SNIP_REGEX = re.compile(r'(\d+\s+[A-Za-z0-9\s,]{5,50}(?:Street|St|Road|Rd|Avenue|Ave|Lane|Ln|Drive|Dr|Way|Square|Sq|Plaza|Pl|Gardens|Gdns|London|Manchester|Birmingham|W1|EC1|SW1|E1|N1|NW1|SE1|WC1|[\s,]{1,2}[A-Z]{1,2}\d{1,2}\s+\d[A-Z]{2}))', re.I)
        for s_doc in all_soups:
            text = s_doc.get_text()
            # Look for common UK Postcode patterns at the end of snippets
            postcode_matches = re.finditer(r'([A-Z]{1,2}\d{1,2}[A-Z]?\s*\d[A-Z]{2})', text, re.I)
            for pm in postcode_matches:
                start = max(0, pm.start() - 100)
                snip = text[start:pm.end()]
                # Extract the last few lines/parts ending in this postcode
                parts = re.split(r'[\r\n]{1,2}', snip)
                potential = parts[-1].strip()
                if len(potential) > 10:
                    addresses.add(potential)
            
            # Explicit search for Broadwick St style addresses
            matches = ADDRESS_SNIP_REGEX.findall(text)
            for m in matches:
                if len(m.strip()) > 10:
                    addresses.add(m.strip())

    opening_hours = []
    for s in schemas:
        if not isinstance(s, dict): continue
        if s.get('openingHours'):
            oh = s['openingHours']
            if isinstance(oh, list): opening_hours.extend(oh)
            else: opening_hours.append(oh)
        
        if s.get('openingHoursSpecification'):
            specs = s['openingHoursSpecification']
            if not isinstance(specs, list): specs = [specs]
            for spec in specs:
                if not isinstance(spec, dict): continue
                days = spec.get('dayOfWeek', [])
                if not isinstance(days, list): days = [days]
                label = f"{', '.join([str(d) for d in days])}: {spec.get('opens', '?')} – {spec.get('closes', '?')}"
                opening_hours.append(label)

    social_links = {}
    for s_doc in all_soups:
        for a in s_doc.find_all('a', href=True):
            try:
                href = a['href']
                for platform, pattern in SOCIAL_PATTERNS.items():
                    if platform not in social_links and pattern.search(href):
                        social_links[platform] = href if href.startswith('http') else f"https://{href}"
            except: pass

    logo_url = None
    for s in schemas:
        if not isinstance(s, dict): continue
        if s.get('logo'):
            l = s['logo']
            if isinstance(l, dict): logo_url = l.get('url')
            elif isinstance(l, str): logo_url = l
            break
    if not logo_url:
        logo_url = raw_meta.get('og:image')

    html_tag = soup.find('html')
    language = (html_tag.get('lang') if html_tag else None) or raw_meta.get('language') or 'unknown'
    
    canonical_tag = soup.find('link', rel='canonical')
    canonical_url = canonical_tag.get('href') if canonical_tag else None
    
    has_mobile_meta = bool(soup.find('meta', attrs={'name': 'viewport'}))
    
    has_analytics = any(x in content for x in [
        'gtag(', 'ga(', '_gaq', 'fbq(', 'analytics.js', 'gtm.js', 'segment.io', 'hotjar.com'
    ])

    page_speed_hints = []
    if 'render-blocking' in content or len(soup.find_all('link', rel='stylesheet', media=False)) > 5:
        page_speed_hints.append('Multiple render-blocking stylesheets detected.')
    
    sync_scripts = [s for s in soup.find_all('script') if not s.get('async') and not s.get('defer') and s.get('src')]
    if len(sync_scripts) > 4:
        page_speed_hints.append('Several synchronous scripts may slow page load.')
    
    if not has_mobile_meta:
        page_speed_hints.append('Missing viewport meta tag — mobile experience may break.')

    # ----- ADVANCED SEO & CONTENT ANALYSIS -----
    h1_count = len(soup.find_all('h1'))
    h2_count = len(soup.find_all('h2'))
    h3_count = len(soup.find_all('h3'))
    
    internal_links = 0
    external_links = 0
    link_tags = soup.find_all('a', href=True)
    domain_loc = urlparse(target_url).netloc
    for link in link_tags:
        h = link['href']
        if h.startswith('http'):
            if domain_loc in h:
                internal_links += 1
            else:
                external_links += 1
        elif h.startswith('/') or h.startswith('#') or not h.startswith('mailto:') and not h.startswith('tel:'):
            internal_links += 1

    images = soup.find_all('img')
    images_count = len(images)
    images_without_alt = sum(1 for img in images if not img.get('alt') or not img.get('alt').strip())
    
    # Text Word Count (Body)
    clean_text = soup.body.get_text(separator=' ', strip=True) if soup.body else ''
    word_count = len(re.findall(r'\b\w+\b', clean_text))

    title_length = len(title_text) if title_text else 0
    desc_length = len(description)

    seo_info = {
        "titleLength": title_length,
        "descLength": desc_length,
        "h1Count": h1_count,
        "h2Count": h2_count,
        "h3Count": h3_count,
        "internalLinks": internal_links,
        "externalLinks": external_links,
        "imagesCount": images_count,
        "imagesWithoutAlt": images_without_alt,
        "wordCount": word_count
    }

    # Technology Detection (Basic)
    technologies = []
    if 'wp-content' in content or 'WordPress' in content: technologies.append('WordPress')
    if 'Shopify' in content or 'cdn.shopify.com' in content: technologies.append('Shopify')
    if 'react' in content.lower() or 'data-reactroot' in content: technologies.append('React')
    if 'next.js' in content.lower() or '__NEXT_DATA__' in content: technologies.append('Next.js')
    if 'cloudflare' in content.lower(): technologies.append('Cloudflare')
    if 'stripe.com' in content: technologies.append('Stripe')
    
    origin = f"{urlparse(target_url).scheme}://{urlparse(target_url).netloc}"
    sitemap_found, robots_txt_found = await asyncio.gather(
        head_check(f"{origin}/sitemap.xml"),
        head_check(f"{origin}/robots.txt")
    )

    ai_debug: Dict[str, Any] = {
        "ran": False,
        "calls": {
            "vision": False,
            "enrichment": False,
            "contact_fallback": False,
            "perplexity_extract": False,
        },
        "models": {
            "vision": None,
            "enrichment": None,
            "contact_fallback": None,
            "perplexity_extract": None,
        },
        "perplexity_extract": {
            "emails": [],
            "phones": [],
            "addresses": [],
            "openingHours": [],
            "confidence": {},
        },
        "confidence": {},
        "thresholds": AI_ENRICHMENT_CONFIDENCE_THRESHOLDS,
        "merged": {"emails": 0, "phones": 0, "addresses": 0, "openingHours": 0, "socialLinks": 0, "bookingPath": False},
        "blocked": {"emails": 0, "phones": 0, "addresses": 0, "openingHours": 0, "socialLinks": 0, "bookingPath": False},
        "reason": None,
    }

    allow_ai_enrichment = enable_ai and (force_full_ai or (not is_protected_site and not timed_out))
    if force_full_ai and (is_protected_site or timed_out):
        warnings.append("Full AI enrichment forced despite protected/slow page mode.")

    # Vision extraction removed by product decision.

    # Focused grounded fallback for missing contact details.
    if PHASE1_ENABLE_CONTACT_FALLBACK and allow_ai_enrichment and (not emails or not phones or not opening_hours):
        ai_debug["calls"]["contact_fallback"] = True
        print("[AI] Phase1 contact fallback started")
        try:
            contact_fallback = await asyncio.wait_for(
                ai_agent.get_phase1_contact_fallback(target_url, business_name),
                timeout=PHASE1_AI_CONTACT_TIMEOUT_SECONDS,
            )
            print("[AI] Phase1 contact fallback completed")
        except asyncio.TimeoutError:
            contact_fallback = {"emails": [], "phones": [], "addresses": [], "openingHours": [], "confidence": {}}
            warnings.append(f"AI contact fallback timed out after {PHASE1_AI_CONTACT_TIMEOUT_SECONDS}s; continuing with available extracted data.")
        except Exception as cf_err:
            contact_fallback = {"emails": [], "phones": [], "addresses": [], "openingHours": [], "confidence": {}}
            warnings.append(f"AI contact fallback failed: {str(cf_err)[:80]}")
        if isinstance(contact_fallback, dict):
            ai_debug["models"]["contact_fallback"] = contact_fallback.get("modelUsed")
        cf_conf = contact_fallback.get("confidence", {}) if isinstance(contact_fallback.get("confidence"), dict) else {}

        def cf_ok(key: str, threshold: int = 65) -> bool:
            raw = cf_conf.get(key, 0)
            return raw >= threshold if isinstance(raw, int) else False

        for e in contact_fallback.get("emails", []):
            e_str = str(e).strip().lower()
            if EMAIL_REGEX.match(e_str) and (cf_ok("emails") or e_str in content.lower()):
                emails.add(e_str)

        for p in contact_fallback.get("phones", []):
            p_str = normalise_phone(str(p))
            if len(re.sub(r'\D', '', p_str)) >= 7 and cf_ok("phones"):
                phones.add(p_str)

        for h in contact_fallback.get("openingHours", []):
            h_str = str(h).strip()
            if h_str and cf_ok("openingHours", 60):
                opening_hours.append(h_str)

        for a in contact_fallback.get("addresses", []):
            a_str = str(a).strip()
            if len(a_str) >= 10 and cf_ok("addresses", 60):
                addresses.add(a_str)

    # ----- AI TEXT ENRICHMENT (PHASE 1) -----
    ai_suggestions: Dict[str, str] = {}
    ai_booking_path = False
    try:
        if not allow_ai_enrichment:
            raise RuntimeError("AI enrichment skipped for reduced analysis mode")
        print("[AI] Phase1 enrichment started (Perplexity extract + GPT deep analyzer)")
        text_blocks = []
        for s_doc in all_soups:
            text_blocks.append(s_doc.get_text(" ", strip=True))
        text_excerpt = re.sub(r'\s+', ' ', " ".join(text_blocks)).strip()[:16000]

        current_data_snapshot = {
            "emails": clean_set(emails),
            "phones": clean_set(phones),
            "addresses": [x for x in addresses if x],
            "openingHours": [str(x) for x in opening_hours],
            "socialLinks": social_links,
        }

        perplexity_extract, deep_analysis = await asyncio.wait_for(
            asyncio.gather(
                ai_agent.get_phase1_perplexity_contact_extraction(
                    url=target_url,
                    business_name=business_name,
                    page_text_excerpt=text_excerpt,
                    current_data=current_data_snapshot,
                ),
                ai_agent.get_phase1_deep_analyzer(
                    url=target_url,
                    business_name=business_name,
                    page_text_excerpt=text_excerpt,
                    current_data=current_data_snapshot,
                ),
            ),
            timeout=PHASE1_AI_ENRICH_TIMEOUT_SECONDS,
        )

        if not isinstance(perplexity_extract, dict):
            perplexity_extract = {}
        if not isinstance(deep_analysis, dict):
            deep_analysis = {}

        ai_debug["ran"] = True
        ai_debug["calls"]["enrichment"] = True
        ai_debug["calls"]["perplexity_extract"] = True
        ai_debug["models"]["perplexity_extract"] = perplexity_extract.get("modelUsed")
        ai_debug["models"]["enrichment"] = deep_analysis.get("modelUsed")
        ai_debug["perplexity_extract"] = {
            "emails": [str(v) for v in (perplexity_extract.get("emails", []) or [])][:5],
            "phones": [str(v) for v in (perplexity_extract.get("phones", []) or [])][:5],
            "addresses": [str(v) for v in (perplexity_extract.get("addresses", []) or [])][:3],
            "openingHours": [str(v) for v in (perplexity_extract.get("openingHours", []) or [])][:5],
            "confidence": perplexity_extract.get("confidence", {}) if isinstance(perplexity_extract.get("confidence"), dict) else {},
        }

        confidence = perplexity_extract.get("confidence", {}) if isinstance(perplexity_extract.get("confidence"), dict) else {}
        ai_debug["confidence"] = confidence
        text_excerpt_lower = text_excerpt.lower()
        target_host = (urlparse(target_url).netloc or "").strip().lower().replace("www.", "")

        def email_domain_matches_target(email_value: str) -> bool:
            if "@" not in email_value:
                return False
            domain = email_value.split("@", 1)[1].strip().lower()
            return bool(domain) and (domain == target_host or domain.endswith(f".{target_host}"))

        def looks_like_opening_hours(value: str) -> bool:
            raw = str(value or "").strip().lower()
            if not raw:
                return False
            has_day = bool(re.search(r"\b(mon|tue|wed|thu|fri|sat|sun|monday|tuesday|wednesday|thursday|friday|saturday|sunday|daily|weekend)\b", raw))
            has_time = bool(re.search(r"\b\d{1,2}(:\d{2})?\s*(am|pm)?\b", raw))
            return has_day or has_time

        def conf_ok(key: str) -> bool:
            raw = confidence.get(key, 0)
            value = raw if isinstance(raw, int) else 0
            return value >= AI_ENRICHMENT_CONFIDENCE_THRESHOLDS[key]

        if conf_ok("phones"):
            for p in perplexity_extract.get("phones", []):
                p_str = normalise_phone(str(p))
                if len(re.sub(r'\D', '', p_str)) >= 7:
                    phones.add(p_str)
                    ai_debug["merged"]["phones"] += 1
        else:
            ai_debug["blocked"]["phones"] = len(perplexity_extract.get("phones", []))

        # Email fallback: if confidence metadata is missing/low, still accept emails that literally exist on page text.
        for e in perplexity_extract.get("emails", []):
            e_str = str(e).strip().lower()
            if not EMAIL_REGEX.match(e_str):
                continue
            if conf_ok("emails") or (e_str in text_excerpt_lower) or email_domain_matches_target(e_str):
                before = len(emails)
                emails.add(e_str)
                if len(emails) > before:
                    ai_debug["merged"]["emails"] += 1
            else:
                ai_debug["blocked"]["emails"] += 1

        if conf_ok("addresses"):
            for a in perplexity_extract.get("addresses", []):
                a_str = str(a).strip()
                if len(a_str) >= 10:
                    addresses.add(a_str)
                    ai_debug["merged"]["addresses"] += 1
        else:
            ai_debug["blocked"]["addresses"] = len(perplexity_extract.get("addresses", []))

        for h in perplexity_extract.get("openingHours", []):
            h_str = str(h).strip()
            if not h_str:
                continue
            if conf_ok("openingHours") or looks_like_opening_hours(h_str):
                opening_hours.append(h_str)
                ai_debug["merged"]["openingHours"] += 1
            else:
                ai_debug["blocked"]["openingHours"] += 1

        if conf_ok("socialLinks"):
            ai_social_links = perplexity_extract.get("socialLinks", {}) if isinstance(perplexity_extract.get("socialLinks"), dict) else {}
            for platform, link in ai_social_links.items():
                if platform not in social_links and isinstance(link, str) and link.strip():
                    social_links[platform] = link.strip()
                    ai_debug["merged"]["socialLinks"] += 1
        else:
            ai_social_links = perplexity_extract.get("socialLinks", {}) if isinstance(perplexity_extract.get("socialLinks"), dict) else {}
            ai_debug["blocked"]["socialLinks"] = len(ai_social_links)

        ai_booking_path = bool(perplexity_extract.get("hasBookingPath", False)) if conf_ok("bookingPath") else False
        ai_debug["merged"]["bookingPath"] = ai_booking_path
        ai_debug["blocked"]["bookingPath"] = bool(perplexity_extract.get("hasBookingPath", False)) and not ai_booking_path

        ai_suggestions = deep_analysis.get("suggestions", {}) if isinstance(deep_analysis.get("suggestions"), dict) else {}
        print("[AI] Phase1 enrichment completed")
    except asyncio.TimeoutError:
        print(f"Phase1 enrichment error: timeout while waiting for AI enrichment response ({PHASE1_AI_ENRICH_TIMEOUT_SECONDS}s)")
        ai_debug["reason"] = "AI enrichment timed out"
        warnings.append(f"AI enrichment timed out after {PHASE1_AI_ENRICH_TIMEOUT_SECONDS}s; returning core analysis data.")
    except Exception as enrich_error:
        print(f"Phase1 enrichment error: {enrich_error.__class__.__name__}: {str(enrich_error)}")
        ai_debug["reason"] = str(enrich_error)[:220]

    b_name_score = 8 if business_name else 0
    desc_score = 9 if description else 0
    logo_score = 4 if logo_url else 0
    lang_score = 4 if language != 'unknown' else 0
    core_identity_score = b_name_score + desc_score + logo_score + lang_score

    phone_score = 10 if len(phones) > 0 else 0
    email_score = 7 if len(emails) > 0 else 0
    address_score = 8 if len(addresses) > 0 else 0
    contact_score = phone_score + email_score + address_score

    body_text = soup.body.get_text().lower() if soup.body else ""
    hours_in_text = bool(re.search(r'\b(monday|opening hours|hours of operation|mon[\s\-–]fri)\b', body_text))
    hours_in_schema = len(opening_hours) > 0
    hours_visible = 8 if (hours_in_text or hours_in_schema) else 0
    hours_structured = 7 if hours_in_schema else 0
    operating_score = hours_visible + hours_structured

    social_score = min(len(social_links) * 2, 8) if social_links else 0
    has_booking = bool(soup.find_all('a', href=re.compile(r'book|appoint|reserv|calendly|acuity', re.I))) or ai_booking_path
    booking_score = 7 if has_booking else 0
    trust_score = social_score + booking_score

    schema_present = 4 if len(schemas) > 0 else 0
    has_business_type = any(
        isinstance(s.get('@type'), str) and re.search(r'business|organization|localbusiness|store|restaurant|service', s['@type'], re.I) 
        for s in schemas if isinstance(s, dict)
    )
    schema_type_score = 3 if has_business_type else 0
    schema_key_fields = 3 if (len(addresses) > 0 and len(phones) > 0 and business_name) else 0
    schema_score = schema_present + schema_type_score + schema_key_fields

    ssl_score = 3 if has_ssl else 0
    mobile_score = 2 if has_mobile_meta else 0
    canonical_score = 2 if canonical_url else 0
    sitemap_score = 2 if sitemap_found else 0
    robots_score = 1 if robots_txt_found else 0
    technical_score = ssl_score + mobile_score + canonical_score + sitemap_score + robots_score

    total_score = core_identity_score + contact_score + operating_score + trust_score + schema_score + technical_score

    if not business_name: warnings.append('No business name found — check og:site_name and JSON-LD.')
    if not description: warnings.append('No meta description found — hurts SEO.')
    if not phones: warnings.append('No phone number detected.')
    if not emails: warnings.append('No email address detected.')
    if not addresses: warnings.append('No address found — important for local SEO.')
    if not schemas: warnings.append('No JSON-LD / Schema.org markup found.')
    if not has_mobile_meta: warnings.append('Missing viewport meta tag.')
    if not canonical_url: warnings.append('No canonical URL — duplicate content risk.')
    if not sitemap_found: warnings.append('No sitemap.xml found at site root.')

    return {
        "url": target_url,
        "title": title_text or "",
        "businessName": business_name,
        "description": description,
        "emails": clean_set(emails),
        "phones": clean_set(phones),
        "addresses": [x for x in addresses if x],
        "socialLinks": social_links,
        "openingHours": [str(x) for x in opening_hours],
        "logoUrl": logo_url,
        "language": str(language),
        "canonicalUrl": canonical_url,
        "sitemapFound": sitemap_found,
        "robotsTxtFound": robots_txt_found,
        "hasSSL": has_ssl,
        "hasMobileMeta": has_mobile_meta,
        "hasAnalytics": has_analytics,
        "pageSpeedHints": page_speed_hints,
        "schemas": schemas,
        "scores": {
            "total": total_score,
            "grade": get_grade(total_score),
            "coreIdentity": {"total": core_identity_score, "businessName": b_name_score, "description": desc_score, "logo": logo_score, "language": lang_score},
            "contact": {"total": contact_score, "phone": phone_score, "email": email_score, "address": address_score},
            "operating": {"total": operating_score, "hoursVisible": hours_visible, "hoursStructured": hours_structured},
            "trust": {"total": trust_score, "socialLinks": social_score, "booking": booking_score},
            "schema": {"total": schema_score, "present": schema_present, "correctType": schema_type_score, "keyFields": schema_key_fields},
            "technical": {"total": technical_score, "ssl": ssl_score, "mobile": mobile_score, "canonical": canonical_score, "sitemap": sitemap_score, "robots": robots_score}
        },
        "rawMeta": raw_meta,
        "warnings": warnings,
        "seoInfo": seo_info,
        "technologies": technologies,
        "aiSuggestions": {str(k): str(v)[:180] for k, v in ai_suggestions.items() if str(k).strip() and str(v).strip()},
        "aiDebug": ai_debug if PHASE1_AI_DEBUG else None,
    }
