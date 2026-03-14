import json
import re
import asyncio
from typing import Dict, List, Set, Any
from urllib.parse import urlparse, unquote
from bs4 import BeautifulSoup
try:
    from playwright.async_api import async_playwright
    PLAYWRIGHT_AVAILABLE = True
except (ImportError, Exception):
    PLAYWRIGHT_AVAILABLE = False
    print("WARNING: Playwright/Greenlet DLLs not found. Falling back to HTTPX (non-JS) scraping.")

import httpx
from models import ScrapeResult, Scores, ScoreBreakdown

SOCIAL_PATTERNS = {
    'facebook':  re.compile(r'facebook\.com/(?!sharer|share|login|dialog)([\w.%-]+)', re.I),
    'instagram': re.compile(r'instagram\.com/([\w.%-]+)', re.I),
    'twitter':   re.compile(r'(?:twitter|x)\.com/([\w.%-]+)', re.I),
    'linkedin':  re.compile(r'linkedin\.com/(?:company|in)/([\w.%-]+)', re.I),
    'youtube':   re.compile(r'youtube\.com/(?:channel|c|user|@)([\w.%-]+)', re.I),
    'tiktok':    re.compile(r'tiktok\.com/@?([\w.%-]+)', re.I),
    'pinterest': re.compile(r'pinterest\.com/([\w.%-]+)', re.I),
}

PHONE_REGEX = re.compile(r'(?:\+?\d{1,3}[\s\-.]?)?(?:\(?\d{2,4}\)?[\s\-.]?)?\d{3,4}[\s\-.]?\d{3,4}(?:[\s\-.]?\d{2,4})?')
EMAIL_REGEX = re.compile(r'([a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,})')

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

async def head_check(url: str, timeout: float = 5.0) -> bool:
    try:
        async with httpx.AsyncClient(verify=False) as client:
            res = await client.head(url, timeout=timeout, follow_redirects=True)
            return res.status_code < 400
    except Exception:
        return False

async def fetch_with_httpx(url: str) -> tuple[str, str]:
    headers = {
        'User-Agent': 'Mozilla/5.0 (compatible; BusinessAuditBot/2.0; +https://example.com/bot)'
    }
    async with httpx.AsyncClient(verify=False, follow_redirects=True, timeout=30.0) as client:
        response = await client.get(url, headers=headers)
        return response.text, str(response.url)

async def scrape_website(url: str) -> dict:
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

    if PLAYWRIGHT_AVAILABLE:
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True, args=[
                    '--disable-gpu',
                    '--disable-dev-shm-usage',
                    '--no-sandbox',
                    '--disable-setuid-sandbox',
                    '--disable-extensions',
                ])
                context = await browser.new_context(
                    user_agent='Mozilla/5.0 (compatible; BusinessAuditBot/2.0; +https://example.com/bot)',
                    viewport={'width': 1280, 'height': 800}
                )
                page = await context.new_page()

                async def route_intercept(route):
                    if route.request.resource_type in ['image', 'media', 'font', 'stylesheet']:
                        await route.abort()
                    else:
                        await route.continue_()

                await page.route("**/*", route_intercept)

                try:
                    await page.goto(target_url, wait_until='domcontentloaded', timeout=25000)
                except Exception as e:
                    if 'Timeout' in str(type(e)):
                        timed_out = True
                        warnings.append('Page load timed out — results may be incomplete.')
                    else:
                        raise e

                if not timed_out:
                    await asyncio.sleep(2.5)

                content = await page.content()
                final_url = page.url
                used_playwright = True

                if final_url != target_url:
                    warnings.append(f'Redirected to {final_url}')

                await browser.close()
        except Exception as e:
            warnings.append(f"Playwright error: {str(e)[:100]}. Falling back to standard fetch.")
            print(f"Playwright Runtime Error: {e}")

    if not used_playwright:
        try:
            content, final_url = await fetch_with_httpx(target_url)
            warnings.append("Note: JS-rendering is disabled. Dynamic content may be missing.")
        except Exception as e:
            raise Exception(f"Failed to fetch {target_url}: {str(e)}")

    soup = BeautifulSoup(content, 'html.parser')

    schemas = []
    for script in soup.find_all('script', type='application/ld+json'):
        try:
            parsed = json.loads(script.string or '')
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
    for a in soup.find_all('a', href=re.compile(r'^tel:')):
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
        for tag in soup.find_all(['footer', 'header', 'address']) + soup.select('.contact, [class*="contact"], [id*="contact"]'):
            text = tag.get_text()
            matches = PHONE_REGEX.findall(text)
            for m in matches:
                clean = m.strip()
                if len(re.sub(r'\D', '', clean)) >= 7:
                    phones.add(normalise_phone(clean))

    emails = set()
    for a in soup.find_all('a', href=re.compile(r'^mailto:')):
        raw = unquote(a['href'].replace('mailto:', '')).split('?')[0].strip().lower()
        if raw: emails.add(raw)

    for s in schemas:
        if isinstance(s, dict) and s.get('email'):
            emails.add(str(s['email']).lower())

    if len(emails) < 2:
        for tag in soup.find_all(['footer', 'header', 'address']) + soup.select('.contact, [class*="contact"], [id*="contact"]'):
            text = tag.get_text()
            matches = EMAIL_REGEX.findall(text)
            for m in matches:
                emails.add(m.strip().lower())

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

    for tag in soup.find_all('address'):
        addresses.add(tag.get_text().strip())

    for a in soup.find_all(['a', 'iframe']):
        href = a.get('href') or a.get('src') or ''
        if 'maps.google' in href or 'goo.gl/maps' in href:
            try:
                q = urllib.parse.parse_qs(urllib.parse.urlparse(href).query).get('q')
                if q: addresses.add(unquote(q[0]))
            except: pass

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
    for a in soup.find_all('a', href=True):
        href = a['href']
        for platform, pattern in SOCIAL_PATTERNS.items():
            if platform not in social_links and pattern.search(href):
                social_links[platform] = href if href.startswith('http') else f"https://{href}"

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
    has_booking = bool(soup.find_all('a', href=re.compile(r'book|appoint|reserv|calendly|acuity', re.I)))
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
        "technologies": technologies
    }
