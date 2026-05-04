import asyncio
import re
import random
from typing import List, Dict, Optional
from bs4 import BeautifulSoup
import httpx
try:
    from playwright.async_api import async_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

from phase2_models import (
    MultiScanRequest,
    MultiScanResult,
    SourceResult,
    ExtractedEntityData,
    ConflictIssue,
    ConflictSource
)

# A simplified extractor that tries to grab generic info from any source URL
def extract_generic_entity_data(html: str, url: str) -> ExtractedEntityData:
    soup = BeautifulSoup(html, 'html.parser')
    
    # 1. Phone extraction
    text = soup.get_text(separator=' ', strip=True)
    phone_pattern = re.compile(r'(\+?44\s?7\d{3}\s?\d{6}|\+?44\s?1\d{2,3}\s?\d{6,7}|\b0\d{4}\s?\d{5,6}\b)')
    phones = phone_pattern.findall(text)
    primary_phone = phones[0].strip() if phones else None

    # 2. Schema check
    has_schema = bool(soup.find('script', type='application/ld+json'))
    
    # 3. Logo check (Standard + OG Tags)
    has_logo = False
    og_image = soup.find('meta', property='og:image')
    if og_image:
        has_logo = True
    else:
        for img in soup.find_all('img'):
            src = img.get('src', '').lower()
            alt = img.get('alt', '').lower()
            if 'logo' in src or 'logo' in alt:
                has_logo = True
                break
            
    # 4. Name extraction (Title or OG Title)
    og_title = soup.find('meta', property='og:title')
    if og_title and og_title.get('content'):
        name = og_title['content'].strip()
    else:
        name = soup.title.string.strip() if soup.title else url
    
    # 5. Hours extraction (very basic keyword search)
    hours_raw = None
    if bool(re.search(r'\b(monday|opening hours|hours of operation|mon[\s\-–]fri)\b', text.lower())):
        hours_raw = "Mentioned in text"

    return ExtractedEntityData(
        name=name,
        phone=primary_phone,
        has_schema=has_schema,
        has_logo=has_logo,
        hours_raw=hours_raw
    )

async def fetch_with_playwright(url: str) -> str:
    if not PLAYWRIGHT_AVAILABLE:
        raise Exception("Advanced Bypass (Playwright) not available in this environment.")
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            viewport={'width': 1280, 'height': 800}
        )
        page = await context.new_page()
        
        # Block heavy resources for speed
        await page.route("**/*", lambda route: route.abort() if route.request.resource_type in ["image", "media", "font"] else route.continue_())
        
        try:
            # Random delay to simulate human timing
            await asyncio.sleep(random.uniform(1.0, 3.0))
            await page.goto(url, wait_until="networkidle", timeout=30000)
            content = await page.content()
            return content
        finally:
            await browser.close()

async def fetch_and_extract(source: ConflictSource) -> SourceResult:
    # Layer 1: Standard HTTPX (Fast)
    used_advanced_bypass = False
    try:
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36'}
            response = await client.get(source.url, headers=headers)
            
            # If blocked, jump to Layer 2 Exception handler
            if response.status_code in [403, 400]:
                raise httpx.HTTPStatusError(f"Platform Shield Detected ({response.status_code})", request=response.request, response=response)
            
            response.raise_for_status()
            html_content = response.text
    except (httpx.HTTPStatusError, httpx.RequestError) as e:
        # Layer 2: Playwright Fallback (Resilient)
        if PLAYWRIGHT_AVAILABLE:
            try:
                html_content = await fetch_with_playwright(source.url)
                used_advanced_bypass = True
            except Exception as inner_e:
                error_msg = f"Dual-Layer Failure: {str(inner_e)}"
                return create_error_result(source, error_msg)
        else:
            return create_error_result(source, str(e))
    except Exception as e:
        return create_error_result(source, str(e))

    # Extraction
    data = extract_generic_entity_data(html_content, source.url)
    return SourceResult(
        source_name=source.name,
        url=source.url,
        status="Pending", 
        extracted_data=data,
        issues=[],
        used_advanced_bypass=used_advanced_bypass
    )

def create_error_result(source: ConflictSource, error_msg: str) -> SourceResult:
    insight_desc = f"AI Visibility Blocked: This platform is currently gating your content from machine-readability. Tech: {error_msg}"
    if "400" in error_msg or "403" in error_msg or "Shield" in error_msg:
         insight_desc = "Platform Shield Detected: This directory is actively blocking external AI agents from auditing your business profile. Our advanced bypass was also restricted, indicating extreme isolation."
         
    return SourceResult(
        source_name=source.name,
        url=source.url,
        status="Error",
        extracted_data=ExtractedEntityData(),
        issues=[ConflictIssue(
            field="Visibility", 
            severity="Critical", 
            primary_value="Open Audit", 
            source_value="Blocked/Gated", 
            description=insight_desc
        )]
    )

def diff_entities(primary: ExtractedEntityData, source: ExtractedEntityData) -> List[ConflictIssue]:
    issues = []
    
    # 1. Phone Conflict (Critical)
    if primary.phone and source.phone and primary.phone != source.phone:
        issues.append(ConflictIssue(
            field="phone",
            severity="Critical",
            primary_value=primary.phone,
            source_value=source.phone,
            description=f"Phone mismatch: Expected '{primary.phone}', found '{source.phone}'"
        ))
        
    # 2. Missing Phone (Warning)
    elif primary.phone and not source.phone:
         issues.append(ConflictIssue(
            field="phone",
            severity="Warning",
            primary_value=primary.phone,
            source_value="Not Found",
            description=f"Primary phone number '{primary.phone}' not found on this directory."
        ))

    # 3. Logo check (Info)
    if primary.has_logo and not source.has_logo:
         issues.append(ConflictIssue(
            field="branding",
            severity="Info",
            primary_value="Standard Logo",
            source_value="Missing/Non-Standard",
            description="Branding Inconsistency: The platform isn't using modern rich-snippet tags to display your logo, forcing AI agents to guess your brand identity."
        ))
        
    # 4. Schema check (Info)
    if not source.has_schema:
         issues.append(ConflictIssue(
            field="readability",
            severity="Info",
            primary_value="AI-Friendly (JSON-LD)",
            source_value="Incompatible Format",
            description="Structural Insight: This platform lacks the machine-readable Schema.org layer, meaning your business appears as 'unstructured text' to bots and search agents."
        ))

    return issues

async def run_multi_scan(request: MultiScanRequest) -> MultiScanResult:
    # 1. Fetch Primary URL
    primary_source = ConflictSource(name="Primary", url=request.primary_url)
    primary_result = await fetch_and_extract(primary_source)
    primary_data = primary_result.extracted_data
    
    # 2. Fetch all secondary URLs concurrently
    tasks = [fetch_and_extract(src) for src in request.sources]
    source_results: List[SourceResult] = await asyncio.gather(*tasks)
    
    total_issues = 0
    critical_issues = 0
    warning_issues = 0
    
    # 3. Diff and score
    for s_res in source_results:
        if s_res.status == "Error":
             total_issues += 1
             critical_issues += 1
             continue
             
        issues = diff_entities(primary_data, s_res.extracted_data)
        s_res.issues = issues
        
        has_crit = any(i.severity == "Critical" for i in issues)
        has_warn = any(i.severity == "Warning" for i in issues)
        
        if has_crit:
            s_res.status = "Critical"
        elif has_warn:
            s_res.status = "Warning"
        elif len(issues) > 0:
            s_res.status = "In-Sync (Info)"
        else:
            s_res.status = "In-Sync"
            
        total_issues += len(issues)
        critical_issues += sum(1 for i in issues if i.severity == "Critical")
        warning_issues += sum(1 for i in issues if i.severity == "Warning")
        
    return MultiScanResult(
        primary_url=request.primary_url,
        primary_data=primary_data,
        sources=source_results,
        total_issues=total_issues,
        critical_issues=critical_issues,
        warning_issues=warning_issues
    )
