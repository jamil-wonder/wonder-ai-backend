import os
import json
import re
import asyncio
import base64
import httpx
from google import genai
from google.genai import types
from dotenv import load_dotenv
from gemini_utils import generate_with_fallback

load_dotenv()

client = genai.Client()


def _openai_model_name() -> str:
    return (os.getenv("OPENAI_MODEL_PHASE1") or os.getenv("OPENAI_MODEL") or "gpt-4o-mini").strip()


def _openai_api_key() -> str:
    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured")
    return api_key


async def _openai_chat_json(
    *,
    prompt: str,
    timeout_seconds: int = 60,
    temperature: float = 0.1,
    max_tokens: int = 2500,
) -> tuple[dict, str]:
    api_key = _openai_api_key()
    model = _openai_model_name()
    payload = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": "Return valid JSON only."},
            {"role": "user", "content": prompt},
        ],
    }
    async with httpx.AsyncClient(timeout=timeout_seconds) as http:
        res = await http.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        res.raise_for_status()
        data = res.json()
    content = (((data.get("choices") or [{}])[0].get("message") or {}).get("content") or "").strip()
    parsed = _safe_json_parse(content)
    return (parsed if isinstance(parsed, dict) else {}), model


async def _openai_vision_json(
    *,
    prompt: str,
    image_bytes: bytes,
    timeout_seconds: int = 60,
    max_tokens: int = 1800,
) -> tuple[dict, str]:
    api_key = _openai_api_key()
    model = _openai_model_name()
    image_b64 = base64.b64encode(image_bytes).decode("ascii")
    payload = {
        "model": model,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                    },
                ],
            }
        ],
    }
    async with httpx.AsyncClient(timeout=timeout_seconds) as http:
        res = await http.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        res.raise_for_status()
        data = res.json()
    content = (((data.get("choices") or [{}])[0].get("message") or {}).get("content") or "").strip()
    parsed = _safe_json_parse(content)
    return (parsed if isinstance(parsed, dict) else {}), model


async def _generate_with_fallback_async(*, contents, config=None):
    # Google SDK call is synchronous; run in a worker thread to keep FastAPI event loop responsive.
    return await asyncio.to_thread(
        generate_with_fallback,
        client,
        contents=contents,
        config=config,
    )


def _model_used(response) -> str:
    model = getattr(response, "_model_used", None) if response is not None else None
    return model if isinstance(model, str) and model.strip() else "unknown"


def _normalize_insight_payload(payload: dict, model_name: str) -> dict:
    summary_raw = str(payload.get("summary") or "No data returned.").strip()
    summary = re.sub(r"\s+", " ", summary_raw)
    if len(summary) > 900:
        summary = summary[:897].rstrip() + "..."

    platforms = payload.get("platforms", [])
    if not isinstance(platforms, list):
        platforms = []
    platforms = [str(p).strip() for p in platforms if str(p).strip()][:8]

    evidence = payload.get("evidence", [])
    if not isinstance(evidence, list):
        evidence = []
    cleaned_evidence = []
    for item in evidence:
        text = re.sub(r"\s+", " ", str(item or "")).strip()
        if not text:
            continue
        if len(text) > 240:
            text = text[:237].rstrip() + "..."
        cleaned_evidence.append(text)
        if len(cleaned_evidence) >= 8:
            break

    return {
        "modelName": model_name,
        "isKnown": bool(payload.get("isKnown", False)),
        "summary": summary,
        "sentiment": str(payload.get("sentiment") or "Unknown"),
        "platforms": cleaned_platforms if (cleaned_platforms := platforms) else [],
        "evidence": cleaned_evidence,
    }

async def get_ai_insights(business_name: str, url: str) -> dict:
    """
    Given a business name and url (extracted from scraping),
    ask the Google Gemini model what it knows about this business online.
    Returns structured dict data expecting: modelName, isKnown, summary, sentiment, platforms, evidence.
    """
    prompt = f"""
    You are an AI research assistant. A user has a local business or website and wants to know what you and the internet know about it.
    The business name is "{business_name}" and its website is "{url}".

    Please provide a balanced, medium-detail summary of what is known about this business from major platforms like Google, Reddit, YouTube, Wikipedia, Google News, or other public sources based on your training data.

    You MUST respond in valid JSON format matching exactly this structure:
    {{
        "modelName": "Gemini",
        "isKnown": true or false,
        "summary": "3-4 sentences summarizing what is known.",
        "sentiment": "Positive" | "Neutral" | "Negative" | "Mixed" | "Unknown",
        "platforms": ["Google", "Reddit", "Facebook", "Wikipedia"],
        "evidence": ["4-8 bullet points with specific external facts or mentions"]
    }}

    Rules:
    - Keep summary and evidence similar depth to a professional analyst brief.
    - Prefer externally verifiable mentions over vague statements.
    - Return exactly 4-8 evidence bullets.
    - JSON only.
    """

    try:
        response = await _generate_with_fallback_async(
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[{"google_search": {}}],
                temperature=0.2,
                top_p=0.95,
                max_output_tokens=2048,
            ),
        )
        if response and response.text:
            parsed = _safe_json_parse(response.text)
            if isinstance(parsed, dict):
                return _normalize_insight_payload(parsed, _model_used(response))
            return _normalize_insight_payload({
                "isKnown": False,
                "summary": "Invalid data returned.",
                "sentiment": "Unknown",
                "platforms": [],
                "evidence": [],
            }, _model_used(response))
        else:
            return _normalize_insight_payload({
                "isKnown": False,
                "summary": "No data returned.",
                "sentiment": "Unknown",
                "platforms": [],
                "evidence": [],
            }, "Gemini")
    except Exception as e:
        print(f"Error fetching AI insights: {e}")
        return _normalize_insight_payload({
            "isKnown": False,
            "summary": "Failed to fetch AI insights.",
            "sentiment": "Unknown",
            "platforms": [],
            "evidence": [],
        }, "Gemini")


async def get_ai_insights_openai(business_name: str, url: str) -> dict:
    prompt = f"""
    You are an AI research assistant. A user has a local business or website and wants to know what is known online.
    The business name is "{business_name}" and website is "{url}".

        Respond in valid JSON with this exact shape:
    {{
      "modelName": "ChatGPT",
      "isKnown": true or false,
            "summary": "3-4 sentences summarizing what is known.",
      "sentiment": "Positive" | "Neutral" | "Negative" | "Mixed" | "Unknown",
      "platforms": ["Google", "Reddit", "YouTube", "Wikipedia"],
            "evidence": ["4-8 bullet points with specific external facts or mentions"]
    }}

        Rules:
        - Keep summary and evidence at medium detail, matching a concise analyst brief.
        - Prefer externally verifiable mentions over generic claims.
        - Return exactly 4-8 evidence bullets.
        - JSON only.
    """
    try:
        parsed, model = await _openai_chat_json(
            prompt=prompt,
            timeout_seconds=90,
            temperature=0.2,
            max_tokens=1800,
        )
        if not isinstance(parsed, dict):
            parsed = {}
        return _normalize_insight_payload(parsed, model)
    except Exception as e:
        print(f"Error fetching OpenAI insights: {e}")
        return _normalize_insight_payload({
            "isKnown": False,
            "summary": "Failed to fetch AI insights.",
            "sentiment": "Unknown",
            "platforms": [],
            "evidence": [],
        }, "gpt-4o-mini")


async def get_ai_insights_multi(business_name: str, url: str) -> list[dict]:
    gemini_task = get_ai_insights(business_name, url)
    gpt_task = get_ai_insights_openai(business_name, url)
    results = await asyncio.gather(gemini_task, gpt_task, return_exceptions=True)
    insights: list[dict] = []
    for r in results:
        if isinstance(r, dict):
            insights.append(r)
    return insights

async def get_vision_extraction(screenshot_bytes: bytes) -> dict:
    """
    Given a screenshot of a website as bytes, ask Google Gemini to extract 
    contact information and operating hours using Vision capabilities.
    """
    prompt = """
    You are an AI data extraction agent. Analyze this screenshot of a business website.
    Extract the following information if perfectly visible and identifiable:
    1. Phone numbers
    2. Email addresses
    3. Physical addresses
    4. Opening / Operating hours

    You MUST respond in valid JSON format matching exactly this structure:
    {
        "phones": [],
        "emails": [],
        "addresses": [],
        "hours": []
    }
    If a category is completely empty, or not explicitly found in the image, return an empty array for it. Do NOT hallucinate data. Make the hours array easy to read, e.g. ["Mon-Fri: 9am-5pm", "Sat: 10am-4pm"].
    """

    try:
        parsed, model = await _openai_vision_json(
            prompt=prompt,
            image_bytes=screenshot_bytes,
            timeout_seconds=75,
            max_tokens=1800,
        )
        if isinstance(parsed, dict) and parsed:
            parsed["modelUsed"] = model
            return parsed
        return {"phones": [], "emails": [], "addresses": [], "hours": []}
    except Exception as e:
        print(f"Error extracting vision data: {e}")
        return {"phones": [], "emails": [], "addresses": [], "hours": []}


def _safe_json_parse(text: str) -> dict:
    payload = (text or "").strip()
    if payload.startswith("```json"):
        payload = payload[7:]
    if payload.endswith("```"):
        payload = payload[:-3]
    payload = payload.strip()
    try:
        return json.loads(payload)
    except Exception:
        start = payload.find("{")
        end = payload.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(payload[start:end + 1])
            except Exception:
                return {}
    return {}


async def get_phase1_enrichment(
    url: str,
    business_name: str,
    page_text_excerpt: str,
    current_data: dict,
) -> dict:
    """
    Enrich Phase 1 extraction and suggestions from visible page text.
    Returns a compact JSON object that can be merged with scraper results.
    """
    compact_text = re.sub(r"\s+", " ", page_text_excerpt or "").strip()[:16000]
    seed = {
        "emails": current_data.get("emails", [])[:8],
        "phones": current_data.get("phones", [])[:8],
        "addresses": current_data.get("addresses", [])[:6],
        "openingHours": current_data.get("openingHours", [])[:8],
        "socialLinks": current_data.get("socialLinks", {}),
    }

    prompt = f"""
    You are a web data extraction and optimization assistant.
    Use only the provided website text excerpt and known extracted fields.

    URL: {url}
    Business name: {business_name}
    Current extracted data: {json.dumps(seed)}

    Visible page text excerpt:
    {compact_text}

    Tasks:
    1) Find any missed contact and operating details from the text.
    2) Provide short, actionable recommendations for each score label below.

    Output strict JSON only with this exact shape:
    {{
      "emails": ["contact@example.com"],
      "phones": ["+44 20 7946 0958"],
      "addresses": ["221B Baker Street, London NW1 6XE"],
      "openingHours": ["Mon-Fri: 9:00-18:00"],
      "socialLinks": {{"instagram": "https://instagram.com/example"}},
      "hasBookingPath": false,
            "confidence": {{
                "emails": 0,
                "phones": 0,
                "addresses": 0,
                "openingHours": 0,
                "socialLinks": 0,
                "bookingPath": 0
            }},
      "suggestions": {{
        "Business name": "...",
        "Description": "...",
        "Logo": "...",
        "Language": "...",
        "Phone": "...",
        "Email": "...",
        "Address": "...",
        "Hours visible": "...",
        "Hours in schema": "...",
        "Social links": "...",
        "Booking path": "...",
        "Schema present": "...",
        "Correct type": "...",
        "Key fields": "...",
        "HTTPS": "...",
        "Mobile": "...",
        "Canonical": "...",
        "Sitemap": "...",
        "Robots": "..."
      }}
    }}

    Rules:
    - Do not invent facts that are not implied by the provided text.
    - Keep suggestion strings concise (max 150 chars each).
    - confidence values must be integers 0..100.
    - Return empty arrays/objects when unknown.
    - JSON only, no markdown.
    """

    try:
        parsed, model = await _openai_chat_json(
            prompt=prompt,
            timeout_seconds=180,
            temperature=0.1,
            max_tokens=3500,
        )
        if not isinstance(parsed, dict):
            return {
                "emails": [],
                "phones": [],
                "addresses": [],
                "openingHours": [],
                "socialLinks": {},
                "hasBookingPath": False,
                "suggestions": {},
                "modelUsed": model,
            }
        return {
            "emails": parsed.get("emails", []) if isinstance(parsed.get("emails"), list) else [],
            "phones": parsed.get("phones", []) if isinstance(parsed.get("phones"), list) else [],
            "addresses": parsed.get("addresses", []) if isinstance(parsed.get("addresses"), list) else [],
            "openingHours": parsed.get("openingHours", []) if isinstance(parsed.get("openingHours"), list) else [],
            "socialLinks": parsed.get("socialLinks", {}) if isinstance(parsed.get("socialLinks"), dict) else {},
            "hasBookingPath": bool(parsed.get("hasBookingPath", False)),
            "confidence": parsed.get("confidence", {}) if isinstance(parsed.get("confidence"), dict) else {},
            "suggestions": parsed.get("suggestions", {}) if isinstance(parsed.get("suggestions"), dict) else {},
            "modelUsed": model,
        }
    except Exception as e:
        print(f"Error in phase1 enrichment: {e}")
        return {
            "emails": [],
            "phones": [],
            "addresses": [],
            "openingHours": [],
            "socialLinks": {},
            "hasBookingPath": False,
            "confidence": {},
            "suggestions": {},
        }


async def get_phase1_contact_fallback(url: str, business_name: str) -> dict:
    """
    Focused fallback extractor for contact details using Gemini with Google Search grounding.
    Used only when core scraper still misses contact fields.
    """
    prompt = f"""
    Extract business contact details for this website using grounded web retrieval.

    Website: {url}
    Business: {business_name}

    Return strict JSON only:
    {{
      "emails": ["contact@example.com"],
      "phones": ["+44 20 7946 0958"],
      "addresses": ["221B Baker Street, London NW1 6XE"],
      "openingHours": ["Mon-Fri: 09:00-18:00"],
      "confidence": {{
        "emails": 0,
        "phones": 0,
        "addresses": 0,
        "openingHours": 0
      }}
    }}

    Rules:
    - Prefer details found on the target domain itself.
    - Do not fabricate values.
    - Confidence values are integers 0..100.
    - JSON only.
    """

    try:
        parsed, model = await _openai_chat_json(
            prompt=prompt,
            timeout_seconds=90,
            temperature=0.0,
            max_tokens=1800,
        )
        if not isinstance(parsed, dict):
            return {"emails": [], "phones": [], "addresses": [], "openingHours": [], "confidence": {}, "modelUsed": model}
        return {
            "emails": parsed.get("emails", []) if isinstance(parsed.get("emails"), list) else [],
            "phones": parsed.get("phones", []) if isinstance(parsed.get("phones"), list) else [],
            "addresses": parsed.get("addresses", []) if isinstance(parsed.get("addresses"), list) else [],
            "openingHours": parsed.get("openingHours", []) if isinstance(parsed.get("openingHours"), list) else [],
            "confidence": parsed.get("confidence", {}) if isinstance(parsed.get("confidence"), dict) else {},
            "modelUsed": model,
        }
    except Exception as e:
        print(f"Error in phase1 contact fallback: {e}")
        return {"emails": [], "phones": [], "addresses": [], "openingHours": [], "confidence": {}}


