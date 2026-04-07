import os
import json
import re
import asyncio
from google import genai
from google.genai import types
from dotenv import load_dotenv
from gemini_utils import generate_with_fallback

load_dotenv()

client = genai.Client()


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

async def get_ai_insights(business_name: str, url: str) -> dict:
    """
    Given a business name and url (extracted from scraping),
    ask the Google Gemini model what it knows about this business online.
    Returns structured dict data expecting: modelName, isKnown, summary, sentiment, platforms, evidence.
    """
    prompt = f"""
    You are an AI research assistant. A user has a local business or website and wants to know what you and the internet know about it.
    The business name is "{business_name}" and its website is "{url}".

    Please provide a concise but comprehensive summary of what is known about this business from major platforms like Google, Reddit, YouTube, Wikipedia, Google News, or other public sources based on your training data.

    You MUST respond in valid JSON format matching exactly this structure:
    {{
        "modelName": "Gemini",
        "isKnown": true or false,
        "summary": "2-3 sentences summarizing what is known.",
        "sentiment": "Positive" | "Neutral" | "Negative" | "Mixed" | "Unknown",
        "platforms": ["Google", "Reddit", "Facebook", "Wikipedia"],
        "evidence": ["Bullet point 1 detailing a specific fact or mention", "Bullet point 2"]
    }}
    """

    try:
        response = await _generate_with_fallback_async(
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[{"google_search": {}}],
                response_mime_type="application/json",
                temperature=0.2,
                top_p=0.95,
                max_output_tokens=2048,
            ),
        )
        if response and response.text:
            parsed = json.loads(response.text)
            if isinstance(parsed, dict):
                parsed["modelName"] = _model_used(response)
                return parsed
            return {
                "modelName": _model_used(response), "isKnown": False, "summary": "Invalid data returned.",
                "sentiment": "Unknown", "platforms": [], "evidence": []
            }
        else:
            return {
                "modelName": "Gemini", "isKnown": False, "summary": "No data returned.",
                "sentiment": "Unknown", "platforms": [], "evidence": []
            }
    except Exception as e:
        print(f"Error fetching AI insights: {e}")
        return {
            "modelName": "Gemini", "isKnown": False, "summary": "Failed to fetch AI insights.",
            "sentiment": "Unknown", "platforms": [], "evidence": []
        }

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
        response = await _generate_with_fallback_async(
            contents=[
                prompt,
                types.Part.from_bytes(
                    data=screenshot_bytes,
                    mime_type="image/jpeg"
                )
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1,
                top_p=0.95,
                max_output_tokens=2048,
            ),
        )
        if response and response.text:
            parsed = json.loads(response.text)
            if isinstance(parsed, dict):
                parsed["modelUsed"] = _model_used(response)
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
        response = await _generate_with_fallback_async(
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1,
                top_p=0.9,
                max_output_tokens=3500,
            ),
        )
        parsed = _safe_json_parse(response.text if response else "")
        if not isinstance(parsed, dict):
            return {
                "emails": [],
                "phones": [],
                "addresses": [],
                "openingHours": [],
                "socialLinks": {},
                "hasBookingPath": False,
                "suggestions": {},
                "modelUsed": _model_used(response),
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
            "modelUsed": _model_used(response),
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
        response = await _generate_with_fallback_async(
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[{"google_search": {}}],
                response_mime_type="application/json",
                temperature=0.0,
                top_p=0.9,
                max_output_tokens=1800,
            ),
        )
        parsed = _safe_json_parse(response.text if response else "")
        if not isinstance(parsed, dict):
            return {"emails": [], "phones": [], "addresses": [], "openingHours": [], "confidence": {}, "modelUsed": _model_used(response)}
        return {
            "emails": parsed.get("emails", []) if isinstance(parsed.get("emails"), list) else [],
            "phones": parsed.get("phones", []) if isinstance(parsed.get("phones"), list) else [],
            "addresses": parsed.get("addresses", []) if isinstance(parsed.get("addresses"), list) else [],
            "openingHours": parsed.get("openingHours", []) if isinstance(parsed.get("openingHours"), list) else [],
            "confidence": parsed.get("confidence", {}) if isinstance(parsed.get("confidence"), dict) else {},
            "modelUsed": _model_used(response),
        }
    except Exception as e:
        print(f"Error in phase1 contact fallback: {e}")
        return {"emails": [], "phones": [], "addresses": [], "openingHours": [], "confidence": {}}


