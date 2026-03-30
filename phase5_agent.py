import os
import re
import json
import asyncio
from google import genai
from google.genai import types


def _normalize_domain(value: str) -> str:
    raw = str(value or "").strip().lower()
    if not raw:
        return ""
    raw = raw.replace("www.", "")
    raw = re.sub(r"^https?://", "", raw)
    raw = raw.split("/")[0]
    return raw

def get_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")
    return genai.Client(api_key=api_key)

async def generate_brand_questions(url: str) -> list[str]:
    """
    Sends a prompt to gemini-2.5-flash to analyze the domain and generate 20 
    highly relevant, realistic search-intent questions.
    """
    client = get_client()
    prompt = f"""
    You are an expert SEO and brand analyst. 
    Analyze the domain '{url}' (infer their business and industry).
    Generate exactly 20 highly relevant, realistic search-intent questions that potential 
    customers might type into a search engine to find businesses like this one. 
    Make sure they are natural, user-centric questions.
    Return ONLY a JSON array of strings containing the 20 questions. Do not include markdown formatting or other text.
    """
    
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
        )
    )
    
    try:
        cleaned_text = response.text.strip()
        if cleaned_text.startswith("```json"):
            cleaned_text = cleaned_text[7:]
        if cleaned_text.endswith("```"):
            cleaned_text = cleaned_text[:-3]
        
        questions = json.loads(cleaned_text.strip())
        if not isinstance(questions, list):
            return [str(q) for q in questions][:20]
        return questions[:20]
    except Exception as e:
        print(f"Error parsing Gemini response: {e}, Response: {response.text}")
        # Fallback to simple regex parsing if JSON fails
        q_list = [line.strip("- *1234567890. ") for line in response.text.split("\n") if len(line) > 10 and "?" in line]
        return q_list[:20]

async def analyze_single_question(url: str, question: dict) -> dict:
    """
    Analyzes a single question using Gemini 2.5 Flash Google Search Grounding.
    Returns status, position, sources, and competitors dynamically.
    """
    client = get_client()

    domain_match = re.search(r'(?:https?://)?(?:www\.)?([^/]+)', url)
    domain = domain_match.group(1).lower() if domain_match else url.lower()

    # Create a prompt that requests a JSON output to cleanly extract real competitor/source data
    prompt = f"""
    You are an expert brand analyst. Use Google Search to answer this user question: '{question['text']}'.
    Look at the top search results carefully.
    
    Then, provide ONLY a JSON object evaluating if the brand '{domain}' (or '{domain.split('.')[0]}') appears in the answers, and what domains are present.
    
    Strictly follow this JSON schematic:
    {{
      "was_mentioned": true or false,
      "position": <integer between 1 and 10 of where it roughly ranks, or null if not mentioned>,
      "sources": ["<domain1.com>", "<domain2.com>"],
      "competitors": ["<comp1.com>", "<comp2.com>"]
    }}
    
    Guidelines:
    - 'sources': List up to 3 domain names of the best actual sources/websites that answer this question (e.g., yelp.com, tripadvisor.com, business.com). DO NOT include "vertexaisearch.cloud.google.com". If it's the brand itself, include the brand's domain.
    - 'competitors': List up to 2 domain names of competing businesses you found in the search results that are NOT '{domain}'. Do not use placeholder names, use the actual domain names you see, like 'hilton.com' or 'local-rival.net'.
    - Do not output any markdown code blocks, just raw JSON.
    """

    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[{"google_search": {}}],
                temperature=0.1
            )
        )

        text = (response.text or "").strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.endswith("```"):
            text = text[:-3]

        # Best-effort extraction: model may return prose with an embedded JSON block.
        data = {}
        try:
            data = json.loads(text.strip())
        except Exception:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                snippet = text[start:end + 1]
                try:
                    data = json.loads(snippet)
                except Exception:
                    data = {}

        if not isinstance(data, dict):
            data = {}
        
        was_mentioned = data.get("was_mentioned", False)
        status = "Mentioned" if was_mentioned else "Not Mentioned"
        
        raw_sources = data.get("sources", [])
        if not isinstance(raw_sources, list):
            raw_sources = []
            
        clean_sources = []
        for s in raw_sources:
            s_str = _normalize_domain(s)
            if s_str and "vertexaisearch.cloud.google.com" not in s_str:
                clean_sources.append(s_str)

        # Grounding fallback: extract web domains from grounding metadata when model JSON is sparse.
        if (not clean_sources) and response.candidates:
            try:
                candidate = response.candidates[0]
                metadata = getattr(candidate, "grounding_metadata", None)
                chunks = getattr(metadata, "grounding_chunks", []) if metadata else []
                for chunk in chunks:
                    web = getattr(chunk, "web", None)
                    uri = getattr(web, "uri", "") if web else ""
                    source_domain = _normalize_domain(uri)
                    if source_domain and "vertexaisearch.cloud.google.com" not in source_domain:
                        clean_sources.append(source_domain)
            except Exception:
                pass

        clean_sources = list(dict.fromkeys(clean_sources))
                
        if not clean_sources and was_mentioned:
            clean_sources = [domain]

        raw_competitors = data.get("competitors", [])
        if not isinstance(raw_competitors, list):
            raw_competitors = []
            
        clean_competitors = []
        for c in raw_competitors:
            c_str = _normalize_domain(c)
            if c_str and "vertex" not in c_str and c_str != domain:
                clean_competitors.append(c_str)

        clean_competitors = list(dict.fromkeys(clean_competitors))

        # Fallback: infer likely competitors from sources if explicit competitors are missing.
        if not clean_competitors:
            generic_domains = {
                "google.com",
                "bing.com",
                "youtube.com",
                "facebook.com",
                "instagram.com",
                "reddit.com",
                "wikipedia.org",
            }
            for src_domain in clean_sources:
                if src_domain == domain or src_domain in generic_domains:
                    continue
                clean_competitors.append(src_domain)

        clean_competitors = list(dict.fromkeys(clean_competitors))

        raw_position = data.get("position")
        position = raw_position if isinstance(raw_position, int) and 1 <= raw_position <= 10 else None

        if not was_mentioned and any(domain == src or domain in src for src in clean_sources):
            was_mentioned = True

        status = "Mentioned" if was_mentioned else "Not Mentioned"

        return {
            "id": question["id"],
            "status": status,
            "position": position,
            "sources": clean_sources[:3],
            "competitors": clean_competitors[:2]
        }

    except Exception as e:
        print(f"Error parsing single question {question['id']}: {e}")
        return {
            "id": question["id"],
            "status": "Not Mentioned",
            "position": None,
            "sources": [],
            "competitors": []
        }

async def rank_brand_in_ai(url: str, questions: list) -> dict:
    """
    Uses Gemini 2.5 Flash with Google Search Grounding enabled.
    Loops through questions, asks Gemini to answer them using live search data,
    and explicitly grades whether the target url or brand name appeared in the 
    top 10 sources provided in the grounded response.
    Input: questions - list of dicts: [{'id': '...', 'text': '...'}]
    Returns: dict mapping question id to dict with status, position, sources.
    """
    client = get_client()
    results = {}
    
    # Simple extraction of domain to use as target
    domain_match = re.search(r'(?:https?://)?(?:www\.)?([^/]+)', url)
    domain = domain_match.group(1).lower() if domain_match else url.lower()
    
    for i, q in enumerate(questions):
        try:
            prompt = f"Using Google Search, answer this question: '{q['text']}'. Provide a comprehensive answer with sources."
            
            # Grounding configuration
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
                config=types.GenerateContentConfig(
                    tools=[{"google_search": {}}],
                    temperature=0.0
                )
            )
            
            # Extract sources if search grounding was used
            sources = []
            mentioned = False
            position = None
            
            # genai library structure for grounding:
            # response.candidates[0].grounding_metadata.grounding_chunks[i].web.uri / title
            if response.candidates and response.candidates[0].grounding_metadata:
                metadata = response.candidates[0].grounding_metadata
                if hasattr(metadata, "grounding_chunks") and metadata.grounding_chunks:
                    for idx, chunk in enumerate(metadata.grounding_chunks):
                        if hasattr(chunk, "web") and chunk.web and hasattr(chunk.web, "uri"):
                            uri = chunk.web.uri
                            title = chunk.web.title if hasattr(chunk.web, "title") else uri
                            domain_only = re.search(r'(?:https?://)?(?:www\.)?([^/]+)', uri)
                            source_name = domain_only.group(1).capitalize() if domain_only else "Web"
                            if source_name not in sources:
                                sources.append(source_name)
                            
                            # Check if the brand's domain is in the URI
                            if domain in uri.lower() or domain.split('.')[0] in title.lower():
                                mentioned = True
                                if position is None:
                                    position = idx + 1
            
            # Fallback text search if metadata chunks didn't yield sources but we still want to check response text
            if not mentioned and domain.split('.')[0] in response.text.lower():
                mentioned = True
                if position is None:
                    position = 5  # Estimated
            
            status = "Mentioned" if mentioned else "Not Mentioned"
            
            results[q['id']] = {
                "status": status,
                "position": position,
                "sources": sources[:3] if sources else ["Google"] if mentioned else []
            }
            
        except Exception as e:
            print(f"Error processing question {q['id']}: {str(e)}")
            results[q['id']] = {
                "status": "Not Mentioned",
                "position": None,
                "sources": []
            }
            
        # Small delay to avoid aggressive rate limiting
        if i < len(questions) - 1:
            await asyncio.sleep(1.0)
            
    return results
