import os
import json
import asyncio
from bs4 import BeautifulSoup
from google import genai
from google.genai import types
from phase3_models import ContentAnalysisResponse
from gemini_utils import generate_with_fallback
import scraper  # Reuse the fetch tools here

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

async def analyze_url_content(url: str) -> ContentAnalysisResponse:
    if not GEMINI_API_KEY:
        print("[ERROR] GEMINI_API_KEY is missing from environment!")
        return ContentAnalysisResponse(
            success=False,
            error="Gemini API Key is not configured."
        )

    try:
                # 1. Fetch the raw HTML
        html = ""
        try:
            html, final_url = await scraper.fetch_with_httpx(url)
        except Exception as e:
            print(f"HTTPX error: {e}")
            pass

        if not html:
            html = await scraper.fetch_with_httpx(url)
            
        if not html:
            return ContentAnalysisResponse(
                success=False,
                error="Could not fetch content from the provided URL."
            )
            
        # 2. Extract visible text using BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer"]):
            script.extract()
            
        text = soup.get_text(separator=' ', strip=True)
        word_count = len(text.split())
        
        # We might need to truncate if the text is huge, but Gemini Flash handles huge contexts nicely.
        # Still it's safe to cap to ~200k characters explicitly just in case.
        text = text[:200000]

        # 3. Use Gemini to analyze the text explicitly with JSON schema instruction
        client = genai.Client(api_key=GEMINI_API_KEY)
        
        prompt = f"""
        Act as an expert SEO and content strategist evaluator. Review the following text extracted from a business homepage.
        Analyze it for readability, tone/sentiment, target audience clarity, core SEO strengths/weaknesses, and score. Provide concrete, short, and highly readable feedback.
        CRITICAL: All generated text (strengths, weaknesses, advice) MUST be brief, easy to understand, and visually clean. Do not generate long, complicated essays or "rubbish". Do NOT use quotes (' or ") around your output text. Do NOT use markdown bolding (**). Focus on quick, punchy insights in plain text.
        Also provide a score breakdown out of 100 for sub-categories that make up the total seoScore (e.g. Structure, Keywords, Relevance, Readability).
        Always return your output as strictly valid JSON according to this schema:
        {{
            "seoScore": int (0-100),
            "scoreBreakdown": {{ "Structure": int, "Keywords": int, "Relevance": int, "Readability": int }},
            "readability": string (MUST be exactly ONE of these 5 options: "Very Easy", "Easy", "Moderate", "Hard", "Very Hard"),
            "sentiment": string (The Tone/Sentiment. YOU MUST INCLUDE an appropriate emoji at the beginning followed by exactly ONE descriptive word. Example: "😊 Welcoming", "👔 Professional", "🔥 Energetic"),
            "targetAudience": string (A very short, clear description of who the text targets, nicely formatted),
            "strengths": [list of short, concise plain text strings without quotes],
            "weaknesses": [list of short, concise plain text strings without quotes],
            "actionableAdvice": [list of short, concise plain text strings without quotes]
        }}
        
        Website Scraped Text:
        {text}
        """

        # Using asyncio for sync method using to_thread since `generate_content` is sync in standard usage
        response = await asyncio.to_thread(
            generate_with_fallback,
            client,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.2,
                top_p=0.95,
                max_output_tokens=4096,
            ),
        )
        
        result_text = response.text
        # Clean markdown formatting if present
        if result_text.startswith("```json"):
            result_text = result_text.replace("```json", "", 1)
            result_text = result_text.strip()
            if result_text.endswith("```"):
                result_text = result_text[:-3].strip()

        data = json.loads(result_text)
        
        return ContentAnalysisResponse(
            success=True,
            seoScore=data.get("seoScore", 50),
            scoreBreakdown=data.get("scoreBreakdown", {}),
            readability=data.get("readability", "Unknown"),
            sentiment=data.get("sentiment", "Neutral"),
            targetAudience=data.get("targetAudience", "Unknown"),
            strengths=data.get("strengths", []),
            weaknesses=data.get("weaknesses", []),
            actionableAdvice=data.get("actionableAdvice", []),
            wordCount=word_count,
            error=None
        )

    except Exception as e:
        print(f"[ERROR] Content analysis failed: {e}")
        return ContentAnalysisResponse(
            success=False,
            error=str(e),
            seoScore=0,
            scoreBreakdown={},
            readability="",
            sentiment="",
            targetAudience="",
            wordCount=0
        )
