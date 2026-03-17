import os
import json
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

client = genai.Client()

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
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
            )
        )
        if response and response.text:
            return json.loads(response.text)
        else:
            return {
                "modelName": "Gemini", "isKnown": False, "summary": "No data returned.",
                "sentiment": "Unknown", "platforms": [], "evidence": []
            }
    except Exception as e:
        print(f"Error fetching AI insights: {{e}}")
        return {
            "modelName": "Gemini", "isKnown": False, "summary": "Failed to fetch AI insights.",
            "sentiment": "Unknown", "platforms": [], "evidence": []
        }

