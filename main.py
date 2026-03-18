import asyncio
import sys

# Windows asyncio workaround for Playwright
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from models import ScrapeRequest, ScrapeResult, AiInsightsRequest, AiInsightsResult, WishlistRequest, TrackUrlRequest
from scraper import scrape_website
from ai_agent import get_ai_insights
from phase2_models import CompareRequest, CompareResult
from competitor_engine import run_competitor_analysis
from phase3_models import ContentAnalysisRequest, ContentAnalysisResponse
from content_agent import analyze_url_content
import traceback
import uvicorn
import os
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Wonder AI Backend")

# Setup MongoDB
MONGO_URL = os.getenv("MONGODB_URL", "mongodb+srv://jamil_db_user:qBfb3HtWmwvEEEkb@wonderai-db.qozs3tl.mongodb.net/?appName=wonderai-db")
try:
    mongo_client = AsyncIOMotorClient(MONGO_URL, serverSelectionTimeoutMS=5000)
    db = mongo_client.get_database("wonderai")
    wishlist_col = db.get_collection("wishlist")
    urls_col = db.get_collection("urls")
except Exception as e:
    print(f"[API] Error connecting to MongoDB: {e}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/scrape", response_model=ScrapeResult)
async def api_scrape(request: ScrapeRequest):
    try:
        try:
            await urls_col.insert_one({"url": request.url, "phase": "phase1"})
        except:
            pass
        result = await scrape_website(request.url)
        return result
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/scan/compare", response_model=CompareResult)
async def api_scan_compare(request: CompareRequest):
    print(f"\n[API] Received comparison request for primary URL: {request.primary_url}")
    print(f"[API] Competitors to check: {request.competitor_urls}")
    try:
        try:
            await urls_col.insert_one({"url": request.primary_url, "phase": "phase2_primary"})
            for comp in request.competitor_urls:
                await urls_col.insert_one({"url": comp, "phase": "phase2_competitor"})
        except:
            pass
        result = await run_competitor_analysis(request)
        print("[API] Comparison completed successfully.\n")
        return result
    except Exception as e:
        print(f"[API] ERROR in comparison: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ai-insights", response_model=AiInsightsResult)
async def api_ai_insights(request: AiInsightsRequest):
    try:
        insight = await get_ai_insights(request.businessName, request.url)
        return AiInsightsResult(success=True, insights=[insight])
    except Exception as e:
        traceback.print_exc()
        return AiInsightsResult(success=False, insights=[], error=str(e))

@app.post("/api/scan/content", response_model=ContentAnalysisResponse)
async def api_scan_content(request: ContentAnalysisRequest):
    print(f"\n[API] Received content analysis request for URL: {request.url}")
    try:
        try:
            await urls_col.insert_one({"url": request.url, "phase": "phase3"})
        except:
            pass
        result = await analyze_url_content(request.url)
        return result
    except Exception as e:
        print(f"[API] ERROR in content analysis: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
@app.post("/api/wishlist")
async def api_add_wishlist(request: WishlistRequest):
    try:
        await wishlist_col.insert_one({"email": request.email})
        return {"success": True}
    except Exception as e:
        print(f"[API] ERROR saving wishlist email {request.email}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/wishlist")
async def api_get_wishlist():
    try:
        emails = []
        async for doc in wishlist_col.find({}):
            emails.append(doc["email"])
        return {"success": True, "emails": emails}
    except Exception as e:
        print(f"[API] ERROR fetching wishlist emails: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/track-url")
async def api_track_url(request: TrackUrlRequest):
    try:
        await urls_col.insert_one({"url": request.url, "phase": request.phase})
        return {"success": True}
    except Exception as e:
        print(f"[API] ERROR saving tracking url {request.url}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/track-url")
async def api_get_urls():
    try:
        urls = []
        async for doc in urls_col.find({}):
            urls.append({"url": doc["url"], "phase": doc.get("phase", "unknown")})
        return {"success": True, "urls": urls}
    except Exception as e:
        print(f"[API] ERROR fetching tracking urls: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
