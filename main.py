from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from models import ScrapeRequest, ScrapeResult
from scraper import scrape_website
from phase2_models import MultiScanRequest, MultiScanResult
from conflict_engine import run_multi_scan
import traceback
import uvicorn

app = FastAPI(title="Wonder AI Backend")

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
        result = await scrape_website(request.url)
        return result
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/scan/multi", response_model=MultiScanResult)
async def api_scan_multi(request: MultiScanRequest):
    try:
        result = await run_multi_scan(request)
        return result
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
