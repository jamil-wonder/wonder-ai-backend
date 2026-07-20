# Generated from the former backend/main.py lines 1-436.
import asyncio
import sys

# Windows asyncio workaround for Playwright
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import OAuth2PasswordBearer
from bson import ObjectId
from models import (
    ScrapeRequest,
    ScrapeResult,
    AiInsightsRequest,
    AiInsightsResult,
    WishlistRequest,
    TrackUrlRequest,
    BlogAnalyzeRequest,
    BlogAnalysisResponse,
    BlogAnalysis,
    BlogGenerateRequest,
    BlogGenerateResponse,
    BlogGenerateResult,
    BlogRewriteSectionRequest,
    BlogRewriteSectionResponse,
    BlogSection,
    BlogUsageResponse,
    BlogWeeklySetupRequest,
    BlogWeeklyEnsureRequest,
    ContentPageGeneratorRequest,
    ContentPageGeneratorResponse,
    CompetitorTrackingRunRequest,
    CompetitorTrackingRunResponse,
    CompetitorTrackingStatusResponse,
)
from scraping.scraper import scrape_website
from agents.ai_agent import (
    get_ai_insights_multi,
    get_blog_analysis_perplexity,
    generate_seo_blog,
    generate_weekly_blog_ideas,
    generate_content_page,
    rewrite_blog_section,
)
from models.phase2_models import CompareRequest, CompareResult
from engines.competitor_engine import run_competitor_analysis
from models.phase3_models import ContentAnalysisRequest, ContentAnalysisResponse
from agents.content_agent import analyze_url_content
from models.phase5_models import (
    Phase5QuestionsRequest,
    Phase5QuestionsResponse,
    Phase5AnalyzeRequest,

    Phase5AnalyzeResponse,
    Phase5AnalyzeSingleRequest,
    Phase5AnalyzeSingleResponse,
    Phase5StartJobRequest,
    Phase5StartJobResponse,
    Phase5JobStatusResponse,
)
from agents.phase5.config import PHASE5_ENABLE_GEMINI
from agents.phase5_agent import (
    generate_brand_questions,
    rank_brand_in_ai,
    analyze_single_question,
    analyze_single_question_multi,
    compute_provider_score,
    _run_with_backoff,
    generate_brand_perception_summary,
    generate_deep_competitor_scores,
    generate_public_competitor_suggestions,
    Phase5RateLimitError,
    _estimate_target_visibility_score,
    _normalize_domain,
)
from models import UserCreate, UserResponse, Token, LoginRequest
from models import BusinessResponse, BusinessUpsertRequest
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests
from pydantic import BaseModel
import secrets
import bcrypt
from jose import JWTError, jwt
from datetime import datetime, timedelta
from dotenv import load_dotenv

import traceback
import uvicorn
import os
import uuid
import json
import re
import hashlib
from concurrent.futures import ThreadPoolExecutor
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import ReturnDocument
from urllib.parse import urlparse

load_dotenv()


def _run_scrape_worker(url: str):
    # Run scrape in a dedicated worker thread so heavy parsing cannot block API responsiveness.
    if sys.platform == "win32":
        try:
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        except Exception:
            pass
    return asyncio.run(scrape_website(url))


def _run_scrape_worker_core(url: str):
    # Reduced fallback: no AI enrichment, no deep crawl, faster response when full scrape times out.
    if sys.platform == "win32":
        try:
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        except Exception:
            pass
    return asyncio.run(scrape_website(url, enable_ai=False, enable_deep_crawl=False))


app = FastAPI(title="Wonder AI Backend")

# Phase 5 worker settings
PHASE5_WORKER_CONCURRENCY = max(1, min(2, int(os.getenv("PHASE5_WORKER_CONCURRENCY", "2"))))
PHASE5_WORKER_POLL_INTERVAL = float(os.getenv("PHASE5_WORKER_POLL_INTERVAL", "0.5"))
PHASE5_JOB_PARALLELISM = max(1, min(8, int(os.getenv("PHASE5_JOB_PARALLELISM", "4"))))
PHASE5_MODEL_MAX_THREADS = max(4, min(16, int(os.getenv("PHASE5_MODEL_MAX_THREADS", "8"))))
PHASE5_QUESTION_TIMEOUT_GEMINI_SEC = int(os.getenv("PHASE5_QUESTION_TIMEOUT_GEMINI_SEC", "140"))
PHASE5_QUESTION_TIMEOUT_OPENAI_SEC = int(os.getenv("PHASE5_QUESTION_TIMEOUT_OPENAI_SEC", "40"))
PHASE5_QUESTION_TIMEOUT_PERPLEXITY_SEC = int(os.getenv("PHASE5_QUESTION_TIMEOUT_PERPLEXITY_SEC", "45"))
PHASE5_QUESTION_TIMEOUT_ANTHROPIC_SEC = int(os.getenv("PHASE5_QUESTION_TIMEOUT_ANTHROPIC_SEC", "45"))
PHASE5_STALE_RUNNING_SECONDS = int(os.getenv("PHASE5_STALE_RUNNING_SECONDS", "120"))
PHASE5_RECOVER_STALE_RUNNING = str(os.getenv("PHASE5_RECOVER_STALE_RUNNING", "false")).strip().lower() == "true"
PHASE5_STALE_QUEUED_SECONDS = int(os.getenv("PHASE5_STALE_QUEUED_SECONDS", "1800"))
PHASE5_RESUME_QUEUED_ON_STARTUP = str(os.getenv("PHASE5_RESUME_QUEUED_ON_STARTUP", "false")).strip().lower() == "true"
PHASE5_WORKER_ID = f"{os.getenv('HOSTNAME', 'local')}-{uuid.uuid4().hex[:8]}"
PHASE5_TERMINAL_STATUSES = {"completed", "failed", "cancelled"}

allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "")
if allowed_origins_env.strip():
    allowed_origins = [origin.strip() for origin in allowed_origins_env.split(",") if origin.strip()]
else:
    allowed_origins = [
        "https://wonderscore.ai",
        "https://www.wonderscore.ai",
        "https://app.wonderscore.ai",
        "https://api.wonderscore.ai",
        "https://wonder-landing-mu.vercel.app",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]

# Setup MongoDB
MONGO_URL = os.getenv("MONGODB_URL")
if not MONGO_URL:
    raise RuntimeError("CRITICAL ERROR: MONGODB_URL is missing from environment variables.")
phase5_jobs_col = None
ai_usage_col = None
user_history_meta_col = None
businesses_col = None
public_rate_limits_col = None
generated_content_pages_col = None
competitor_tracking_runs_col = None
weekly_blog_suggestions_col = None
auth_handoffs_col = None
try:
    mongo_client = AsyncIOMotorClient(MONGO_URL, serverSelectionTimeoutMS=5000)
    db = mongo_client.get_database("wonderai")
    wishlist_col = db.get_collection("wishlist")
    urls_col = db.get_collection("urls")
    users_col = db.get_collection("users")
    businesses_col = db.get_collection("businesses")
    generated_content_pages_col = db.get_collection("generated_content_pages")
    competitor_tracking_runs_col = db.get_collection("competitor_tracking_runs")
    weekly_blog_suggestions_col = db.get_collection("weekly_blog_suggestions")
    auth_handoffs_col = db.get_collection("auth_handoffs")
    phase5_jobs_col = db.get_collection("phase5_jobs")
    ai_usage_col = db.get_collection("ai_usage_events")
    user_history_meta_col = db.get_collection("user_history_meta")
    public_rate_limits_col = db.get_collection("public_rate_limits")
except Exception as e:
    print(f"[API] Error connecting to MongoDB: {type(e).__name__}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/healthz")
async def healthz():
    return {
        "status": "ok",
        "service": "wonder-ai-backend",
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/readyz")
async def readyz():
    checks = {
        "mongodb": "unknown",
    }
    ready = True

    try:
        await mongo_client.admin.command("ping")
        checks["mongodb"] = "ok"
    except Exception as e:
        ready = False
        checks["mongodb"] = f"error:{type(e).__name__}"

    return {
        "status": "ok" if ready else "degraded",
        "service": "wonder-ai-backend",
        "checks": checks,
        "timestamp": datetime.utcnow().isoformat(),
    }

# Authentication Config
SECRET_KEY = os.getenv("JWT_SECRET_KEY")
if not SECRET_KEY:
    raise RuntimeError("CRITICAL SECURITY ERROR: JWT_SECRET_KEY is missing from environment variables.")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7 # 7 days

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))

def get_password_hash(password: str) -> str:
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')

def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# --- Dependencies ---
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login", auto_error=False)

async def get_current_user_optional(token: str = Depends(oauth2_scheme)):
    if not token:
        return None
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("id")
        if user_id is None:
            return None
        user = await users_col.find_one({"_id": ObjectId(user_id)})
        if user:
            return {
                "id": str(user["_id"]), 
                "email": user["email"], 
                "role": user.get("role", "user"), 
                "status": user.get("status", "active")
            }
    except Exception:
        pass
    return None

async def get_current_user(user: dict = Depends(get_current_user_optional)):
    if not user:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    if user.get("status") == "banned":
        raise HTTPException(status_code=403, detail="Your account has been restricted.")
    return user

async def get_current_admin_user(current_user: dict = Depends(get_current_user)):
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Insufficient privileges. Admin access required.")
    return current_user


async def _log_ai_usage_event(event: dict):
    """Best-effort logger for AI feature usage analytics."""
    if ai_usage_col is None:
        return
    try:
        now = datetime.utcnow()
        payload = {
            "timestamp": now,
            "timestamp_iso": now.isoformat(),
            **(event or {}),
        }

        model_provider = str(payload.get("model_provider") or "").strip().lower()
        if model_provider == "chatgpt":
            model_provider = "openai"
        elif model_provider == "google":
            model_provider = "gemini"
        model_name = payload.get("model_name")
        if not model_name and model_provider == "gemini":
            model_name = (os.getenv("GEMINI_MODEL_PRIMARY") or os.getenv("GEMINI_MODEL") or "").strip() or None
        if not model_name and model_provider == "openai":
            model_name = (os.getenv("OPENAI_MODEL_PHASE5") or "").strip() or None
        if not model_name and model_provider == "perplexity":
            model_name = (os.getenv("PERPLEXITY_MODEL_PHASE5") or "sonar-pro").strip()
        if not model_name and model_provider in {"anthropic", "claude"}:
            model_name = (os.getenv("ANTHROPIC_MODEL_PHASE5") or "claude-sonnet-4-5").strip()

        model_family = payload.get("model_family")
        lowered_model = str(model_name or "").lower()
        if not model_family:
            if model_provider == "gemini":
                model_family = "gemini"
            elif model_provider == "openai":
                model_family = "gpt"
            elif model_provider == "perplexity":
                model_family = "perplexity"
            elif model_provider in {"anthropic", "claude"}:
                model_family = "claude"
            elif "gemini" in lowered_model:
                model_family = "gemini"
            elif any(tag in lowered_model for tag in ["gpt", "o1", "o3", "o4"]):
                model_family = "gpt"
            elif "claude" in lowered_model:
                model_family = "claude"
            elif "perplexity" in lowered_model:
                model_family = "perplexity"
            else:
                model_family = "unknown"

        provider = payload.get("provider")
        if not provider:
            if model_family == "gemini":
                provider = "google"
            elif model_family == "gpt" or model_provider == "openai":
                provider = "openai"
            elif model_family == "claude":
                provider = "anthropic"
            elif model_family == "perplexity":
                provider = "perplexity"
            else:
                provider = "unknown"

        payload["model_name"] = model_name
        payload["model_family"] = model_family
        payload["provider"] = provider

        payload = {
            **payload,
        }
        await ai_usage_col.insert_one(payload)
    except Exception as e:
        print(f"[AI Usage] log failed: {e}")

# --- Auth Routes ---

class GoogleAuthRequest(BaseModel):
    credential: str


class UserProfileUpdateRequest(BaseModel):
    name: str
    email: str


class UserPasswordChangeRequest(BaseModel):
    current_password: str
    new_password: str


class AuthHandoffExchangeRequest(BaseModel):
    code: str


class PublicCompetitorsRequest(BaseModel):
    url: str
    businessName: str | None = None
    category: str | None = None
    location: str | None = None
    description: str | None = None


class PublicCompetitorsResponse(BaseModel):
    success: bool
    competitors: list[dict] = []
    error: str | None = None


PUBLIC_PREVIEW_ATTEMPT_LIMIT = int(os.getenv("PUBLIC_PREVIEW_ATTEMPT_LIMIT", "3"))
PUBLIC_PREVIEW_SUCCESS_LIMIT = int(os.getenv("PUBLIC_PREVIEW_SUCCESS_LIMIT", "1"))
PUBLIC_PREVIEW_WINDOW_HOURS = int(os.getenv("PUBLIC_PREVIEW_WINDOW_HOURS", "24"))


def _normalize_site(value: str) -> str:
    raw = (value or "").strip().lower()
    if not raw:
        return ""
    parsed = urlparse(raw if "://" in raw else f"https://{raw}")
    host = (parsed.netloc or parsed.path or "").strip().lower()
    if host.startswith("www."):
        host = host[4:]
    return host


def _get_public_client_ip(request: Request) -> str:
    forwarded = (request.headers.get("x-forwarded-for") or "").split(",")[0].strip()
    return (
        forwarded
        or request.headers.get("cf-connecting-ip")
        or request.headers.get("x-real-ip")
        or (request.client.host if request.client else "")
        or "unknown"
    )


def _get_public_device_id(request: Request) -> str:
    raw = (request.headers.get("x-wonder-device-id") or "").strip()
    safe = re.sub(r"[^a-zA-Z0-9_-]", "", raw)[:96]
    return safe or "unknown-device"


def _get_public_scan_id(request: Request) -> str:
    raw = (request.headers.get("x-wonder-scan-id") or "").strip()
    safe = re.sub(r"[^a-zA-Z0-9_-]", "", raw)[:96]
    return safe or uuid.uuid4().hex


def _public_limit_key(request: Request) -> str:
    fingerprint = f"{_get_public_client_ip(request)}|{_get_public_device_id(request)}"
    return hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()
