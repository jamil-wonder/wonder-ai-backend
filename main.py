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
)
from scraping.scraper import scrape_website
from agents.ai_agent import get_ai_insights_multi, get_blog_analysis_perplexity
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
from agents.phase5_agent import (
    generate_brand_questions,
    rank_brand_in_ai,
    analyze_single_question,
    analyze_single_question_multi,
    compute_provider_score,
    _run_with_backoff,
    generate_brand_perception_summary,
    generate_deep_competitor_scores,
    Phase5RateLimitError,
    _estimate_target_visibility_score,
    _normalize_domain,
)
from models import UserCreate, UserResponse, Token, LoginRequest
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
import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import ReturnDocument
from urllib.parse import urlparse

load_dotenv()
from redis import asyncio as aioredis
from dotenv import load_dotenv

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
PHASE5_STALE_RUNNING_SECONDS = int(os.getenv("PHASE5_STALE_RUNNING_SECONDS", "120"))
PHASE5_RECOVER_STALE_RUNNING = str(os.getenv("PHASE5_RECOVER_STALE_RUNNING", "false")).strip().lower() == "true"
PHASE5_STALE_QUEUED_SECONDS = int(os.getenv("PHASE5_STALE_QUEUED_SECONDS", "1800"))
PHASE5_RESUME_QUEUED_ON_STARTUP = str(os.getenv("PHASE5_RESUME_QUEUED_ON_STARTUP", "false")).strip().lower() == "true"
PHASE5_WORKER_ID = f"{os.getenv('HOSTNAME', 'local')}-{uuid.uuid4().hex[:8]}"
PHASE5_REDIS_URL = os.getenv("REDIS_URL", "").strip()
PHASE5_REDIS_QUEUE_KEY = os.getenv("PHASE5_REDIS_QUEUE_KEY", "phase5:jobs:queue")
PHASE5_CACHE_REUSE_SECONDS = int(os.getenv("PHASE5_CACHE_REUSE_SECONDS", "21600"))
PHASE5_TERMINAL_STATUSES = {"completed", "failed", "cancelled"}

allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "")
if allowed_origins_env.strip():
    allowed_origins = [origin.strip() for origin in allowed_origins_env.split(",") if origin.strip()]
else:
    allowed_origins = [
        "https://wonderscore.ai",
        "https://www.wonderscore.ai",
        "https://api.wonderscore.ai",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]

# Setup MongoDB
MONGO_URL = os.getenv("MONGODB_URL")
if not MONGO_URL:
    raise RuntimeError("CRITICAL ERROR: MONGODB_URL is missing from environment variables.")
phase5_jobs_col = None
ai_usage_col = None
user_history_meta_col = None
try:
    mongo_client = AsyncIOMotorClient(MONGO_URL, serverSelectionTimeoutMS=5000)
    db = mongo_client.get_database("wonderai")
    wishlist_col = db.get_collection("wishlist")
    urls_col = db.get_collection("urls")
    users_col = db.get_collection("users")
    phase5_jobs_col = db.get_collection("phase5_jobs")
    ai_usage_col = db.get_collection("ai_usage_events")
    user_history_meta_col = db.get_collection("user_history_meta")
except Exception as e:
    print(f"[API] Error connecting to MongoDB: {type(e).__name__}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

        model_family = payload.get("model_family")
        lowered_model = str(model_name or "").lower()
        if not model_family:
            if model_provider == "gemini":
                model_family = "gemini"
            elif model_provider == "openai":
                model_family = "gpt"
            elif model_provider == "perplexity":
                model_family = "perplexity"
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


def _normalize_site(value: str) -> str:
    raw = (value or "").strip().lower()
    if not raw:
        return ""
    parsed = urlparse(raw if "://" in raw else f"https://{raw}")
    host = (parsed.netloc or parsed.path or "").strip().lower()
    if host.startswith("www."):
        host = host[4:]
    return host


def _iso_from_range(range_value: str) -> str | None:
    rv = (range_value or "").strip().lower()
    if rv in {"", "all"}:
        return None
    now = datetime.utcnow()
    if rv == "week":
        return (now - timedelta(days=7)).isoformat()
    if rv == "month":
        return (now - timedelta(days=30)).isoformat()
    if rv == "year":
        return (now - timedelta(days=365)).isoformat()
    return None


def _to_datetime(iso_value: str | None) -> datetime | None:
    if not iso_value:
        return None
    try:
        return datetime.fromisoformat(str(iso_value).replace("Z", "+00:00"))
    except Exception:
        return None


def _blog_clamp(value: int | float, min_value: int = 0, max_value: int = 100) -> int:
    return int(max(min_value, min(max_value, value)))


def _blog_split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"[.!?]+", re.sub(r"\s+", " ", text or "")) if s.strip()]


def _blog_count_matches(text: str, pattern: str) -> int:
    return len(re.findall(pattern, text or "", flags=re.I))


def _build_blog_base_analysis(text: str, attachment_count: int) -> dict:
    cleaned = (text or "").strip()
    words = [w for w in re.split(r"\s+", cleaned) if w]
    word_count = len(words)
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", cleaned) if p.strip()]
    paragraph_count = len(paragraphs)
    lines = [line.strip() for line in cleaned.split("\n") if line.strip()]
    heading_count = len([
        line for line in lines
        if re.match(r"^#{1,3}\s+", line) or re.match(r"^(?:[A-Z][A-Za-z0-9 ,:&-]{5,80})\??$", line)
    ])
    list_count = len([line for line in lines if re.match(r"^(-|\*|\d+\.)\s+", line)])
    link_count = _blog_count_matches(cleaned, r"https?://") + _blog_count_matches(cleaned, r"\[[^\]]+\]\([^\)]+\)")
    cta_count = _blog_count_matches(cleaned.lower(), r"\b(learn more|read more|sign up|subscribe|download|book a demo|get started|contact us|buy now|try it now)\b")

    sentence_lengths = [len([w for w in s.split() if w]) for s in _blog_split_sentences(cleaned)]
    avg_sentence_length = (sum(sentence_lengths) / len(sentence_lengths)) if sentence_lengths else 0

    structure = 10
    structure += _blog_clamp(heading_count * 4, 0, 16)
    structure += _blog_clamp(list_count * 2, 0, 8)
    structure += _blog_clamp(8 if paragraph_count >= 4 else paragraph_count * 2, 0, 8)
    structure += 8 if word_count >= 700 else 4 if word_count >= 400 else 0
    structure += 4 if attachment_count > 0 else 0
    structure = _blog_clamp(structure, 0, 35)

    readability = 10
    if avg_sentence_length > 0:
        if 12 <= avg_sentence_length <= 22:
            readability += 18
        elif 8 <= avg_sentence_length <= 28:
            readability += 12
        else:
            readability += 4
    readability += 10 if paragraph_count >= 4 else 6 if paragraph_count >= 2 else 2
    readability += 8 if word_count >= 500 else 5 if word_count >= 250 else 1
    readability = _blog_clamp(readability, 0, 30)

    seo = 10
    seo += 10 if heading_count > 0 else 0
    seo += 6 if link_count > 0 else 0
    seo += 8 if word_count >= 800 else 5 if word_count >= 500 else 1
    seo += 4 if attachment_count > 0 else 0
    seo = _blog_clamp(seo, 0, 30)

    engagement = 8
    engagement += 10 if cta_count > 0 else 0
    engagement += 8 if list_count > 0 else 0
    engagement += 4 if _blog_count_matches(cleaned, r"\?") > 2 else 0
    engagement += 4 if _blog_count_matches(cleaned, r"\b(example|tip|guide|step|how to|why|benefit)\b") > 2 else 0
    engagement = _blog_clamp(engagement, 0, 20)

    score = round(structure * 0.3 + readability * 0.24 + seo * 0.28 + engagement * 0.18)
    score = _blog_clamp(score, 0, 100)
    if word_count < 150:
        score = max(18, score - 12)
    elif word_count < 350:
        score = max(28, score - 6)
    if attachment_count > 0:
        score = _blog_clamp(score + 3, 0, 100)

    strengths = []
    if heading_count > 0:
        strengths.append("Clear sectioning with headings")
    if list_count > 0:
        strengths.append("Easy-to-scan list formatting")
    if link_count > 0:
        strengths.append("Includes useful references or links")
    if cta_count > 0:
        strengths.append("Has a conversion-focused CTA")
    if attachment_count > 0:
        strengths.append("Supported by an attached PDF reference")

    suggestions = []
    if word_count < 500:
        suggestions.append(
            "Expand the article to at least 700 to 1,000 words if this is meant to rank competitively."
        )
    if heading_count < 2:
        suggestions.append(
            "Add more H2 and H3 sections so search engines and readers can scan the content faster."
        )
    if link_count == 0:
        suggestions.append(
            "Add internal or external links to strengthen topical authority and help users explore more."
        )
    if list_count == 0:
        suggestions.append(
            "Break dense sections into bullets or numbered steps to improve readability."
        )
    if cta_count == 0:
        suggestions.append(
            "Add a stronger CTA at the end to guide the reader toward the next action."
        )
    if paragraph_count < 4:
        suggestions.append(
            "Split long walls of text into shorter paragraphs for better mobile readability."
        )
    if attachment_count > 0:
        suggestions.append(
            "Use the attached PDF to reinforce claims, cite statistics, or support the article with original evidence."
        )
    if not suggestions:
        suggestions.append(
            "The blog is already in good shape. Consider a light title optimization and one extra supporting link for polish."
        )

    grade = "Needs Work"
    if score >= 90:
        grade = "Exceptional"
    elif score >= 80:
        grade = "Strong"
    elif score >= 70:
        grade = "Good"
    elif score >= 55:
        grade = "Fair"

    overview = (
        "This article is structured well enough to compete, with only a few optimization gaps left to close."
        if score >= 80
        else "This draft has a workable base, but it needs clearer structure and stronger SEO signals before it can compete well."
        if score >= 55
        else "This draft needs more structure, clearer search intent coverage, and stronger on-page SEO signals before it is competitive."
    )

    engagement_label = "high" if engagement > 70 else "moderate"
    summary = (
        f"A {word_count}-word {grade.lower()} draft. It features {heading_count} headings and {link_count} links, aiming for a {engagement_label} level of reader engagement."
        if word_count > 0
        else "An empty or very short draft that requires more content to analyze effectively."
    )

    weak_spots = [
        s for s in suggestions
        if "Expand" in s or "H2 and H3" in s or "Split long walls" in s
    ]
    improvements = [s for s in suggestions if s not in weak_spots]

    return {
        "metrics": {
            "wordCount": word_count,
            "paragraphCount": paragraph_count,
            "headingCount": heading_count,
            "listCount": list_count,
            "linkCount": link_count,
            "ctaCount": cta_count,
            "readability": readability,
            "structure": structure,
            "seo": seo,
            "engagement": engagement,
            "score": score,
            "grade": grade,
            "attachmentCount": attachment_count,
        },
        "strengths": strengths,
        "suggestions": suggestions,
        "overview": overview,
        "summary": summary,
        "weakSpots": weak_spots,
        "improvements": improvements,
    }

@app.post("/api/auth/google", response_model=Token)
async def api_google_login(request: GoogleAuthRequest):
    try:
        client_id = os.getenv("GOOGLE_CLIENT_ID")
        if not client_id:
            raise HTTPException(status_code=500, detail="Google Client ID not configured")
            
        idinfo = id_token.verify_oauth2_token(
            request.credential, google_requests.Request(), client_id
        )

        email = idinfo.get("email")
        name = idinfo.get("name")
        
        if not email:
            raise HTTPException(status_code=400, detail="Invalid Google token payload")

        user = await users_col.find_one({"email": email})
        
        if not user:
            total_users = await users_col.count_documents({})
            assigned_role = "admin" if total_users == 0 else "user"
            
            hashed_password = get_password_hash(secrets.token_urlsafe(32))
            
            new_user = {
                "name": name,
                "email": email,
                "hashed_password": hashed_password,
                "role": assigned_role,
                "status": "active",
                "created_at": datetime.utcnow().isoformat()
            }
            result = await users_col.insert_one(new_user)
            user_id = str(result.inserted_id)
            user_role = assigned_role
            user_status = "active"
            created_at = new_user["created_at"]
        else:
            user_id = str(user["_id"])
            user_role = user.get("role", "user")
            user_status = user.get("status", "active")
            created_at = user.get("created_at", datetime.utcnow().isoformat())
            name = user.get("name", name)

        if user_status == "banned":
            raise HTTPException(status_code=403, detail="Your account has been restricted.")

        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": email, "id": user_id}, expires_delta=access_token_expires
        )

        user_response = UserResponse(
            id=user_id,
            name=name,
            email=email,
            created_at=created_at,
            role=user_role,
            status=user_status
        )

        return Token(access_token=access_token, token_type="bearer", user=user_response)

    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid Google token")
    except HTTPException:
        raise
    except Exception as e:
        print(f"[API] ERROR during google auth: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/api/auth/signup", response_model=Token)
async def api_signup(user: UserCreate):
    try:
        # Check if user already exists
        existing_user = await users_col.find_one({"email": user.email})
        if existing_user:
            raise HTTPException(status_code=400, detail="Email already registered")

        # Create new user (automatically assign first user as admin conditionally if no users exist)
        total_users = await users_col.count_documents({})
        assigned_role = "admin" if total_users == 0 else "user"

        hashed_password = get_password_hash(user.password)
        new_user = {
            "name": user.name,
            "email": user.email,
            "hashed_password": hashed_password,
            "role": assigned_role,
            "status": "active",
            "created_at": datetime.utcnow().isoformat()
        }
        
        result = await users_col.insert_one(new_user)
        user_id = str(result.inserted_id)

        # Generate JWT token
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user.email, "id": user_id}, expires_delta=access_token_expires
        )

        user_response = UserResponse(
            id=user_id,
            name=user.name,
            email=user.email,
            created_at=new_user["created_at"],
            role=new_user["role"],
            status=new_user["status"]
        )

        return Token(access_token=access_token, token_type="bearer", user=user_response)
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"[API] ERROR during signup: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/api/auth/login", response_model=Token)
async def api_login(request: LoginRequest):
    try:
        # Find user
        user = await users_col.find_one({"email": request.email})
        if not user:
            raise HTTPException(status_code=401, detail="Invalid email or password")

        # Verify password
        if not verify_password(request.password, user["hashed_password"]):
            raise HTTPException(status_code=401, detail="Invalid email or password")

        user_id = str(user["_id"])

        # Generate JWT token
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user["email"], "id": user_id}, expires_delta=access_token_expires
        )

        user_response = UserResponse(
            id=user_id,
            name=user["name"],
            email=user["email"],
            created_at=user.get("created_at", datetime.utcnow().isoformat()),
            role=user.get("role", "user"),
            status=user.get("status", "active")
        )

        return Token(access_token=access_token, token_type="bearer", user=user_response)

    except HTTPException:
        raise
    except Exception as e:
        print(f"[API] ERROR during login: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/user/profile", response_model=UserResponse)
async def api_user_profile(current_user: dict = Depends(get_current_user)):
    user = await users_col.find_one({"_id": ObjectId(current_user["id"])})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    return UserResponse(
        id=str(user["_id"]),
        name=user.get("name", ""),
        email=user.get("email", ""),
        created_at=user.get("created_at", datetime.utcnow().isoformat()),
        role=user.get("role", "user"),
        status=user.get("status", "active"),
    )


@app.put("/api/user/profile", response_model=UserResponse)
async def api_user_profile_update(request: UserProfileUpdateRequest, current_user: dict = Depends(get_current_user)):
    name = (request.name or "").strip()
    email = (request.email or "").strip().lower()

    if not name:
        raise HTTPException(status_code=400, detail="Name is required")
    if not email:
        raise HTTPException(status_code=400, detail="Email is required")

    existing = await users_col.find_one({"email": email})
    if existing and str(existing.get("_id")) != current_user["id"]:
        raise HTTPException(status_code=400, detail="Email already in use")

    updated = await users_col.find_one_and_update(
        {"_id": ObjectId(current_user["id"])},
        {
            "$set": {
                "name": name,
                "email": email,
                "updated_at": datetime.utcnow().isoformat(),
            }
        },
        return_document=ReturnDocument.AFTER,
    )

    if not updated:
        raise HTTPException(status_code=404, detail="User not found")

    return UserResponse(
        id=str(updated["_id"]),
        name=updated.get("name", ""),
        email=updated.get("email", ""),
        created_at=updated.get("created_at", datetime.utcnow().isoformat()),
        role=updated.get("role", "user"),
        status=updated.get("status", "active"),
    )


@app.put("/api/user/password")
async def api_user_change_password(request: UserPasswordChangeRequest, current_user: dict = Depends(get_current_user)):
    if not request.current_password or not request.new_password:
        raise HTTPException(status_code=400, detail="Both current and new passwords are required")
    if len(request.new_password) < 8:
        raise HTTPException(status_code=400, detail="New password must be at least 8 characters")

    user = await users_col.find_one({"_id": ObjectId(current_user["id"])})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if not verify_password(request.current_password, user.get("hashed_password", "")):
        raise HTTPException(status_code=401, detail="Current password is incorrect")

    await users_col.update_one(
        {"_id": ObjectId(current_user["id"])},
        {
            "$set": {
                "hashed_password": get_password_hash(request.new_password),
                "updated_at": datetime.utcnow().isoformat(),
            }
        },
    )
    return {"message": "Password updated successfully"}


@app.get("/api/user/history")
async def api_user_history(
    page: int = 1,
    limit: int = 10,
    status: str = "all",
    model: str = "all",
    site: str = "",
    range: str = "all",
    current_user: dict = Depends(get_current_user),
):
    safe_page = max(1, page)
    safe_limit = max(1, min(limit, 50))
    skip = (safe_page - 1) * safe_limit
    user_id = current_user["id"]

    clear_cutoff = None
    if user_history_meta_col is not None:
        meta = await user_history_meta_col.find_one({"user_id": user_id}, {"_id": 0, "cleared_at": 1})
        clear_cutoff = meta.get("cleared_at") if meta else None

    history_query = {"user_id": user_id}
    status_filter = (status or "all").strip().lower()
    if status_filter and status_filter != "all":
        history_query["status"] = status_filter

    model_filter = (model or "all").strip().lower()
    if model_filter and model_filter != "all":
        if model_filter in {"chatgpt", "openai"}:
            history_query["$or"] = [
                {"model": "openai"},
                {"provider_scores.chatgpt": {"$exists": True}},
            ]
        elif model_filter == "perplexity":
            history_query["$or"] = [
                {"model": "perplexity"},
                {"provider_scores.perplexity": {"$exists": True}},
            ]
        elif model_filter == "gemini":
            history_query["$or"] = [
                {"model": "gemini"},
                {"provider_scores.gemini": {"$exists": True}},
            ]
        else:
            history_query["model"] = model_filter

    site_host = _normalize_site(site)
    if site_host:
        history_query["url"] = {"$regex": site_host, "$options": "i"}

    cutoff_iso = _iso_from_range(range)
    created_at_bounds = {}
    if cutoff_iso:
        created_at_bounds["$gte"] = cutoff_iso
    if clear_cutoff:
        created_at_bounds["$gt"] = max(clear_cutoff, created_at_bounds.get("$gte", clear_cutoff))
    if created_at_bounds:
        history_query["created_at"] = created_at_bounds

    total_runs = 0
    if phase5_jobs_col is not None:
        total_runs = await phase5_jobs_col.count_documents(history_query)

    phase5_runs = []
    if phase5_jobs_col is not None:
        cursor = phase5_jobs_col.find(
            history_query,
            {
                "_id": 0,
                "job_id": 1,
                "job_type": 1,
                "model": 1,
                "url": 1,
                "status": 1,
                "total": 1,
                "processed": 1,
                "overall_score": 1,
                "provider_scores": 1,
                "created_at": 1,
                "updated_at": 1,
                "error": 1,
            },
        ).sort("created_at", -1).skip(skip).limit(safe_limit)
        raw_runs = await cursor.to_list(length=safe_limit)

        for run in raw_runs:
            providers = run.get("provider_scores") or {}
            models_ran = []
            if isinstance(providers, dict):
                if "perplexity" in providers:
                    models_ran.append("perplexity")
                if "chatgpt" in providers:
                    models_ran.append("chatgpt")
                if "gemini" in providers:
                    models_ran.append("gemini")

            if not models_ran:
                model_name = str(run.get("model") or "").strip().lower()
                if model_name in {"openai", "chatgpt"}:
                    models_ran = ["chatgpt"]
                elif model_name in {"perplexity", "gemini", "claude"}:
                    models_ran = [model_name]
                elif model_name == "multi":
                    models_ran = ["perplexity", "chatgpt"]

            run["models_ran"] = models_ran
            phase5_runs.append(run)

    recent_activity = []
    if urls_col is not None:
        activity_query = {"user_id": user_id}
        if site_host:
            activity_query["url"] = {"$regex": site_host, "$options": "i"}

        ts_bounds = {}
        if cutoff_iso:
            ts_bounds["$gte"] = cutoff_iso
        if clear_cutoff:
            ts_bounds["$gt"] = max(clear_cutoff, ts_bounds.get("$gte", clear_cutoff))
        if ts_bounds:
            activity_query["timestamp"] = ts_bounds

        cursor = urls_col.find(
            activity_query,
            {"_id": 0, "url": 1, "phase": 1, "timestamp": 1},
        ).sort("timestamp", -1).limit(safe_limit)
        recent_activity = await cursor.to_list(length=safe_limit)

    total_pages = max(1, (total_runs + safe_limit - 1) // safe_limit)

    return {
        "phase5_runs": phase5_runs,
        "recent_activity": recent_activity,
        "pagination": {
            "page": safe_page,
            "limit": safe_limit,
            "total": total_runs,
            "total_pages": total_pages,
            "has_next": safe_page < total_pages,
            "has_prev": safe_page > 1,
        },
    }


@app.post("/api/user/history/clear")
async def api_user_history_clear(current_user: dict = Depends(get_current_user)):
    if user_history_meta_col is None:
        raise HTTPException(status_code=503, detail="history meta storage unavailable")

    now_iso = datetime.utcnow().isoformat()
    await user_history_meta_col.update_one(
        {"user_id": current_user["id"]},
        {
            "$set": {
                "user_id": current_user["id"],
                "cleared_at": now_iso,
                "updated_at": now_iso,
            },
            "$setOnInsert": {
                "created_at": now_iso,
            },
        },
        upsert=True,
    )

    return {"message": "History cleared for your view", "cleared_at": now_iso}


@app.get("/api/user/history/site-trend")
async def api_user_history_site_trend(site: str, range: str = "month", current_user: dict = Depends(get_current_user)):
    if phase5_jobs_col is None:
        raise HTTPException(status_code=503, detail="phase5 job storage unavailable")

    site_host = _normalize_site(site)
    if not site_host:
        raise HTTPException(status_code=400, detail="A valid site is required")

    query = {
        "user_id": current_user["id"],
        "url": {"$regex": site_host, "$options": "i"},
        "overall_score": {"$type": "number"},
    }

    cutoff_iso = _iso_from_range(range)
    if cutoff_iso:
        query["created_at"] = {"$gte": cutoff_iso}

    cursor = phase5_jobs_col.find(
        query,
        {
            "_id": 0,
            "job_id": 1,
            "url": 1,
            "status": 1,
            "overall_score": 1,
            "created_at": 1,
            "model": 1,
            "provider_scores": 1,
        },
    ).sort("created_at", 1)
    runs = await cursor.to_list(length=500)

    points = []
    for run in runs:
        created_at = run.get("created_at")
        score = run.get("overall_score")
        if created_at is None or not isinstance(score, (int, float)):
            continue

        providers = run.get("provider_scores") or {}
        models_ran = []
        if isinstance(providers, dict):
            if "perplexity" in providers:
                models_ran.append("perplexity")
            if "chatgpt" in providers:
                models_ran.append("chatgpt")
            if "gemini" in providers:
                models_ran.append("gemini")
        if not models_ran:
            model_name = str(run.get("model") or "").strip().lower()
            if model_name in {"openai", "chatgpt"}:
                models_ran = ["chatgpt"]
            elif model_name in {"perplexity", "gemini", "claude"}:
                models_ran = [model_name]
            elif model_name == "multi":
                models_ran = ["perplexity", "chatgpt"]

        points.append(
            {
                "job_id": run.get("job_id"),
                "url": run.get("url"),
                "status": run.get("status"),
                "score": round(float(score), 2),
                "created_at": created_at,
                "models_ran": models_ran,
            }
        )

    first_score = points[0]["score"] if points else None
    last_score = points[-1]["score"] if points else None
    delta = None
    direction = "flat"
    if first_score is not None and last_score is not None:
        delta = round(float(last_score) - float(first_score), 2)
        if delta > 0:
            direction = "up"
        elif delta < 0:
            direction = "down"

    avg_score = None
    if points:
        avg_score = round(sum(p["score"] for p in points) / len(points), 2)

    return {
        "site": site_host,
        "range": (range or "month").strip().lower(),
        "runs_count": len(points),
        "average_score": avg_score,
        "first_score": first_score,
        "last_score": last_score,
        "delta": delta,
        "direction": direction,
        "points": points,
    }


# --- Existing Data Routes ---

@app.post("/api/scrape", response_model=ScrapeResult)
async def api_scrape(request: ScrapeRequest, current_user: dict = Depends(get_current_user_optional)):
    try:
        started_at = datetime.utcnow()
        print(f"[API] /api/scrape started: {request.url}")
        try:
            doc = {
                "url": request.url, 
                "phase": "phase1", 
                "timestamp": datetime.utcnow().isoformat()
            }
            if current_user:
                doc["user_id"] = current_user["id"]
                doc["user_email"] = current_user["email"]
            await urls_col.insert_one(doc)
        except:
            pass
        scrape_timeout_seconds = int(os.getenv("PHASE1_SCRAPE_TIMEOUT_SECONDS", "420"))
        print(f"[API] /api/scrape timeout budget: {scrape_timeout_seconds}s")
        result = await asyncio.wait_for(
            asyncio.to_thread(_run_scrape_worker, request.url),
            timeout=scrape_timeout_seconds,
        )

        ai_debug = result.get("aiDebug", {}) if isinstance(result, dict) else {}
        ai_calls = ai_debug.get("calls", {}) if isinstance(ai_debug, dict) else {}
        ai_models = ai_debug.get("models", {}) if isinstance(ai_debug, dict) else {}
        for call_name in ["vision", "enrichment", "contact_fallback"]:
            if bool(ai_calls.get(call_name)):
                model_name = ai_models.get(call_name) if isinstance(ai_models, dict) else None
                await _log_ai_usage_event({
                    "feature": f"phase1_scrape_{call_name}",
                    "endpoint": "/api/scrape",
                    "url": request.url,
                    "user_id": current_user.get("id") if current_user else None,
                    "user_email": current_user.get("email") if current_user else None,
                    "model_name": model_name,
                    "model_family": "gemini" if isinstance(model_name, str) and model_name.lower().startswith("gemini") else "unknown",
                    "ai_calls_estimate": 1,
                    "details": {
                        "call_name": call_name,
                        "merged": ai_debug.get("merged", {}),
                        "blocked": ai_debug.get("blocked", {}),
                        "confidence": ai_debug.get("confidence", {}),
                    },
                })
        elapsed = (datetime.utcnow() - started_at).total_seconds()
        print(f"[API] /api/scrape completed in {elapsed:.1f}s: {request.url}")
        return result
    except asyncio.TimeoutError:
        print(f"[API] /api/scrape timeout: {request.url}")
        try:
            fallback = await asyncio.wait_for(
                asyncio.to_thread(_run_scrape_worker_core, request.url),
                timeout=120,
            )
            if isinstance(fallback, dict):
                warnings = fallback.get("warnings") if isinstance(fallback.get("warnings"), list) else []
                warnings.append("Returned reduced analysis after full scrape timeout. AI enrichment was skipped.")
                fallback["warnings"] = warnings
            print(f"[API] /api/scrape reduced fallback returned: {request.url}")
            return fallback
        except Exception as fallback_error:
            print(f"[API] /api/scrape fallback failed: {fallback_error}")
            raise HTTPException(status_code=504, detail="Scrape timed out and fallback also failed. Please retry.")
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/scan/compare", response_model=CompareResult)
async def api_scan_compare(request: CompareRequest, current_user: dict = Depends(get_current_user_optional)):
    print(f"\n[API] Received comparison request for primary URL: {request.primary_url}")
    print(f"[API] Competitors to check: {request.competitor_urls}")
    try:
        try:
            doc = {
                "url": request.primary_url, 
                "phase": "phase2_primary",
                "timestamp": datetime.utcnow().isoformat()
            }
            if current_user:
                doc["user_id"] = current_user["id"]
                doc["user_email"] = current_user["email"]
            await urls_col.insert_one(doc)

            for comp in request.competitor_urls:
                comp_doc = {
                    "url": comp, 
                    "phase": "phase2_competitor",
                    "timestamp": datetime.utcnow().isoformat()
                }
                if current_user:
                    comp_doc["user_id"] = current_user["id"]
                    comp_doc["user_email"] = current_user["email"]
                await urls_col.insert_one(comp_doc)
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
async def api_ai_insights(request: AiInsightsRequest, current_user: dict = Depends(get_current_user_optional)):
    try:
        started_at = datetime.utcnow()
        print(f"[API] /api/ai-insights started: {request.url}")
        insights = await asyncio.wait_for(get_ai_insights_multi(request.businessName, request.url), timeout=120)
        if not isinstance(insights, list):
            insights = []

        for insight in insights:
            model_name = insight.get("modelName") if isinstance(insight, dict) else None
            lowered_model = str(model_name or "").lower()
            if any(k in lowered_model for k in ["gpt", "o1", "o3", "o4"]):
                provider_hint = "openai"
            elif any(k in lowered_model for k in ["pplx", "sonar", "perplexity"]):
                provider_hint = "perplexity"
            else:
                provider_hint = "gemini"
            await _log_ai_usage_event({
                "feature": "phase1_ai_insights",
                "endpoint": "/api/ai-insights",
                "url": request.url,
                "user_id": current_user.get("id") if current_user else None,
                "user_email": current_user.get("email") if current_user else None,
                "model_name": model_name,
                "model_provider": provider_hint,
                "ai_calls_estimate": 1,
                "details": {
                    "isKnown": bool(insight.get("isKnown", False)) if isinstance(insight, dict) else None,
                    "platform_count": len(insight.get("platforms", [])) if isinstance(insight, dict) and isinstance(insight.get("platforms"), list) else 0,
                },
            })
        elapsed = (datetime.utcnow() - started_at).total_seconds()
        print(f"[API] /api/ai-insights completed in {elapsed:.1f}s: {request.url}")
        return AiInsightsResult(success=True, insights=insights)
    except asyncio.TimeoutError:
        print(f"[API] /api/ai-insights timeout: {request.url}")
        return AiInsightsResult(success=False, insights=[], error="AI insights timed out. Please retry.")
    except Exception as e:
        traceback.print_exc()
        return AiInsightsResult(success=False, insights=[], error=str(e))

@app.post("/api/scan/content", response_model=ContentAnalysisResponse)
async def api_scan_content(request: ContentAnalysisRequest, current_user: dict = Depends(get_current_user_optional)):
    print(f"\n[API] Received content analysis request for URL: {request.url}")
    try:
        try:
            doc = {
                "url": request.url, 
                "phase": "phase3",
                "timestamp": datetime.utcnow().isoformat()
            }
            if current_user:
                doc["user_id"] = current_user["id"]
                doc["user_email"] = current_user["email"]
            await urls_col.insert_one(doc)
        except:
            pass
        result = await analyze_url_content(request.url)
        return result
    except Exception as e:
        print(f"[API] ERROR in content analysis: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/blogs/analyze", response_model=BlogAnalysisResponse)
async def api_blogs_analyze(request: BlogAnalyzeRequest, current_user: dict = Depends(get_current_user_optional)):
    text = str(request.text or "").strip()
    if not text:
        return BlogAnalysisResponse(success=False, error="Blog text is required.")

    model = (request.model or "perplexity").strip().lower()
    if model != "perplexity":
        raise HTTPException(status_code=400, detail="Only Perplexity is supported for now.")

    base = _build_blog_base_analysis(text, max(0, int(request.attachment_count or 0)))
    metrics = base.get("metrics", {})

    try:
        llm = await get_blog_analysis_perplexity(text, metrics)
    except Exception as e:
        print(f"[API] blog analysis error: {e}")
        llm = {}

    strengths = llm.get("strengths") if isinstance(llm.get("strengths"), list) else base["strengths"]
    suggestions = llm.get("suggestions") if isinstance(llm.get("suggestions"), list) else base["suggestions"]
    weak_spots = llm.get("weakSpots") if isinstance(llm.get("weakSpots"), list) else [
        s for s in suggestions if "Expand" in s or "H2 and H3" in s or "Split long walls" in s
    ]
    improvements = llm.get("improvements") if isinstance(llm.get("improvements"), list) else [
        s for s in suggestions if s not in weak_spots
    ]
    overview = str(llm.get("overview") or base["overview"])
    summary = str(llm.get("summary") or base["summary"])

    result = BlogAnalysis(
        score=int(metrics.get("score", 0)),
        grade=str(metrics.get("grade", "Needs Work")),
        overview=overview,
        summary=summary,
        wordCount=int(metrics.get("wordCount", 0)),
        paragraphCount=int(metrics.get("paragraphCount", 0)),
        headingCount=int(metrics.get("headingCount", 0)),
        listCount=int(metrics.get("listCount", 0)),
        linkCount=int(metrics.get("linkCount", 0)),
        ctaCount=int(metrics.get("ctaCount", 0)),
        readability=int(metrics.get("readability", 0)),
        structure=int(metrics.get("structure", 0)),
        seo=int(metrics.get("seo", 0)),
        engagement=int(metrics.get("engagement", 0)),
        strengths=[str(s) for s in strengths if str(s).strip()],
        weakSpots=[str(s) for s in weak_spots if str(s).strip()],
        improvements=[str(s) for s in improvements if str(s).strip()],
        suggestions=[str(s) for s in suggestions if str(s).strip()],
    )

    try:
        await _log_ai_usage_event({
            "feature": "blog_seo_analyzer",
            "endpoint": "/api/blogs/analyze",
            "user_id": current_user.get("id") if current_user else None,
            "user_email": current_user.get("email") if current_user else None,
            "model_name": llm.get("modelUsed") if isinstance(llm, dict) else None,
            "model_provider": "perplexity",
            "ai_calls_estimate": 1,
            "details": {
                "word_count": result.wordCount,
                "attachment_count": int(metrics.get("attachmentCount", 0)),
            },
        })
    except Exception as e:
        print(f"[API] blog usage logging failed: {e}")

    return BlogAnalysisResponse(success=True, result=result)
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

@app.delete("/api/wishlist")
async def api_delete_wishlist(email: str):
    try:
        await wishlist_col.delete_one({"email": email})
        return {"success": True}
    except Exception as e:
        print(f"[API] ERROR deleting wishlist email {email}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/track-url")
async def api_track_url(request: TrackUrlRequest, current_user: dict = Depends(get_current_user_optional)):
    try:
        doc = {
            "url": request.url, 
            "phase": request.phase,
            "timestamp": datetime.utcnow().isoformat()
        }
        if current_user:
            doc["user_id"] = current_user["id"]
            doc["user_email"] = current_user["email"]
        await urls_col.insert_one(doc)
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

@app.delete("/api/track-url")
async def api_delete_track_url(url: str, phase: str):
    try:
        await urls_col.delete_one({"url": url, "phase": phase})
        return {"success": True}
    except Exception as e:
        print(f"[API] ERROR deleting tracking url {url}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- Admin Dash Routes ---

from models import UserRoleUpdateRequest

@app.get("/api/admin/users")
async def admin_get_users(admin: dict = Depends(get_current_admin_user)):
    try:
        users = []
        async for u in users_col.find({}):
            users.append({
                "id": str(u["_id"]),
                "name": u.get("name"),
                "email": u.get("email"),
                "role": u.get("role", "user"),
                "status": u.get("status", "active"),
                "created_at": u.get("created_at")
            })
        return {"success": True, "users": users}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/admin/users/{user_id}/role")
async def admin_update_user_role(user_id: str, request: UserRoleUpdateRequest, admin: dict = Depends(get_current_admin_user)):
    try:
        if admin["id"] == user_id:
            raise HTTPException(status_code=400, detail="Admins cannot change their own role")
        
        upd = await users_col.update_one(
            {"_id": ObjectId(user_id)},
            {"$set": {"role": request.role}}
        )
        if upd.matched_count == 0:
            raise HTTPException(status_code=404, detail="User not found")
        return {"success": True, "role": request.role}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/admin/searches")
async def admin_get_searches(admin: dict = Depends(get_current_admin_user)):
    try:
        searches = []
        async for doc in urls_col.find({}).sort("timestamp", -1):
            searches.append({
                "id": str(doc["_id"]),
                "url": doc["url"],
                "phase": doc.get("phase", "unknown"),
                "timestamp": doc.get("timestamp"),
                "user_id": doc.get("user_id"),
                "user_email": doc.get("user_email")
            })
        return {"success": True, "searches": searches}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/admin/searches")
async def admin_delete_searches(
    from_date: str | None = None,
    to_date: str | None = None,
    admin: dict = Depends(get_current_admin_user),
):
    try:
        if not from_date and not to_date:
            result = await urls_col.delete_many({})
            return {"success": True, "deleted_count": int(result.deleted_count)}

        from_dt = datetime.fromisoformat(from_date) if from_date else None
        to_dt = (datetime.fromisoformat(to_date) + timedelta(days=1)) if to_date else None

        def _parse_ts(value):
            if isinstance(value, datetime):
                return value
            if isinstance(value, str):
                try:
                    return datetime.fromisoformat(value)
                except Exception:
                    return None
            return None

        ids_to_delete = []
        async for doc in urls_col.find({}, {"_id": 1, "timestamp": 1}):
            ts = _parse_ts(doc.get("timestamp"))
            if ts is None:
                continue
            if from_dt and ts < from_dt:
                continue
            if to_dt and ts >= to_dt:
                continue
            ids_to_delete.append(doc.get("_id"))

        if not ids_to_delete:
            return {"success": True, "deleted_count": 0}

        result = await urls_col.delete_many({"_id": {"$in": ids_to_delete}})
        return {"success": True, "deleted_count": int(result.deleted_count)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/admin/wishlist-full")
async def admin_get_wishlist_full(admin: dict = Depends(get_current_admin_user)):
    try:
        emails = []
        # Support full documents, with potentially a logged created_at date
        async for doc in wishlist_col.find({}):
            emails.append({
                "id": str(doc["_id"]),
                "email": doc["email"],
                "timestamp": doc.get("timestamp")
            })
        return {"success": True, "wishlist": emails}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/admin/ai-usage")
async def admin_get_ai_usage(
    limit: int = 200,
    from_date: str | None = None,
    to_date: str | None = None,
    admin: dict = Depends(get_current_admin_user),
):
    try:
        if ai_usage_col is None:
            return {
                "success": True,
                "summary": {
                    "total_events": 0,
                    "total_ai_calls_estimate": 0,
                    "unique_users": 0,
                    "by_feature": {},
                    "by_model": {},
                },
                "recent": [],
            }

        bounded_limit = max(20, min(1000, int(limit)))
        from_dt = datetime.fromisoformat(from_date) if from_date else None
        to_dt = (datetime.fromisoformat(to_date) + timedelta(days=1)) if to_date else None

        def _parse_ts(value):
            if isinstance(value, datetime):
                return value
            if isinstance(value, str):
                try:
                    return datetime.fromisoformat(value)
                except Exception:
                    return None
            return None

        scan_limit = max(bounded_limit, 3000 if (from_dt or to_dt) else bounded_limit)

        events = []
        async for doc in ai_usage_col.find({}).sort("timestamp", -1).limit(scan_limit):
            event_ts = _parse_ts(doc.get("timestamp"))
            if from_dt and (event_ts is None or event_ts < from_dt):
                continue
            if to_dt and (event_ts is None or event_ts >= to_dt):
                continue
            events.append({
                "id": str(doc.get("_id")),
                "timestamp": (event_ts.isoformat() if event_ts else doc.get("timestamp_iso") or doc.get("timestamp")),
                "feature": doc.get("feature"),
                "endpoint": doc.get("endpoint"),
                "provider": doc.get("provider", "google"),
                "model_family": doc.get("model_family", "gemini"),
                "model_name": doc.get("model_name"),
                "user_id": doc.get("user_id"),
                "user_email": doc.get("user_email"),
                "url": doc.get("url"),
                "ai_calls_estimate": doc.get("ai_calls_estimate", 0),
                "details": doc.get("details", {}),
            })
            if len(events) >= bounded_limit:
                break

        total_events = len(events)
        total_ai_calls = sum(int(e.get("ai_calls_estimate", 0) or 0) for e in events)
        unique_users = len({e.get("user_email") for e in events if e.get("user_email")})

        by_feature = {}
        by_model = {}
        by_model_family = {}
        by_user = {}
        for e in events:
            feature = e.get("feature") or "unknown"
            model = e.get("model_name") or "unknown"
            model_family = e.get("model_family") or "unknown"
            user = e.get("user_email") or "anonymous"
            calls = int(e.get("ai_calls_estimate", 0) or 0)

            by_feature[feature] = by_feature.get(feature, 0) + calls
            by_model[model] = by_model.get(model, 0) + calls
            by_model_family[model_family] = by_model_family.get(model_family, 0) + calls
            by_user[user] = by_user.get(user, 0) + calls

        top_users = [
            {"user_email": k, "ai_calls_estimate": v}
            for k, v in sorted(by_user.items(), key=lambda kv: kv[1], reverse=True)[:20]
        ]

        return {
            "success": True,
            "summary": {
                "total_events": total_events,
                "total_ai_calls_estimate": total_ai_calls,
                "unique_users": unique_users,
                "by_feature": by_feature,
                "by_model": by_model,
                "by_model_family": by_model_family,
                "top_users": top_users,
            },
            "recent": events,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/admin/ai-usage")
async def admin_clear_ai_usage(admin: dict = Depends(get_current_admin_user)):
    try:
        if ai_usage_col is None:
            return {"success": True, "deleted_count": 0}
        result = await ai_usage_col.delete_many({})
        return {"success": True, "deleted_count": int(result.deleted_count)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==============================================================================
# PHASE 5 ROUTES
# ==============================================================================
@app.post("/api/phase5/generate-questions", response_model=Phase5QuestionsResponse)
async def api_phase5_generate_questions(req: Phase5QuestionsRequest, current_user: dict = Depends(get_current_user_optional)):
    try:
        questions = await generate_brand_questions(req.url)
        configured_model = (os.getenv("PERPLEXITY_MODEL_PHASE5") or "sonar-pro").strip() or None
        await _log_ai_usage_event({
            "feature": "phase5_generate_questions",
            "endpoint": "/api/phase5/generate-questions",
            "url": req.url,
            "user_id": current_user.get("id") if current_user else None,
            "user_email": current_user.get("email") if current_user else None,
            "model_name": configured_model,
            "model_provider": "perplexity",
            "ai_calls_estimate": 1,
            "details": {
                "questions_count": len(questions or []),
            },
        })
        return {"questions": questions}
    except ValueError as e:
        print(f"[Phase5] generate-questions validation failed: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail="Questions cannot be generated at this moment. Please try again or refresh.",
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=503,
            detail="Questions cannot be generated at this moment. Please try again or refresh.",
        )

@app.post("/api/phase5/analyze", response_model=Phase5AnalyzeResponse)
async def api_phase5_analyze(req: Phase5AnalyzeRequest, current_user: dict = Depends(get_current_user_optional)):
    try:
        # q.model_dump() turns the pydantic model into a dictionary
        questions_dicts = [q.model_dump() for q in req.questions]
        tasks = [_run_with_backoff(req.url, q, model_provider="perplexity") for q in questions_dicts]
        response_items = await asyncio.gather(*tasks)
        results = {
            item["id"]: item
            for item in response_items
            if isinstance(item, dict) and isinstance(item.get("id"), str)
        }
        configured_model = (os.getenv("PERPLEXITY_MODEL_PHASE5") or "sonar-pro").strip() or None
        await _log_ai_usage_event({
            "feature": "phase5_analyze_direct",
            "endpoint": "/api/phase5/analyze",
            "url": req.url,
            "user_id": current_user.get("id") if current_user else None,
            "user_email": current_user.get("email") if current_user else None,
            "model_name": configured_model,
            "model_provider": "perplexity",
            "ai_calls_estimate": max(1, len(questions_dicts)),
            "details": {
                "questions_count": len(questions_dicts),
            },
        })
        return {"results": results}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/phase5/analyze-single", response_model=Phase5AnalyzeSingleResponse)
async def api_phase5_analyze_single(req: Phase5AnalyzeSingleRequest, current_user: dict = Depends(get_current_user_optional)):
    try:
        result = await analyze_single_question(req.url, req.question.model_dump(), model_provider="perplexity")
        configured_model = (os.getenv("PERPLEXITY_MODEL_PHASE5") or "sonar-pro").strip() or None
        await _log_ai_usage_event({
            "feature": "phase5_analyze_single",
            "endpoint": "/api/phase5/analyze-single",
            "url": req.url,
            "user_id": current_user.get("id") if current_user else None,
            "user_email": current_user.get("email") if current_user else None,
            "model_name": configured_model,
            "model_provider": "perplexity",
            "ai_calls_estimate": 1,
            "details": {
                "question_id": req.question.id,
            },
        })
        return result
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


async def _process_phase5_job(job_doc: dict):
    job_id = job_doc["job_id"]
    job_type = job_doc.get("job_type", "core")
    model_provider = str(job_doc.get("model", "perplexity") or "perplexity").strip().lower()
    if model_provider == "gemini":
        await phase5_jobs_col.update_one(
            {"job_id": job_id},
            {
                "$set": {
                    "status": "failed",
                    "current_question_id": None,
                    "error": "Gemini is temporarily disabled for Phase 5. Use OpenAI or Perplexity.",
                    "updated_at": datetime.utcnow().isoformat(),
                }
            },
        )
        return
    include_competitors = job_type == "deep"
    print(f"[Phase5] worker start job_id={job_id} type={job_type} provider={model_provider}")
    try:
        async def _is_cancelled() -> bool:
            latest = await phase5_jobs_col.find_one(
                {"job_id": job_id},
                {"status": 1, "_id": 0},
            )
            return not latest or latest.get("status") == "cancelled"

        def _compute_provider_scores(results_map: dict[str, dict]) -> tuple[dict, float | None]:
            providers = ["perplexity", "chatgpt"]
            score_map: dict[str, dict] = {}
            for provider_name in providers:
                score_map[provider_name] = compute_provider_score(results_map, provider_name)
            numeric_scores = [float(v.get("score", 0)) for v in score_map.values() if isinstance(v, dict)]
            overall = round(sum(numeric_scores) / len(numeric_scores), 2) if numeric_scores else None
            return score_map, overall

        questions = list(job_doc.get("questions", []))
        seed_results = job_doc.get("seed_results", {}) or {}
        accumulated_results = {}

        if include_competitors:
            deep_competitors = await generate_deep_competitor_scores(
                url=job_doc["url"],
                questions=questions,
                seed_results=seed_results,
            )

            await phase5_jobs_col.update_one(
                {"job_id": job_id, "status": {"$ne": "cancelled"}},
                {
                    "$set": {
                        "results": seed_results if isinstance(seed_results, dict) else {},
                        "processed": len(questions),
                        "deep_competitors": deep_competitors,
                        "status": "completed",
                        "current_question_id": None,
                        "updated_at": datetime.utcnow().isoformat(),
                    }
                },
            )

            async def _finalize_brand_summary_async():
                try:
                    brand_summary = await generate_brand_perception_summary(
                        url=job_doc["url"],
                        questions=questions,
                        results=seed_results if isinstance(seed_results, dict) else {},
                    )
                    await phase5_jobs_col.update_one(
                        {"job_id": job_id, "status": "completed"},
                        {
                            "$set": {
                                "brand_summary": brand_summary,
                                "updated_at": datetime.utcnow().isoformat(),
                            }
                        },
                    )
                except Exception:
                    traceback.print_exc()

            asyncio.create_task(_finalize_brand_summary_async())
            return

        lock = asyncio.Lock()
        queue: asyncio.Queue[dict] = asyncio.Queue()
        for q in questions:
            queue.put_nowait(q)

        # Background finalization: start it immediately in parallel with workers
        # so it doesn't block question analysis. It will use its own Perplexity
        # probes if seed_results is empty/incomplete.
        always_run_deep = os.getenv("PHASE5_ALWAYS_RUN_DEEP", "false").strip().lower() == "true"
        finalize_task = None
        if always_run_deep or include_competitors:
            async def _finalize_background_task():
                print(f"[Phase5] background competitor finalizing job_id={job_id}")
                try:
                    # Run competitor discovery early using Perplexity probes.
                    # It will return 4 external competitors. Target site is added later.
                    deep_competitors = await generate_deep_competitor_scores(
                        url=job_doc["url"],
                        questions=questions,
                        seed_results={}, # Intentionally empty to force fast probe
                    )
                    
                    # Update deep competitors immediately so UI has early partial data
                    await phase5_jobs_col.update_one(
                        {"job_id": job_id, "status": {"$ne": "cancelled"}},
                        {
                            "$set": {
                                "deep_competitors": deep_competitors,
                                "updated_at": datetime.utcnow().isoformat(),
                            }
                        },
                    )
                    print(f"[Phase5] background competitor finalizing done job_id={job_id}")
                    return deep_competitors
                except Exception:
                    traceback.print_exc()
                    return []

            finalize_task = asyncio.create_task(_finalize_background_task())

        async def _worker_run():
            while True:
                if await _is_cancelled():
                    break

                try:
                    q = queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

                qid = q["id"]
                print(f"[Phase5] run qid={qid} job_id={job_id} provider={model_provider}")
                await phase5_jobs_col.update_one(
                    {"job_id": job_id, "status": {"$ne": "cancelled"}},
                    {
                        "$set": {
                            "current_question_id": qid,
                            "updated_at": datetime.utcnow().isoformat(),
                        }
                    },
                )

                provider_for_q = str(job_doc.get("model", "perplexity") or "perplexity").strip().lower()
                if provider_for_q == "openai":
                    per_question_timeout = PHASE5_QUESTION_TIMEOUT_OPENAI_SEC
                elif provider_for_q == "perplexity":
                    per_question_timeout = PHASE5_QUESTION_TIMEOUT_PERPLEXITY_SEC
                elif provider_for_q == "multi":
                    per_question_timeout = max(
                        PHASE5_QUESTION_TIMEOUT_OPENAI_SEC,
                        PHASE5_QUESTION_TIMEOUT_PERPLEXITY_SEC,
                        PHASE5_QUESTION_TIMEOUT_GEMINI_SEC,
                    )
                else:
                    per_question_timeout = PHASE5_QUESTION_TIMEOUT_GEMINI_SEC

                try:
                    if provider_for_q == "multi":
                        result = await asyncio.wait_for(
                            analyze_single_question_multi(
                                url=job_doc["url"],
                                question=q,
                                include_competitors=include_competitors,
                            ),
                            timeout=max(10, per_question_timeout),
                        )
                    else:
                        result = await asyncio.wait_for(
                            _run_with_backoff(
                                job_doc["url"],
                                q,
                                model_provider=provider_for_q,
                                include_competitors=include_competitors,
                            ),
                            timeout=max(10, per_question_timeout),
                        )
                except asyncio.TimeoutError:
                    print(
                        f"[Phase5] qid={qid} job_id={job_id} provider={provider_for_q} "
                        f"hard-timeout after {per_question_timeout}s"
                    )
                    result = {
                        "id": qid,
                        "status": "Not Mentioned",
                        "position": None,
                        "sources": [],
                        "source_urls": [],
                        "references": [],
                        "competitors": [],
                        "competitor_scores": [],
                        "reasoning": f"Timed out after {per_question_timeout}s",
                    }
                    if provider_for_q == "multi":
                        result = {
                            "id": qid,
                            "providers": {
                                "gemini": {"mentioned": False, "position": None, "sources": [], "cited": False, "status": "Not Mentioned"},
                                "perplexity": {"mentioned": False, "position": None, "sources": [], "cited": False, "status": "Not Mentioned"},
                                "chatgpt": {"mentioned": False, "position": None, "sources": [], "cited": False, "status": "Not Mentioned"},
                            },
                        }

                async with lock:
                    accumulated_results[qid] = result

                running_scores, running_overall = _compute_provider_scores(accumulated_results) if provider_for_q == "multi" else ({}, None)

                print(
                    f"[Phase5] done qid={qid} job_id={job_id} provider={model_provider} "
                    f"status={result.get('status', 'multi')} pos={result.get('position')}"
                )

                await phase5_jobs_col.update_one(
                    {"job_id": job_id, "status": {"$ne": "cancelled"}},
                    {
                        "$set": {
                            f"results.{qid}": result,
                            "provider_scores": running_scores,
                            "overall_score": running_overall,
                            "updated_at": datetime.utcnow().isoformat(),
                        },
                        "$inc": {"processed": 1},
                    },
                )
                queue.task_done()

        workers = [
            asyncio.create_task(_worker_run())
            for _ in range(max(1, PHASE5_JOB_PARALLELISM))
        ]
        await asyncio.gather(*workers)
        
        deep_competitors = []
        # Ensure the background finalization task (started earlier) is finished
        if finalize_task:
            try:
                # Give it a small extra timeout just in case, but it should be done.
                deep_competitors = await asyncio.wait_for(finalize_task, timeout=40)
            except Exception as e:
                print(f"[Phase5] background finalization wait error: {e}")

        processed_count = len(accumulated_results)

        if await _is_cancelled():
            await phase5_jobs_col.update_one(
                {"job_id": job_id},
                {
                    "$set": {
                        "current_question_id": None,
                        "updated_at": datetime.utcnow().isoformat(),
                    }
                },
            )
            return

        # Calculate accurate target score now that all questions are finished
        target_score = _estimate_target_visibility_score(accumulated_results)
        domain = _normalize_domain(job_doc["url"])
        
        # Remove any existing target site entry from deep_competitors
        deep_competitors = [c for c in (deep_competitors or []) if c.get("domain") != domain]
        
        # Append target site with the fully accurate score
        deep_competitors.append({
            "domain": domain,
            "position": None,
            "score": target_score,
            "evidence": "Your site score from visibility and rank consistency across analyzed prompts.",
        })

        # Generate accurate brand summary now that questions are finished
        brand_summary = None
        if always_run_deep or include_competitors:
            try:
                brand_summary = await generate_brand_perception_summary(
                    url=job_doc["url"],
                    questions=questions,
                    results=accumulated_results,
                )
            except Exception as e:
                print(f"[Phase5] brand summary error: {e}")

        # Finalize job status
        final_scores, final_overall = _compute_provider_scores(accumulated_results) if model_provider == "multi" else ({}, None)
        await phase5_jobs_col.update_one(
            {"job_id": job_id},
            {
                "$set": {
                    "status": "completed",
                    "current_question_id": None,
                    "provider_scores": final_scores,
                    "overall_score": final_overall,
                    "deep_competitors": deep_competitors,
                    "brand_summary": brand_summary,
                    "updated_at": datetime.utcnow().isoformat(),
                }
            },
        )
        print(f"[Phase5] worker completed job_id={job_id} provider={model_provider}")

        await _log_ai_usage_event({
            "feature": "phase5_job_completed",
            "endpoint": "/api/phase5/start-job",
            "url": job_doc.get("url"),
            "user_id": job_doc.get("user_id"),
            "user_email": job_doc.get("user_email"),
            "model_provider": model_provider,
            "ai_calls_estimate": int(processed_count),
            "details": {
                "job_id": job_id,
                "job_type": job_type,
                "model_provider": model_provider,
                "processed": processed_count,
                "total": len(questions),
            },
        })
    except Phase5RateLimitError as e:
        processed_count = len(locals().get("accumulated_results", {}) or {})
        await _log_ai_usage_event({
            "feature": "phase5_job_failed_rate_limit",
            "endpoint": "/api/phase5/start-job",
            "url": job_doc.get("url"),
            "user_id": job_doc.get("user_id"),
            "user_email": job_doc.get("user_email"),
            "model_provider": model_provider,
            "ai_calls_estimate": int(processed_count),
            "details": {
                "job_id": job_id,
                "job_type": job_type,
                "model_provider": model_provider,
                "processed": processed_count,
                "error": str(e),
            },
        })
        await phase5_jobs_col.update_one(
            {"job_id": job_id},
            {
                "$set": {
                    "status": "failed",
                    "current_question_id": None,
                    "error": str(e),
                    "updated_at": datetime.utcnow().isoformat(),
                }
            },
        )
    except Exception as e:
        processed_count = len(locals().get("accumulated_results", {}) or {})
        await _log_ai_usage_event({
            "feature": "phase5_job_failed",
            "endpoint": "/api/phase5/start-job",
            "url": job_doc.get("url"),
            "user_id": job_doc.get("user_id"),
            "user_email": job_doc.get("user_email"),
            "model_provider": model_provider,
            "ai_calls_estimate": int(processed_count),
            "details": {
                "job_id": job_id,
                "job_type": job_type,
                "model_provider": model_provider,
                "processed": processed_count,
                "error": str(e),
            },
        })
        traceback.print_exc()
        await phase5_jobs_col.update_one(
            {"job_id": job_id},
            {
                "$set": {
                    "status": "failed",
                    "current_question_id": None,
                    "error": str(e),
                    "updated_at": datetime.utcnow().isoformat(),
                }
            },
        )


async def _phase5_worker_loop():
    while True:
        try:
            redis_client = getattr(app.state, "phase5_redis", None)

            claimed = None
            if redis_client is not None:
                queue_item = await redis_client.blpop(PHASE5_REDIS_QUEUE_KEY, timeout=5)
                if queue_item:
                    _, raw_job_id = queue_item
                    queued_job_id = raw_job_id.decode("utf-8")
                    claimed = await phase5_jobs_col.find_one_and_update(
                        {"job_id": queued_job_id, "status": "queued"},
                        {
                            "$set": {
                                "status": "running",
                                "worker_id": PHASE5_WORKER_ID,
                                "updated_at": datetime.utcnow().isoformat(),
                            }
                        },
                        return_document=ReturnDocument.AFTER,
                    )
            if claimed is None:
                claimed = await phase5_jobs_col.find_one_and_update(
                    {"status": "queued"},
                    {
                        "$set": {
                            "status": "running",
                            "worker_id": PHASE5_WORKER_ID,
                            "updated_at": datetime.utcnow().isoformat(),
                        }
                    },
                    sort=[("queue_priority", 1), ("created_at", 1)],
                    return_document=ReturnDocument.AFTER,
                )

            if not claimed:
                await asyncio.sleep(PHASE5_WORKER_POLL_INTERVAL)
                continue

            await _process_phase5_job(claimed)
        except asyncio.CancelledError:
            break
        except Exception:
            traceback.print_exc()
            await asyncio.sleep(PHASE5_WORKER_POLL_INTERVAL)


async def _phase5_kick_job_processing(job_id: str):
    """Best-effort direct claim path so newly queued jobs are not stranded."""
    if phase5_jobs_col is None:
        return
    try:
        claimed = await phase5_jobs_col.find_one_and_update(
            {"job_id": job_id, "status": "queued"},
            {
                "$set": {
                    "status": "running",
                    "worker_id": PHASE5_WORKER_ID,
                    "updated_at": datetime.utcnow().isoformat(),
                }
            },
            return_document=ReturnDocument.AFTER,
        )
        if not claimed:
            return
        print(f"[Phase5] kick start job_id={job_id} provider={claimed.get('model')}")
        await _process_phase5_job(claimed)
    except Exception:
        traceback.print_exc()


async def _phase5_try_start_immediately(job_id: str) -> None:
    """Try to claim a newly queued job right away and process in background."""
    if phase5_jobs_col is None:
        return
    try:
        claimed = await phase5_jobs_col.find_one_and_update(
            {"job_id": job_id, "status": "queued"},
            {
                "$set": {
                    "status": "running",
                    "worker_id": PHASE5_WORKER_ID,
                    "updated_at": datetime.utcnow().isoformat(),
                }
            },
            return_document=ReturnDocument.AFTER,
        )
        if not claimed:
            return
        print(f"[Phase5] immediate start job_id={job_id} provider={claimed.get('model')}")
        asyncio.create_task(_process_phase5_job(claimed))
    except Exception:
        traceback.print_exc()


def _phase5_request_hash(
    url: str,
    questions: list[dict],
    job_type: str,
    model: str,
    seed_results: dict | None = None,
) -> str:
    normalized_questions = []
    for q in questions:
        qid = str((q or {}).get("id", "")).strip()
        text = str((q or {}).get("text", "")).strip()
        if not text:
            continue
        normalized_questions.append({"id": qid, "text": text})

    normalized_questions.sort(key=lambda item: (item.get("id", ""), item.get("text", "")))

    payload = {
        "url": str(url or "").strip().lower(),
        "job_type": str(job_type or "core").strip().lower(),
        "model": str(model or "perplexity").strip().lower(),
        "questions": normalized_questions,
    }

    if payload["job_type"] == "deep":
        payload["seed_results"] = seed_results or {}

    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _ts_within_cache_window(ts_value: str | None) -> bool:
    if not ts_value:
        return False
    try:
        updated_at = datetime.fromisoformat(ts_value)
    except Exception:
        return False
    age_seconds = (datetime.utcnow() - updated_at).total_seconds()
    return age_seconds <= max(60, PHASE5_CACHE_REUSE_SECONDS)


async def _find_reusable_phase5_job(request_hash: str, job_type: str) -> dict | None:
    if phase5_jobs_col is None:
        return None

    cursor = phase5_jobs_col.find(
        {
            "request_hash": request_hash,
            "job_type": job_type,
            "status": {"$in": ["queued", "running", "finalizing", "completed"]},
        },
        {"_id": 0},
    ).sort("updated_at", -1).limit(5)

    reusable_completed = None
    async for job in cursor:
        status = (job or {}).get("status")
        if status in {"queued", "finalizing"}:
            return job
        if status == "running":
            if _ts_within_cache_window(job.get("updated_at")):
                return job
            stale_job_id = job.get("job_id")
            if PHASE5_RECOVER_STALE_RUNNING:
                # Optional mode: recover stale running jobs by re-queueing them.
                await phase5_jobs_col.update_one(
                    {"job_id": stale_job_id, "status": "running"},
                    {
                        "$set": {
                            "status": "queued",
                            "worker_id": None,
                            "current_question_id": None,
                            "updated_at": datetime.utcnow().isoformat(),
                        }
                    },
                )
                refreshed = await phase5_jobs_col.find_one(
                    {"job_id": stale_job_id},
                    {"_id": 0},
                )
                if refreshed:
                    print(f"[Phase5] requeued stale running job_id={refreshed.get('job_id')}")
                    return refreshed
            else:
                # Default mode: mark stale running jobs failed so they don't flood workers after restart.
                await phase5_jobs_col.update_one(
                    {"job_id": stale_job_id, "status": "running"},
                    {
                        "$set": {
                            "status": "failed",
                            "current_question_id": None,
                            "error": "stale_running_job_recovered_as_failed",
                            "updated_at": datetime.utcnow().isoformat(),
                        }
                    },
                )
                print(f"[Phase5] marked stale running job_id={stale_job_id} as failed")
            continue
        if status == "completed" and _ts_within_cache_window(job.get("updated_at")):
            reusable_completed = job
            break

    return reusable_completed


@app.on_event("startup")
async def _phase5_worker_startup():
    if phase5_jobs_col is None:
        app.state.phase5_worker_tasks = []
        app.state.phase5_redis = None
        return

    app.state.phase5_redis = None
    loop = asyncio.get_running_loop()
    app.state.phase5_executor = ThreadPoolExecutor(max_workers=max(4, PHASE5_MODEL_MAX_THREADS))
    loop.set_default_executor(app.state.phase5_executor)

    if PHASE5_REDIS_URL:
        try:
            redis_client = aioredis.from_url(PHASE5_REDIS_URL, decode_responses=False)
            await redis_client.ping()
            app.state.phase5_redis = redis_client
            print(f"[Phase5] Redis queue enabled at {PHASE5_REDIS_URL}")
        except Exception:
            print(f"[Phase5] Redis unavailable ({PHASE5_REDIS_URL}). Falling back to Mongo polling workers.")
            app.state.phase5_redis = None

    # Indexes tuned to current Phase 5 query/update patterns.
    try:
        # Direct lookups
        await phase5_jobs_col.create_index("job_id", unique=True)

        # Worker claim path: find queued jobs sorted by priority and creation time.
        await phase5_jobs_col.create_index([
            ("status", 1),
            ("queue_priority", 1),
            ("created_at", 1),
        ])

        # Reuse path: request hash + job type + status sorted by latest update.
        await phase5_jobs_col.create_index([
            ("request_hash", 1),
            ("job_type", 1),
            ("status", 1),
            ("updated_at", -1),
        ])

        # User history and trend listing paths.
        await phase5_jobs_col.create_index([
            ("user_id", 1),
            ("created_at", -1),
        ])
        await phase5_jobs_col.create_index([
            ("user_id", 1),
            ("status", 1),
            ("created_at", -1),
        ])

        # Stale job sweeps and startup recovery updates.
        await phase5_jobs_col.create_index([
            ("status", 1),
            ("updated_at", 1),
        ])

        # Existing standalone index kept for compatibility with older lookups.
        await phase5_jobs_col.create_index("request_hash")

        # Related collections used by history endpoints.
        if urls_col is not None:
            await urls_col.create_index([
                ("user_id", 1),
                ("timestamp", -1),
            ])
        if user_history_meta_col is not None:
            await user_history_meta_col.create_index("user_id", unique=True)
    except Exception:
        print("[Phase5] warning: index creation failed; continuing without blocking startup")
        traceback.print_exc()

    # Cost-safety default: do not auto-resume previously queued/in-progress jobs after restart
    # unless explicitly enabled via env.
    if not PHASE5_RESUME_QUEUED_ON_STARTUP:
        try:
            startup_failed = await phase5_jobs_col.update_many(
                {"status": {"$in": ["queued", "running", "finalizing"]}},
                {
                    "$set": {
                        "status": "failed",
                        "worker_id": None,
                        "current_question_id": None,
                        "error": "startup_queue_reset",
                        "updated_at": datetime.utcnow().isoformat(),
                    }
                },
            )
            if int(startup_failed.modified_count or 0) > 0:
                print(f"[Phase5] startup reset previous queued/running jobs={startup_failed.modified_count}")
        except Exception:
            traceback.print_exc()

    # Hard safety gate: while Gemini is disabled for Phase 5, fail stale Gemini jobs on startup.
    try:
        startup_gemini_failed = await phase5_jobs_col.update_many(
            {
                "model": "gemini",
                "status": {"$in": ["queued", "running", "finalizing"]},
            },
            {
                "$set": {
                    "status": "failed",
                    "worker_id": None,
                    "current_question_id": None,
                    "error": "gemini_disabled_phase5",
                    "updated_at": datetime.utcnow().isoformat(),
                }
            },
        )
        if int(startup_gemini_failed.modified_count or 0) > 0:
            print(f"[Phase5] startup failed stale gemini jobs={startup_gemini_failed.modified_count}")
    except Exception:
        traceback.print_exc()

    stale_running_cutoff_iso = (datetime.utcnow() - timedelta(seconds=max(30, PHASE5_STALE_RUNNING_SECONDS))).isoformat()
    stale_queued_cutoff_iso = (datetime.utcnow() - timedelta(seconds=max(120, PHASE5_STALE_QUEUED_SECONDS))).isoformat()
    try:
        if PHASE5_RECOVER_STALE_RUNNING:
            stale_reset = await phase5_jobs_col.update_many(
                {"status": "running", "updated_at": {"$lt": stale_running_cutoff_iso}},
                {
                    "$set": {
                        "status": "queued",
                        "worker_id": None,
                        "current_question_id": None,
                        "updated_at": datetime.utcnow().isoformat(),
                    }
                },
            )
            if int(stale_reset.modified_count or 0) > 0:
                print(f"[Phase5] recovered stale running jobs={stale_reset.modified_count}")
        else:
            stale_failed = await phase5_jobs_col.update_many(
                {"status": "running", "updated_at": {"$lt": stale_running_cutoff_iso}},
                {
                    "$set": {
                        "status": "failed",
                        "worker_id": None,
                        "current_question_id": None,
                        "error": "stale_running_job_recovered_as_failed",
                        "updated_at": datetime.utcnow().isoformat(),
                    }
                },
            )
            if int(stale_failed.modified_count or 0) > 0:
                print(f"[Phase5] marked stale running jobs as failed={stale_failed.modified_count}")

        stale_queued_failed = await phase5_jobs_col.update_many(
            {"status": "queued", "updated_at": {"$lt": stale_queued_cutoff_iso}},
            {
                "$set": {
                    "status": "failed",
                    "error": "stale_queued_job_recovered_as_failed",
                    "updated_at": datetime.utcnow().isoformat(),
                }
            },
        )
        if int(stale_queued_failed.modified_count or 0) > 0:
            print(f"[Phase5] marked stale queued jobs as failed={stale_queued_failed.modified_count}")
    except Exception:
        traceback.print_exc()

    print(
        f"[Phase5] startup workers={max(1, PHASE5_WORKER_CONCURRENCY)} "
        f"parallelism={PHASE5_JOB_PARALLELISM} timeout_gemini={PHASE5_QUESTION_TIMEOUT_GEMINI_SEC}s "
        f"timeout_openai={PHASE5_QUESTION_TIMEOUT_OPENAI_SEC}s "
        f"timeout_perplexity={PHASE5_QUESTION_TIMEOUT_PERPLEXITY_SEC}s"
    )
    print(
        f"[Phase5] providers openai_model={(os.getenv('OPENAI_MODEL_PHASE5') or 'unset').strip() or 'unset'} "
        f"perplexity_model={(os.getenv('PERPLEXITY_MODEL_PHASE5') or 'sonar-pro').strip() or 'sonar-pro'}"
    )
    app.state.phase5_worker_tasks = [
        asyncio.create_task(_phase5_worker_loop())
        for _ in range(max(1, PHASE5_WORKER_CONCURRENCY))
    ]


@app.on_event("shutdown")
async def _phase5_worker_shutdown():
    tasks = getattr(app.state, "phase5_worker_tasks", [])
    for t in tasks:
        t.cancel()
    for t in tasks:
        try:
            await t
        except asyncio.CancelledError:
            pass
        except Exception:
            pass

    redis_client = getattr(app.state, "phase5_redis", None)
    if redis_client is not None:
        try:
            await redis_client.close()
        except Exception:
            pass

    executor = getattr(app.state, "phase5_executor", None)
    if executor is not None:
        try:
            executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass


@app.post("/api/phase5/start-job", response_model=Phase5StartJobResponse)
async def api_phase5_start_job(req: Phase5StartJobRequest, current_user: dict = Depends(get_current_user_optional)):
    try:
        if phase5_jobs_col is None:
            raise HTTPException(status_code=503, detail="phase5 job storage unavailable")
        questions_dicts = [q.model_dump() for q in req.questions]
        if not questions_dicts:
            raise HTTPException(status_code=400, detail="questions cannot be empty")
        model_provider = "multi"

        request_hash = _phase5_request_hash(
            url=req.url,
            questions=questions_dicts,
            job_type="core",
            model=model_provider,
        )
        reusable = await _find_reusable_phase5_job(request_hash, "core")
        if reusable:
            return {
                "job_id": reusable["job_id"],
                "status": reusable.get("status", "queued"),
                "total": reusable.get("total", len(questions_dicts)),
            }

        job_id = uuid.uuid4().hex
        now_iso = datetime.utcnow().isoformat()
        job_doc = {
            "job_id": job_id,
            "request_hash": request_hash,
            "job_type": "core",
            "model": model_provider,
            "queue_priority": 0,
            "url": req.url,
            "user_id": current_user.get("id") if current_user else None,
            "user_email": current_user.get("email") if current_user else None,
            "questions": questions_dicts,
            "seed_results": {},
            "status": "queued",
            "total": len(questions_dicts),
            "processed": 0,
            "current_question_id": None,
            "results": {},
            "deep_competitors": [],
            "brand_summary": None,
            "error": None,
            "created_at": now_iso,
            "updated_at": now_iso,
        }
        await phase5_jobs_col.insert_one(job_doc)

        redis_client = getattr(app.state, "phase5_redis", None)
        if redis_client is not None:
            try:
                await redis_client.rpush(PHASE5_REDIS_QUEUE_KEY, job_id.encode("utf-8"))
            except Exception:
                print("[Phase5] Redis enqueue failed. Job remains queued in Mongo and will be picked by polling worker.")

        print(
            f"[Phase5] queued job_id={job_id} provider={model_provider} "
            f"type=core questions={len(questions_dicts)}"
        )
        await phase5_jobs_col.update_one(
            {"job_id": job_id},
            {
                "$set": {
                    "status": "running",
                    "worker_id": PHASE5_WORKER_ID,
                    "updated_at": datetime.utcnow().isoformat(),
                }
            },
        )
        print(f"[Phase5] immediate start job_id={job_id} provider={model_provider}")
        running_doc = {**job_doc, "status": "running", "worker_id": PHASE5_WORKER_ID}
        asyncio.create_task(_process_phase5_job(running_doc))

        await _log_ai_usage_event({
            "feature": "phase5_job_started",
            "endpoint": "/api/phase5/start-job",
            "url": req.url,
            "user_id": current_user.get("id") if current_user else None,
            "user_email": current_user.get("email") if current_user else None,
            "model_provider": model_provider,
            "ai_calls_estimate": 0,
            "details": {
                "job_id": job_id,
                "job_type": "core",
                "model_provider": model_provider,
                "providers": ["gemini", "perplexity", "chatgpt"],
                "question_count": len(questions_dicts),
            },
        })

        return {
            "job_id": job_id,
            "status": "queued",
            "total": len(questions_dicts),
        }
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/phase5/start-deep-job", response_model=Phase5StartJobResponse)
async def api_phase5_start_deep_job(req: Phase5StartJobRequest, current_user: dict = Depends(get_current_user_optional)):
    try:
        if phase5_jobs_col is None:
            raise HTTPException(status_code=503, detail="phase5 job storage unavailable")
        questions_dicts = [q.model_dump() for q in req.questions]
        if not questions_dicts:
            raise HTTPException(status_code=400, detail="questions cannot be empty")
        model_provider = str(req.model or "perplexity").strip().lower()
        if model_provider == "gemini":
            raise HTTPException(status_code=400, detail="Gemini is temporarily disabled for Phase 5")
        if model_provider not in {"openai", "perplexity"}:
            raise HTTPException(status_code=400, detail="unsupported model provider")

        request_hash = _phase5_request_hash(
            url=req.url,
            questions=questions_dicts,
            job_type="deep",
            model=model_provider,
            seed_results=req.seed_results or {},
        )
        reusable = await _find_reusable_phase5_job(request_hash, "deep")
        if reusable:
            return {
                "job_id": reusable["job_id"],
                "status": reusable.get("status", "queued"),
                "total": reusable.get("total", len(questions_dicts)),
            }

        job_id = uuid.uuid4().hex
        now_iso = datetime.utcnow().isoformat()
        job_doc = {
            "job_id": job_id,
            "request_hash": request_hash,
            "job_type": "deep",
            "model": model_provider,
            "queue_priority": 0 if model_provider in {"openai", "perplexity"} else 1,
            "url": req.url,
            "user_id": current_user.get("id") if current_user else None,
            "user_email": current_user.get("email") if current_user else None,
            "questions": questions_dicts,
            "seed_results": req.seed_results or {},
            "status": "queued",
            "total": len(questions_dicts),
            "processed": 0,
            "current_question_id": None,
            "results": {},
            "deep_competitors": [],
            "brand_summary": None,
            "error": None,
            "created_at": now_iso,
            "updated_at": now_iso,
        }
        await phase5_jobs_col.insert_one(job_doc)

        redis_client = getattr(app.state, "phase5_redis", None)
        if redis_client is not None:
            try:
                await redis_client.rpush(PHASE5_REDIS_QUEUE_KEY, job_id.encode("utf-8"))
            except Exception:
                print("[Phase5] Redis enqueue failed. Job remains queued in Mongo and will be picked by polling worker.")

        print(
            f"[Phase5] queued job_id={job_id} provider={model_provider} "
            f"type=deep questions={len(questions_dicts)}"
        )
        await phase5_jobs_col.update_one(
            {"job_id": job_id},
            {
                "$set": {
                    "status": "running",
                    "worker_id": PHASE5_WORKER_ID,
                    "updated_at": datetime.utcnow().isoformat(),
                }
            },
        )
        print(f"[Phase5] immediate start job_id={job_id} provider={model_provider}")
        running_doc = {**job_doc, "status": "running", "worker_id": PHASE5_WORKER_ID}
        asyncio.create_task(_process_phase5_job(running_doc))

        await _log_ai_usage_event({
            "feature": "phase5_deep_job_started",
            "endpoint": "/api/phase5/start-deep-job",
            "url": req.url,
            "user_id": current_user.get("id") if current_user else None,
            "user_email": current_user.get("email") if current_user else None,
            "model_provider": model_provider,
            "ai_calls_estimate": 0,
            "details": {
                "job_id": job_id,
                "job_type": "deep",
                "model_provider": model_provider,
                "question_count": len(questions_dicts),
            },
        })

        return {
            "job_id": job_id,
            "status": "queued",
            "total": len(questions_dicts),
        }
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/phase5/job-status/{job_id}", response_model=Phase5JobStatusResponse)
async def api_phase5_job_status(job_id: str):
    if phase5_jobs_col is None:
        raise HTTPException(status_code=503, detail="phase5 job storage unavailable")
    job = await phase5_jobs_col.find_one({"job_id": job_id}, {"_id": 0})
    if not job:
        raise HTTPException(status_code=404, detail="job not found")

    return {
        "job_id": job["job_id"],
        "status": job["status"],
        "total": job["total"],
        "processed": job["processed"],
        "current_question_id": job.get("current_question_id"),
        "results": job["results"],
        "deep_competitors": job.get("deep_competitors", []),
        "brand_summary": job.get("brand_summary"),
        "error": job.get("error"),
    }


@app.get("/api/phase5/job-stream/{job_id}")
async def api_phase5_job_stream(job_id: str, request: Request):
    """Stream Phase 5 job progress via Server-Sent Events."""
    if phase5_jobs_col is None:
        raise HTTPException(status_code=503, detail="phase5 job storage unavailable")

    async def _event_generator():
        last_processed = -1
        last_status = ""
        last_deep = None
        last_brand = None
        idle_ticks = 0
        heartbeat_ticks = 0

        while True:
            if await request.is_disconnected():
                return

            job = await phase5_jobs_col.find_one(
                {"job_id": job_id},
                {
                    "_id": 0,
                    "job_id": 1,
                    "status": 1,
                    "processed": 1,
                    "total": 1,
                    "current_question_id": 1,
                    "results": 1,
                    "deep_competitors": 1,
                    "brand_summary": 1,
                    "provider_scores": 1,
                    "overall_score": 1,
                    "error": 1,
                },
            )

            if not job:
                yield 'event: error\ndata: {"error":"job not found"}\n\n'
                return

            processed = int(job.get("processed", 0) or 0)
            status = str(job.get("status", "") or "")

            # Also emit when deep_competitors or brand_summary change so clients
            # reliably receive finalization results even if processed/status
            # didn't change.
            deep_serial = json.dumps(job.get("deep_competitors", []), default=str, sort_keys=True)
            brand_now = str(job.get("brand_summary") or "")
            if processed != last_processed or status != last_status or deep_serial != last_deep or brand_now != last_brand:
                payload = {
                    "job_id": job_id,
                    "status": status,
                    "processed": processed,
                    "total": int(job.get("total", 0) or 0),
                    "current_question_id": job.get("current_question_id"),
                    "results": job.get("results", {}),
                    "provider_scores": job.get("provider_scores", {}),
                    "overall_score": job.get("overall_score"),
                    "brand_summary": job.get("brand_summary"),
                    "deep_competitors": job.get("deep_competitors", []),
                    "error": job.get("error"),
                }
                yield f"data: {json.dumps(payload, default=str)}\n\n"
                last_processed = processed
                last_status = status
                last_deep = deep_serial
                last_brand = brand_now
                idle_ticks = 0
                heartbeat_ticks = 0
            else:
                idle_ticks += 1
                heartbeat_ticks += 1
                if heartbeat_ticks >= 30:
                    # SSE comment heartbeat keeps proxies/load balancers from closing the stream.
                    yield ": keep-alive\n\n"
                    heartbeat_ticks = 0

            if status in PHASE5_TERMINAL_STATUSES:
                return

            if idle_ticks >= 600:
                return

            await asyncio.sleep(0.5)

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/phase5/stop-job/{job_id}")
async def api_phase5_stop_job(job_id: str):
    if phase5_jobs_col is None:
        raise HTTPException(status_code=503, detail="phase5 job storage unavailable")

    job = await phase5_jobs_col.find_one({"job_id": job_id}, {"_id": 0, "status": 1})
    if not job:
        raise HTTPException(status_code=404, detail="job not found")

    if job.get("status") in {"completed", "failed", "cancelled"}:
        return {"job_id": job_id, "status": job.get("status")}

    await phase5_jobs_col.update_one(
        {"job_id": job_id},
        {
            "$set": {
                "status": "cancelled",
                "current_question_id": None,
                "updated_at": datetime.utcnow().isoformat(),
            }
        },
    )
    return {"job_id": job_id, "status": "cancelled"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
