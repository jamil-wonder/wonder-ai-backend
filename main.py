import asyncio
import sys

# Windows asyncio workaround for Playwright
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from bson import ObjectId
from models import ScrapeRequest, ScrapeResult, AiInsightsRequest, AiInsightsResult, WishlistRequest, TrackUrlRequest
from scraper import scrape_website
from ai_agent import get_ai_insights
from phase2_models import CompareRequest, CompareResult
from competitor_engine import run_competitor_analysis
from phase3_models import ContentAnalysisRequest, ContentAnalysisResponse
from content_agent import analyze_url_content
from phase5_models import (
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
from phase5_agent import (
    generate_brand_questions,
    rank_brand_in_ai,
    analyze_single_question,
    generate_brand_perception_summary,
    generate_deep_competitor_scores,
    Phase5RateLimitError,
)
from models import UserCreate, UserResponse, Token, LoginRequest
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests
from pydantic import BaseModel
import secrets
import bcrypt
from jose import JWTError, jwt
from datetime import datetime, timedelta

import traceback
import uvicorn
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import ReturnDocument
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

# Phase 5 worker settings (intentionally fixed for reliability)
PHASE5_WORKER_CONCURRENCY = 1
PHASE5_WORKER_POLL_INTERVAL = 0.75
PHASE5_JOB_PARALLELISM = 1
PHASE5_MODEL_MAX_THREADS = 4
PHASE5_WORKER_ID = f"{os.getenv('HOSTNAME', 'local')}-{uuid.uuid4().hex[:8]}"
PHASE5_REDIS_URL = os.getenv("REDIS_URL", "").strip()
PHASE5_REDIS_QUEUE_KEY = os.getenv("PHASE5_REDIS_QUEUE_KEY", "phase5:jobs:queue")

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
try:
    mongo_client = AsyncIOMotorClient(MONGO_URL, serverSelectionTimeoutMS=5000)
    db = mongo_client.get_database("wonderai")
    wishlist_col = db.get_collection("wishlist")
    urls_col = db.get_collection("urls")
    users_col = db.get_collection("users")
    phase5_jobs_col = db.get_collection("phase5_jobs")
    ai_usage_col = db.get_collection("ai_usage_events")
except Exception as e:
    print(f"[API] Error connecting to MongoDB: {e}")

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
        model_name = (event or {}).get("model_name")
        inferred_family = "gemini" if isinstance(model_name, str) and model_name.lower().startswith("gemini") else "unknown"
        payload = {
            "timestamp": now,
            "timestamp_iso": now.isoformat(),
            "provider": "google",
            "model_family": inferred_family,
            **(event or {}),
        }
        await ai_usage_col.insert_one(payload)
    except Exception as e:
        print(f"[AI Usage] log failed: {e}")

# --- Auth Routes ---

class GoogleAuthRequest(BaseModel):
    credential: str

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
        insight = await asyncio.wait_for(get_ai_insights(request.businessName, request.url), timeout=90)
        model_name = insight.get("modelName") if isinstance(insight, dict) else None
        await _log_ai_usage_event({
            "feature": "phase1_ai_insights",
            "endpoint": "/api/ai-insights",
            "url": request.url,
            "user_id": current_user.get("id") if current_user else None,
            "user_email": current_user.get("email") if current_user else None,
            "model_name": model_name,
            "model_family": "gemini" if isinstance(model_name, str) and model_name.lower().startswith("gemini") else "unknown",
            "ai_calls_estimate": 1,
            "details": {
                "isKnown": bool(insight.get("isKnown", False)) if isinstance(insight, dict) else None,
                "platform_count": len(insight.get("platforms", [])) if isinstance(insight, dict) and isinstance(insight.get("platforms"), list) else 0,
            },
        })
        elapsed = (datetime.utcnow() - started_at).total_seconds()
        print(f"[API] /api/ai-insights completed in {elapsed:.1f}s: {request.url}")
        return AiInsightsResult(success=True, insights=[insight])
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
        configured_model = (os.getenv("GEMINI_MODEL_PRIMARY") or os.getenv("GEMINI_MODEL") or "").strip() or None
        await _log_ai_usage_event({
            "feature": "phase5_generate_questions",
            "endpoint": "/api/phase5/generate-questions",
            "url": req.url,
            "user_id": current_user.get("id") if current_user else None,
            "user_email": current_user.get("email") if current_user else None,
            "model_name": configured_model,
            "model_family": "gemini" if isinstance(configured_model, str) and configured_model.lower().startswith("gemini") else "unknown",
            "ai_calls_estimate": 1,
            "details": {
                "questions_count": len(questions or []),
            },
        })
        return {"questions": questions}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/phase5/analyze", response_model=Phase5AnalyzeResponse)
async def api_phase5_analyze(req: Phase5AnalyzeRequest, current_user: dict = Depends(get_current_user_optional)):
    try:
        # q.model_dump() turns the pydantic model into a dictionary
        questions_dicts = [q.model_dump() for q in req.questions]
        results = await rank_brand_in_ai(req.url, questions_dicts)
        configured_model = (os.getenv("GEMINI_MODEL_PRIMARY") or os.getenv("GEMINI_MODEL") or "").strip() or None
        await _log_ai_usage_event({
            "feature": "phase5_analyze_direct",
            "endpoint": "/api/phase5/analyze",
            "url": req.url,
            "user_id": current_user.get("id") if current_user else None,
            "user_email": current_user.get("email") if current_user else None,
            "model_name": configured_model,
            "model_family": "gemini" if isinstance(configured_model, str) and configured_model.lower().startswith("gemini") else "unknown",
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
        result = await analyze_single_question(req.url, req.question.model_dump())
        configured_model = (os.getenv("GEMINI_MODEL_PRIMARY") or os.getenv("GEMINI_MODEL") or "").strip() or None
        await _log_ai_usage_event({
            "feature": "phase5_analyze_single",
            "endpoint": "/api/phase5/analyze-single",
            "url": req.url,
            "user_id": current_user.get("id") if current_user else None,
            "user_email": current_user.get("email") if current_user else None,
            "model_name": configured_model,
            "model_family": "gemini" if isinstance(configured_model, str) and configured_model.lower().startswith("gemini") else "unknown",
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
    include_competitors = job_type == "deep"
    print(f"[Phase5] worker start job_id={job_id} type={job_type}")
    try:
        async def _is_cancelled() -> bool:
            latest = await phase5_jobs_col.find_one(
                {"job_id": job_id},
                {"status": 1, "_id": 0},
            )
            return not latest or latest.get("status") == "cancelled"

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

        async def _worker_run():
            while True:
                if await _is_cancelled():
                    break

                try:
                    q = queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

                qid = q["id"]
                await phase5_jobs_col.update_one(
                    {"job_id": job_id, "status": {"$ne": "cancelled"}},
                    {
                        "$set": {
                            "current_question_id": qid,
                            "updated_at": datetime.utcnow().isoformat(),
                        }
                    },
                )

                result = await analyze_single_question(
                    job_doc["url"],
                    q,
                    include_competitors=include_competitors,
                )

                async with lock:
                    accumulated_results[qid] = result

                await phase5_jobs_col.update_one(
                    {"job_id": job_id, "status": {"$ne": "cancelled"}},
                    {
                        "$set": {
                            f"results.{qid}": result,
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

        # Auto finalization: generate quick competitor score + brand summary
        # so users do not need a second click/flow.
        await phase5_jobs_col.update_one(
            {"job_id": job_id},
            {
                "$set": {
                    "status": "finalizing",
                    "current_question_id": None,
                    "updated_at": datetime.utcnow().isoformat(),
                }
            },
        )
        print(f"[Phase5] job_id={job_id} entering finalizing")

        deep_competitors = []
        try:
            deep_competitors = await asyncio.wait_for(
                generate_deep_competitor_scores(
                    url=job_doc["url"],
                    questions=questions,
                    seed_results=accumulated_results,
                ),
                timeout=15,
            )
        except Exception as e:
            print(f"[Phase5] deep competitor finalize fallback: {e}")
            deep_competitors = []

        brand_summary = None
        try:
            brand_summary = await asyncio.wait_for(
                generate_brand_perception_summary(
                    url=job_doc["url"],
                    questions=questions,
                    results=accumulated_results,
                ),
                timeout=10,
            )
        except Exception as e:
            print(f"[Phase5] brand summary finalize fallback: {e}")
            domain = str(job_doc.get("url", "")).replace("https://", "").replace("http://", "").split("/")[0]
            brand_summary = f"{domain} appears to have a clear business identity and a visible presence across customer search intent."

        await phase5_jobs_col.update_one(
            {"job_id": job_id},
            {
                "$set": {
                    "status": "completed",
                    "current_question_id": None,
                    "deep_competitors": deep_competitors,
                    "brand_summary": brand_summary,
                    "updated_at": datetime.utcnow().isoformat(),
                }
            },
        )
        print(f"[Phase5] worker completed job_id={job_id} competitors={len(deep_competitors)}")
        await _log_ai_usage_event({
            "feature": "phase5_job_completed",
            "endpoint": "/api/phase5/start-job",
            "url": job_doc.get("url"),
            "user_id": job_doc.get("user_id"),
            "user_email": job_doc.get("user_email"),
            "ai_calls_estimate": int(job_doc.get("processed", 0) or 0) + 2,
            "details": {
                "job_id": job_id,
                "job_type": job_type,
                "processed": job_doc.get("processed", 0),
                "total": len(questions),
                "competitors_count": len(deep_competitors or []),
            },
        })
    except Phase5RateLimitError as e:
        await _log_ai_usage_event({
            "feature": "phase5_job_failed_rate_limit",
            "endpoint": "/api/phase5/start-job",
            "url": job_doc.get("url"),
            "user_id": job_doc.get("user_id"),
            "user_email": job_doc.get("user_email"),
            "ai_calls_estimate": int(job_doc.get("processed", 0) or 0),
            "details": {
                "job_id": job_id,
                "job_type": job_type,
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
        await _log_ai_usage_event({
            "feature": "phase5_job_failed",
            "endpoint": "/api/phase5/start-job",
            "url": job_doc.get("url"),
            "user_id": job_doc.get("user_id"),
            "user_email": job_doc.get("user_email"),
            "ai_calls_estimate": int(job_doc.get("processed", 0) or 0),
            "details": {
                "job_id": job_id,
                "job_type": job_type,
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
            else:
                claimed = await phase5_jobs_col.find_one_and_update(
                    {"status": "queued"},
                    {
                        "$set": {
                            "status": "running",
                            "worker_id": PHASE5_WORKER_ID,
                            "updated_at": datetime.utcnow().isoformat(),
                        }
                    },
                    sort=[("created_at", 1)],
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

    await phase5_jobs_col.create_index("job_id", unique=True)
    await phase5_jobs_col.create_index("status")
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

        job_id = uuid.uuid4().hex
        await phase5_jobs_col.insert_one({
            "job_id": job_id,
            "job_type": "core",
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
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
        })

        redis_client = getattr(app.state, "phase5_redis", None)
        if redis_client is not None:
            try:
                await redis_client.rpush(PHASE5_REDIS_QUEUE_KEY, job_id.encode("utf-8"))
            except Exception:
                print("[Phase5] Redis enqueue failed. Job remains queued in Mongo and will be picked by polling worker.")

        await _log_ai_usage_event({
            "feature": "phase5_job_started",
            "endpoint": "/api/phase5/start-job",
            "url": req.url,
            "user_id": current_user.get("id") if current_user else None,
            "user_email": current_user.get("email") if current_user else None,
            "ai_calls_estimate": 0,
            "details": {"job_id": job_id, "job_type": "core", "question_count": len(questions_dicts)},
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

        job_id = uuid.uuid4().hex
        await phase5_jobs_col.insert_one({
            "job_id": job_id,
            "job_type": "deep",
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
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
        })

        redis_client = getattr(app.state, "phase5_redis", None)
        if redis_client is not None:
            try:
                await redis_client.rpush(PHASE5_REDIS_QUEUE_KEY, job_id.encode("utf-8"))
            except Exception:
                print("[Phase5] Redis enqueue failed. Job remains queued in Mongo and will be picked by polling worker.")

        await _log_ai_usage_event({
            "feature": "phase5_deep_job_started",
            "endpoint": "/api/phase5/start-deep-job",
            "url": req.url,
            "user_id": current_user.get("id") if current_user else None,
            "user_email": current_user.get("email") if current_user else None,
            "ai_calls_estimate": 0,
            "details": {"job_id": job_id, "job_type": "deep", "question_count": len(questions_dicts)},
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
