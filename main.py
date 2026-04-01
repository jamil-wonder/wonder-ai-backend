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
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import ReturnDocument
from redis import asyncio as aioredis
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Wonder AI Backend")

# Phase 5 persistent worker settings
PHASE5_WORKER_CONCURRENCY = int(os.getenv("PHASE5_WORKER_CONCURRENCY", "2"))
PHASE5_WORKER_POLL_INTERVAL = float(os.getenv("PHASE5_WORKER_POLL_INTERVAL", "0.5"))
PHASE5_JOB_PARALLELISM = int(os.getenv("PHASE5_JOB_PARALLELISM", "5"))
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
MONGO_URL = os.getenv("MONGODB_URL", "mongodb+srv://jamil_db_user:qBfb3HtWmwvEEEkb@wonderai-db.qozs3tl.mongodb.net/?appName=wonderai-db")
phase5_jobs_col = None
try:
    mongo_client = AsyncIOMotorClient(MONGO_URL, serverSelectionTimeoutMS=5000)
    db = mongo_client.get_database("wonderai")
    wishlist_col = db.get_collection("wishlist")
    urls_col = db.get_collection("urls")
    users_col = db.get_collection("users")
    phase5_jobs_col = db.get_collection("phase5_jobs")
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
        result = await scrape_website(request.url)
        return result
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
async def api_ai_insights(request: AiInsightsRequest):
    try:
        insight = await get_ai_insights(request.businessName, request.url)
        return AiInsightsResult(success=True, insights=[insight])
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

# ==============================================================================
# PHASE 5 ROUTES
# ==============================================================================
@app.post("/api/phase5/generate-questions", response_model=Phase5QuestionsResponse)
async def api_phase5_generate_questions(req: Phase5QuestionsRequest):
    try:
        questions = await generate_brand_questions(req.url)
        return {"questions": questions}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/phase5/analyze", response_model=Phase5AnalyzeResponse)
async def api_phase5_analyze(req: Phase5AnalyzeRequest):
    try:
        # q.model_dump() turns the pydantic model into a dictionary
        questions_dicts = [q.model_dump() for q in req.questions]
        results = await rank_brand_in_ai(req.url, questions_dicts)
        return {"results": results}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/phase5/analyze-single", response_model=Phase5AnalyzeSingleResponse)
async def api_phase5_analyze_single(req: Phase5AnalyzeSingleRequest):
    try:
        result = await analyze_single_question(req.url, req.question.model_dump())
        return result
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


async def _process_phase5_job(job_doc: dict):
    job_id = job_doc["job_id"]
    try:
        async def _is_cancelled() -> bool:
            latest = await phase5_jobs_col.find_one(
                {"job_id": job_id},
                {"status": 1, "_id": 0},
            )
            return not latest or latest.get("status") == "cancelled"

        questions = list(job_doc.get("questions", []))
        accumulated_results = {}
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

                result = await analyze_single_question(job_doc["url"], q)

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

        brand_summary = await generate_brand_perception_summary(
            url=job_doc["url"],
            questions=questions,
            results=accumulated_results,
        )

        await phase5_jobs_col.update_one(
            {"job_id": job_id},
            {
                "$set": {
                    "status": "completed",
                    "current_question_id": None,
                    "brand_summary": brand_summary,
                    "updated_at": datetime.utcnow().isoformat(),
                }
            },
        )
    except Exception as e:
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


@app.post("/api/phase5/start-job", response_model=Phase5StartJobResponse)
async def api_phase5_start_job(req: Phase5StartJobRequest):
    try:
        if phase5_jobs_col is None:
            raise HTTPException(status_code=503, detail="phase5 job storage unavailable")
        questions_dicts = [q.model_dump() for q in req.questions]
        if not questions_dicts:
            raise HTTPException(status_code=400, detail="questions cannot be empty")

        job_id = uuid.uuid4().hex
        await phase5_jobs_col.insert_one({
            "job_id": job_id,
            "url": req.url,
            "questions": questions_dicts,
            "status": "queued",
            "total": len(questions_dicts),
            "processed": 0,
            "current_question_id": None,
            "results": {},
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
