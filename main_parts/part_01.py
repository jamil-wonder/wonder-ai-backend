# Generated from the former backend/main.py lines 1-400.
import asyncio
import os
import json
import uuid
import re
import time
import math
import hashlib
import statistics
import traceback

from datetime import datetime, timedelta, timezone
from typing import List, Optional, Dict, Any, Union
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, Field

import pandas as pd

from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import UpdateOne
from bson.objectid import ObjectId

from fastapi import FastAPI, HTTPException, Request, Depends, status, Query, BackgroundTasks, Header
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from passlib.context import CryptContext
from jose import JWTError, jwt

from part_02 import *
from part_03 import *
from part_04 import *
from part_05 import *
from part_06 import *
from part_07 import *

SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production-please")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    if users_col is None:
        raise HTTPException(status_code=503, detail="Database unavailable")

    try:
        user = await users_col.find_one({"_id": ObjectId(user_id)})
    except Exception:
        user = await users_col.find_one({"id": user_id})

    if user is None:
        raise credentials_exception

    return {
        "id": str(user.get("_id", user.get("id"))),
        "email": user.get("email"),
        "name": user.get("name", user.get("email", "").split("@")[0]),
        "role": user.get("role", "user")
    }

async def get_optional_current_user(token: Optional[str] = Depends(OAuth2PasswordBearer(tokenUrl="token", auto_error=False))):
    if not token:
        return None
    try:
        return await get_current_user(token)
    except HTTPException:
        return None

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")

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
        "https://wonder-new-dashboard.vercel.app",
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
google_integrations_col = None
analytics_snapshots_col = None
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
    google_integrations_col = db.get_collection("google_integrations")
    analytics_snapshots_col = db.get_collection("analytics_snapshots")
    phase5_jobs_col = db.get_collection("phase5_jobs")
    ai_usage_col = db.get_collection("ai_usage_logs")
    user_history_meta_col = db.get_collection("user_history_meta")
    public_rate_limits_col = db.get_collection("public_rate_limits")
    print("Connected to MongoDB successfully.")
except Exception as e:
    print(f"MongoDB Connection Warning: {e}")
