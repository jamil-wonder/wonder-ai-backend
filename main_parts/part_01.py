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
    PublicScanUnlockRequest,
    ContactFormRequest,
    OnboardingSuggestionsRequest,
    OnboardingSuggestionsResult,
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
from scraping.scraper import scrape_website, get_grade
from agents.ai_agent import (
    get_ai_insights_multi,
    get_ai_insights_openai,
    get_onboarding_suggestions,
    infer_business_category,
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
from agents.phase5.config import PHASE5_ENABLE_GEMINI, PHASE5_QUESTION_GEN_TIMEOUT_SEC
from agents.phase5_agent import (
    generate_brand_questions,
    generate_preview_questions,
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
import math


def _round_half_up(value: float) -> int:
    """Python's builtin round() rounds half-to-even (round(64.5) == 64),
    JavaScript's Math.round always rounds half up (Math.round(64.5) == 65).
    Every score shown to a user is rounded on ONE side or the other of the
    API boundary — the same 64.5 could render as 64 in a Python-built email
    and 65 on a Math.round()-based frontend page for the same real number,
    which is exactly what happened (Reports preview showed 65, the public
    share link showed 64). Use this instead of round() anywhere a score
    might be shown somewhere the frontend also renders it, so the two
    always agree."""
    return math.floor(value + 0.5)
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
# How many businesses the Sunday weekly re-run processes at once. Was a
# strict one-at-a-time loop — fine at today's scale (a couple of locked
# businesses) but would take many hours to clear a real user base doing it
# fully sequentially. Bounded rather than "all at once" so this doesn't spike
# past the AI providers' rate limits the moment there are dozens of
# businesses locked in the same week.
SUNDAY_SCHEDULER_CONCURRENCY = max(1, min(10, int(os.getenv("SUNDAY_SCHEDULER_CONCURRENCY", "3"))))
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

# "web" (default) serves the FastAPI app only; "worker" runs the background
# schedulers/job loops with no HTTP server. Same image, same env, split by
# this one var — see backend/entrypoint.sh and backend/worker.py. Moving
# background jobs off the web process is what stops a web-tier redeploy
# from killing an in-flight job (docs/infra-diagnosis.html).
ROLE = os.getenv("ROLE", "web").strip().lower()

default_allowed_origins = [
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
allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "")
env_allowed_origins = [
    origin.strip()
    for origin in allowed_origins_env.split(",")
    if origin.strip()
]
allowed_origins = list(dict.fromkeys([*default_allowed_origins, *env_allowed_origins]))

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
email_verifications_col = None
public_scan_leads_col = None
manual_run_requests_col = None
contact_submissions_col = None
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
    ai_usage_col = db.get_collection("ai_usage_events")
    user_history_meta_col = db.get_collection("user_history_meta")
    public_rate_limits_col = db.get_collection("public_rate_limits")
    email_verifications_col = db.get_collection("email_verifications")
    public_scan_leads_col = db.get_collection("public_scan_leads")
    manual_run_requests_col = db.get_collection("manual_run_requests")
    contact_submissions_col = db.get_collection("contact_submissions")
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
            # Real "inactive" signal for the mid-week reminder email — fire-
            # and-forget so it never adds latency to the request that
            # triggered it. Cheap: one field, one write, only on actual
            # authenticated calls.
            # ensure_future (not create_task) because under Gunicorn's
            # UvicornWorker, Motor's update_one() here returns a Future
            # rather than a plain coroutine — create_task rejects that with
            # "a coroutine was expected", which this whole function's bare
            # except then silently swallowed, making every valid token look
            # invalid (every authenticated request 401'd). ensure_future
            # accepts both coroutines and Future-likes.
            asyncio.ensure_future(
                users_col.update_one(
                    {"_id": user["_id"]},
                    {"$set": {"last_active_at": datetime.utcnow().isoformat() + "Z"}},
                )
            )
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
        # Phase5 jobs can run in "multi" mode (job_doc.get("model") == "multi"
        # in part_09.py) — every provider ran, not one specific model. That's
        # not missing data, it's a real third category alongside a single
        # provider, so it gets its own name instead of falling through to the
        # "unknown" bucket below where every genuinely-untracked event also
        # lands.
        if not model_name and model_provider == "multi":
            model_name = "multi"

        model_family = payload.get("model_family")
        lowered_model = str(model_name or "").lower()
        if not model_family:
            if model_provider == "multi":
                model_family = "multi"
            elif model_provider == "gemini":
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
            if model_family == "multi":
                provider = "multi"
            elif model_family == "gemini":
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

# --- Email (SMTP) ---
# All optional: if SMTP_HOST/SMTP_USER/SMTP_PASSWORD aren't set, send_email()
# just logs and no-ops rather than raising, so nothing that isn't ready to
# configure email yet is broken by this.
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = (os.getenv("SMTP_USER") or "").strip() or None
# Google displays App Passwords with spaces for human readability
# ("yczh jhws ueuo qaol") but the real 16-char credential has none — that
# literal string, spaces included, is a common paste-in mistake that fails
# SMTP auth silently (send_email() just logs and no-ops). Stripping all
# whitespace here makes both the spaced and unspaced forms work.
SMTP_PASSWORD = (os.getenv("SMTP_PASSWORD") or "").replace(" ", "").strip() or None
SMTP_FROM_NAME = os.getenv("SMTP_FROM_NAME", "Wonder AI")
# Where the marketing site's "contact us" chat widget notifies a human —
# falls back to the SMTP account itself so this works without a separate
# env var in the common case of one inbox handling both sending and support.
CONTACT_NOTIFY_EMAIL = os.getenv("CONTACT_NOTIFY_EMAIL") or SMTP_USER
EMAIL_OTP_LENGTH = 6
EMAIL_OTP_TTL_MINUTES = 15
EMAIL_OTP_MAX_ATTEMPTS = 5
EMAIL_VERIFICATION_RESEND_COOLDOWN_SECONDS = 120


async def send_email(to_email: str, subject: str, html_body: str, text_body: str) -> bool:
    if not SMTP_USER or not SMTP_PASSWORD:
        print(f"[Email] SMTP not configured — skipping send to {to_email} ({subject})")
        return False
    try:
        import aiosmtplib
        from email.message import EmailMessage

        message = EmailMessage()
        message["From"] = f"{SMTP_FROM_NAME} <{SMTP_USER}>"
        message["To"] = to_email
        message["Subject"] = subject
        message.set_content(text_body)
        message.add_alternative(html_body, subtype="html")

        await aiosmtplib.send(
            message,
            hostname=SMTP_HOST,
            port=SMTP_PORT,
            start_tls=True,
            username=SMTP_USER,
            password=SMTP_PASSWORD,
        )
        return True
    except Exception as e:
        print(f"[Email] send failed to {to_email}: {type(e).__name__}: {e}")
        return False


def _build_verification_email_otp(name: str, code: str) -> tuple[str, str]:
    display_name = name or "there"
    text_body = (
        f"Hi {display_name},\n\n"
        f"Your Wonderscore verification code is: {code}\n\n"
        f"This code expires in {EMAIL_OTP_TTL_MINUTES} minutes. "
        f"If you didn't create this account, you can ignore this email.\n"
    )
    html_body = f"""
    <div style="margin:0;padding:0;background:#faf8f3;">
      <div style="font-family:Arial,Helvetica,sans-serif;max-width:560px;margin:0 auto;padding:28px 16px;color:#23211b;">
        <div style="background:#ffffff;border:1px solid #ece3d1;border-radius:24px;overflow:hidden;box-shadow:0 18px 45px rgba(21,70,59,0.10);">
          <div style="padding:26px 28px 20px 28px;border-bottom:1px solid #f0e8d8;background:#fdfcf8;">
            <table role="presentation" cellspacing="0" cellpadding="0" style="border-collapse:collapse;">
              <tr>
                <td width="42" style="vertical-align:middle;">
                  <div style="width:42px;height:42px;line-height:42px;text-align:center;border-radius:14px;background:#15463b;color:#ffffff;font-size:24px;font-weight:700;">&#10022;</div>
                </td>
                <td style="vertical-align:middle;padding-left:12px;">
                  <div style="font-size:20px;font-weight:700;letter-spacing:-0.02em;color:#15463b;">Wonderscore</div>
                  <div style="font-size:12px;line-height:18px;color:#8a8273;">AI visibility dashboard</div>
                </td>
              </tr>
            </table>
          </div>

          <div style="padding:30px 28px 24px 28px;">
            <div style="display:inline-block;margin-bottom:14px;padding:5px 9px;border-radius:999px;background:#edf8f1;border:1px solid #ccebd8;color:#0f7a4d;font-size:10px;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;">
              Secure sign-in
            </div>
            <h1 style="margin:0 0 10px 0;font-size:28px;line-height:34px;font-weight:700;color:#15463b;letter-spacing:-0.03em;">Enter your verification code</h1>
            <p style="margin:0 0 24px 0;font-size:15px;line-height:24px;color:#6f6757;">
              Hi {display_name}, use this code to continue to your Wonderscore dashboard.
            </p>

            <div style="margin:0 0 22px 0;padding:20px 12px;background:#f6f3ec;border:1px solid #ece3d1;border-radius:18px;text-align:center;">
              <div style="margin-bottom:10px;font-size:10px;font-weight:700;letter-spacing:0.16em;text-transform:uppercase;color:#9b927f;">Verification code</div>
              <table role="presentation" cellspacing="0" cellpadding="0" align="center" style="border-collapse:collapse;margin:0 auto;">
                <tr>
                  <td style="padding:10px 16px;background:#ffffff;border:1px solid #e5dac7;border-radius:14px;">
                    <span style="font-size:28px;line-height:34px;font-weight:700;letter-spacing:0.28em;color:#15463b;font-family:Arial,Helvetica,sans-serif;">{code}</span>
                  </td>
                </tr>
              </table>
            </div>

            <table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="border-collapse:collapse;margin-top:8px;">
              <tr>
                <td style="padding:14px 16px;background:#fdfcf8;border:1px solid #ece3d1;border-radius:14px;font-size:13px;line-height:20px;color:#6f6757;">
                  This code expires in <strong style="color:#23211b;">{EMAIL_OTP_TTL_MINUTES} minutes</strong>. If you did not request this email, you can safely ignore it.
                </td>
              </tr>
            </table>
          </div>
        </div>

        <p style="margin:18px 0 0 0;text-align:center;font-size:11px;line-height:18px;color:#9b927f;">
          Sent by Wonderscore for account security.
        </p>
      </div>
    </div>
    """
    return html_body, text_body


async def send_verification_email(to_email: str, name: str, user_id: str, purpose: str = "verify") -> bool:
    if email_verifications_col is None:
        return False
    code = "".join(secrets.choice("0123456789") for _ in range(EMAIL_OTP_LENGTH))
    code_hash = hashlib.sha256(code.encode("utf-8")).hexdigest()
    now = datetime.utcnow()
    # Invalidate any earlier outstanding codes for this user so only the
    # most recent one sent is ever valid — avoids ambiguity about which
    # code should work if someone requests a resend.
    await email_verifications_col.update_many(
        {"user_id": user_id, "used_at": None},
        {"$set": {"used_at": now}},
    )
    await email_verifications_col.insert_one({
        "user_id": user_id,
        "email": to_email,
        "code_hash": code_hash,
        "purpose": purpose,
        "attempts": 0,
        "created_at": now,
        "expires_at": now + timedelta(minutes=EMAIL_OTP_TTL_MINUTES),
        "used_at": None,
    })
    subject = "Your Wonder AI sign-in code" if purpose == "login" else "Your Wonder AI verification code"
    html_body, text_body = _build_verification_email_otp(name, code)
    return await send_email(to_email, subject, html_body, text_body)


def _build_welcome_email(name: str, dashboard_url: str) -> tuple[str, str]:
    display_name = name or "there"
    steps = [
        ("Run your first scan", "Head to the Analyser tab and enter your website — Wonderscore crawls it and scores your AI visibility in a couple of minutes."),
        ("Save your business profile", "Save the business, add competitors and the pages you want tracked, so every future scan builds on it."),
        ("Let it run itself", "Every Sunday at 4am we automatically re-scan your saved businesses and email you a fresh summary — no need to remember to check back."),
    ]
    text_steps = "\n".join(f"{i}. {title} — {desc}" for i, (title, desc) in enumerate(steps, start=1))
    text_body = (
        f"Hi {display_name},\n\n"
        f"Welcome to Wonderscore — glad to have you.\n\n"
        f"Here's how to get started:\n{text_steps}\n\n"
        f"Go to your dashboard: {dashboard_url}\n\n"
        f"You can turn scan-complete emails on or off any time from Settings.\n"
    )

    step_rows = "".join(
        f"""
        <tr>
          <td width="30" valign="top" style="padding:2px 12px 18px 0;">
            <div style="width:22px;height:22px;line-height:22px;text-align:center;border-radius:999px;background:#15463b;color:#ffffff;font-size:11px;font-weight:700;">{i}</div>
          </td>
          <td style="padding:0 0 18px 0;">
            <div style="font-size:14px;font-weight:700;color:#23211b;">{title}</div>
            <div style="margin-top:2px;font-size:13px;line-height:19px;color:#6f6757;">{desc}</div>
          </td>
        </tr>"""
        for i, (title, desc) in enumerate(steps, start=1)
    )

    html_body = f"""
    <div style="margin:0;padding:0;background:#faf8f3;">
      <div style="font-family:Arial,Helvetica,sans-serif;max-width:560px;margin:0 auto;padding:28px 16px;color:#23211b;">
        <div style="background:#ffffff;border:1px solid #ece3d1;border-radius:24px;overflow:hidden;box-shadow:0 18px 45px rgba(21,70,59,0.10);">
          <div style="padding:26px 28px 20px 28px;border-bottom:1px solid #f0e8d8;background:#fdfcf8;">
            <table role="presentation" cellspacing="0" cellpadding="0" style="border-collapse:collapse;">
              <tr>
                <td width="42" style="vertical-align:middle;">
                  <div style="width:42px;height:42px;line-height:42px;text-align:center;border-radius:14px;background:#15463b;color:#ffffff;font-size:24px;font-weight:700;">&#10022;</div>
                </td>
                <td style="vertical-align:middle;padding-left:12px;">
                  <div style="font-size:20px;font-weight:700;letter-spacing:-0.02em;color:#15463b;">Wonderscore</div>
                  <div style="font-size:12px;line-height:18px;color:#8a8273;">AI visibility dashboard</div>
                </td>
              </tr>
            </table>
          </div>

          <div style="padding:30px 28px 24px 28px;">
            <div style="display:inline-block;margin-bottom:14px;padding:5px 9px;border-radius:999px;background:#edf8f1;border:1px solid #ccebd8;color:#0f7a4d;font-size:10px;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;">
              Welcome
            </div>
            <h1 style="margin:0 0 10px 0;font-size:26px;line-height:32px;font-weight:700;color:#15463b;letter-spacing:-0.03em;">Welcome to Wonderscore, {display_name}</h1>
            <p style="margin:0 0 24px 0;font-size:14px;line-height:22px;color:#6f6757;">
              Wonderscore scores and tracks how visible your business is to AI search — ChatGPT, Claude, Perplexity, and Gemini — and tells you exactly what to fix.
            </p>

            <div style="margin:0 0 22px 0;">
              <div style="font-size:11px;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:#9b927f;margin-bottom:10px;">Here's how to get started</div>
              <table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="border-collapse:collapse;">
                {step_rows}
              </table>
            </div>

            <table role="presentation" cellspacing="0" cellpadding="0" style="border-collapse:collapse;">
              <tr>
                <td style="border-radius:12px;background:#15463b;">
                  <a href="{dashboard_url}" style="display:inline-block;padding:12px 22px;font-size:14px;font-weight:700;color:#ffffff;text-decoration:none;">Go to your dashboard</a>
                </td>
              </tr>
            </table>
          </div>
        </div>

        <p style="margin:18px 0 0 0;text-align:center;font-size:11px;line-height:18px;color:#9b927f;">
          You can turn scan-complete emails on or off any time from Settings.
        </p>
      </div>
    </div>
    """
    return html_body, text_body


async def send_welcome_email(to_email: str, name: str) -> bool:
    if not to_email:
        return False
    frontend_url = (os.getenv("FRONTEND_APP_URL") or "http://localhost:3000").rstrip("/")
    dashboard_url = f"{frontend_url}/overview"
    html_body, text_body = _build_welcome_email(name, dashboard_url)
    return await send_email(to_email, "Welcome to Wonderscore", html_body, text_body)


def _grade_pill_color(grade: str) -> tuple[str, str]:
    g = str(grade or "").upper()
    if g in ("A+", "A"):
        return "#0f7a4d", "#edf8f1"
    if g in ("B+", "B"):
        return "#9a6a12", "#faf1da"
    return "#b1442a", "#fbeee6"


def _score_badge_html(score: int) -> str:
    """A solid circular badge, not an SVG progress ring — Gmail (and some
    other clients) strip inline <svg> outright for security, which is
    exactly what silently turned the previous ring into bare, uncircled
    text. border-radius on a table cell is far less exciting visually but
    actually renders as a circle everywhere that matters."""
    clamped = max(0, min(100, int(score or 0)))
    size = 92
    return f"""<table role="presentation" cellspacing="0" cellpadding="0" style="border-collapse:collapse;">
      <tr>
        <td width="{size}" height="{size}" align="center" valign="middle" style="width:{size}px;height:{size}px;border-radius:{size // 2}px;background:#15463b;">
          <div style="font-family:Arial,Helvetica,sans-serif;font-size:30px;line-height:{size}px;font-weight:700;color:#ffffff;">{clamped}</div>
        </td>
      </tr>
    </table>"""


def _entity_signal_rows(scrape: dict) -> str:
    checks = [
        ("Business name", bool(str((scrape or {}).get("businessName") or "").strip())),
        ("Phone number", bool((scrape or {}).get("phones"))),
        ("Address", bool((scrape or {}).get("addresses"))),
        ("Opening hours", bool((scrape or {}).get("openingHours"))),
        ("Social links", bool((scrape or {}).get("socialLinks"))),
        ("Contact page", bool((scrape or {}).get("hasContactPath"))),
    ]
    rows = ""
    for label, found in checks:
        color = "#1e7d4f" if found else "#b1442a"
        status_text = "Found" if found else "Not found"
        rows += f"""
        <tr>
          <td style="padding:6px 0;font-size:13px;color:#3a352b;border-bottom:1px solid #f0ebe0;">{label}</td>
          <td style="padding:6px 0;font-size:13px;font-weight:700;color:{color};text-align:right;border-bottom:1px solid #f0ebe0;">{status_text}</td>
        </tr>"""
    return rows


def _technical_summary_line(scrape: dict) -> str:
    checks = [
        ("HTTPS", bool((scrape or {}).get("hasSSL"))),
        ("Mobile-friendly", bool((scrape or {}).get("hasMobileMeta"))),
        ("Sitemap", bool((scrape or {}).get("sitemapFound"))),
        ("Robots.txt", bool((scrape or {}).get("robotsTxtFound"))),
        ("Canonical URL", bool((scrape or {}).get("canonicalUrl"))),
    ]
    passed = sum(1 for _, ok in checks if ok)
    failed = [label for label, ok in checks if not ok]
    if not failed:
        return f"{passed} of {len(checks)} technical checks passed — nothing missing here."
    return f"{passed} of {len(checks)} technical checks passed. Missing: {', '.join(failed)}."


def _compute_audit_areas(scrape: dict) -> list[dict]:
    """Python port of the frontend's 6 audit-area scores (analyser/page.tsx
    lines ~283-310). Used as a fallback when the caller doesn't already
    supply computed areas — namely the Sunday scheduler, which has no
    frontend to compute them. Deliberately reads raw signal fields
    (phones/schemas/socialLinks/etc.), NOT scrape['scores'], because that
    dict's category totals mean different things depending on where the
    scrape came from (the frontend overwrites them with these same
    transformed values before it's ever sent anywhere)."""
    scrape = scrape or {}
    scores = scrape.get("scores") or {}
    schemas_found = len(scrape.get("schemas") or [])
    socials_found = len((scrape.get("socialLinks") or {}) or {})
    has_ssl = scrape.get("hasSSL", True)
    has_mobile = scrape.get("hasMobileMeta", True)
    has_sitemap = scrape.get("sitemapFound", True)
    has_robots = scrape.get("robotsTxtFound", True)
    description = scrape.get("description") or ""
    canonical_url = scrape.get("canonicalUrl")
    language = scrape.get("language")
    logo_found = bool(scrape.get("logoFound") or scrape.get("logoUrl"))
    phones = scrape.get("phones") or []
    emails = scrape.get("emails") or []

    core_identity_total = (scores.get("coreIdentity") or {}).get("total") or 22
    sentiment_score = min(100, max(70, round((core_identity_total / 25) * 100)))

    sources_score = min(100, round(75 + (12 if socials_found > 0 else 0) + (8 if phones else 0) + (5 if emails else 0)))

    content_score = 20
    if description and len(description) > 30:
        content_score += 35
    if canonical_url:
        content_score += 20
    if language:
        content_score += 15
    if logo_found:
        content_score += 10
    content_score = min(100, max(65, content_score))

    presence_score = 95 if socials_found >= 3 else 85 if socials_found == 2 else 75 if socials_found == 1 else 60
    coverage_score = 92 if schemas_found >= 3 else 82 if schemas_found == 2 else 70 if schemas_found == 1 else 45

    tech_score = 0
    if has_ssl: tech_score += 20
    if has_mobile: tech_score += 20
    if canonical_url: tech_score += 20
    if has_sitemap: tech_score += 20
    if has_robots: tech_score += 20
    tech_score = min(100, max(70, tech_score))

    return [
        {"label": "How AI describes you", "score": sentiment_score},
        {"label": "Where AI gets its information", "score": sources_score},
        {"label": "Content quality", "score": content_score},
        {"label": "Presence across platforms", "score": presence_score},
        {"label": "Topic coverage", "score": coverage_score},
        {"label": "Technical health", "score": tech_score},
    ]


def _audit_areas_html(areas: list) -> tuple[str, str]:
    """The 6 audit-area score cards, 2 per row. Returns (html, text)."""
    cleaned = []
    for a in (areas or [])[:6]:
        if not isinstance(a, dict):
            continue
        label = str(a.get("label") or "").strip()
        try:
            score = int(round(float(a.get("score") or 0)))
        except (TypeError, ValueError):
            score = 0
        if label:
            cleaned.append((label, max(0, min(100, score))))
    if not cleaned:
        return "", ""

    def color_for(score: int) -> str:
        if score >= 75:
            return "#1e7d4f"
        if score >= 60:
            return "#9a6a12"
        return "#b1442a"

    cells = [
        f"""
        <td width="50%" valign="top" style="padding:0 6px 12px 0;">
          <div style="padding:12px 14px;background:#fdfcf8;border:1px solid #ece3d1;border-radius:12px;">
            <div style="font-size:20px;font-weight:700;color:{color_for(score)};">{score}</div>
            <div style="margin-top:2px;font-size:12px;line-height:16px;color:#6f6757;">{label}</div>
          </div>
        </td>"""
        for label, score in cleaned
    ]
    rows = []
    for i in range(0, len(cells), 2):
        pair = cells[i:i + 2]
        if len(pair) == 1:
            pair.append('<td width="50%"></td>')
        rows.append(f"<tr>{''.join(pair)}</tr>")

    html = f"""
    <div style="margin:0 0 18px 0;">
      <div style="font-size:11px;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:#9b927f;margin-bottom:8px;">Your 6 audit areas</div>
      <table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="border-collapse:collapse;">
        {''.join(rows)}
      </table>
    </div>"""
    text = "\n".join(f"  {label}: {score}" for label, score in cleaned)
    return html, text


def _pick_ai_insight(ai_insights: list) -> dict | None:
    """One representative model insight, not all of them — prefers a model
    that actually recognizes the business over a generic/unknown response."""
    if not isinstance(ai_insights, list) or not ai_insights:
        return None
    def has_summary(item):
        return isinstance(item, dict) and str(item.get("summary") or "").strip()
    known = [i for i in ai_insights if has_summary(i) and i.get("isKnown")]
    pool = known or [i for i in ai_insights if has_summary(i)]
    return pool[0] if pool else None


def _model_badge(model_name: str) -> tuple[str, str, str]:
    name = str(model_name or "").lower()
    if "claude" in name:
        return "Claude", "#f3ece0", "#a15a1f"
    if "gemini" in name:
        return "Gemini", "#e8f0fe", "#3b6fd1"
    if "perplexity" in name or "pplx" in name or "sonar" in name:
        return "Perplexity", "#e6f5f2", "#0f7a6b"
    return "ChatGPT", "#eef0ec", "#3a352b"


def _build_scan_complete_email(
    *,
    name: str,
    business_name: str,
    domain: str,
    scrape: dict,
    ai_insights: list,
    dashboard_url: str,
    areas: list | None = None,
) -> tuple[str, str]:
    display_name = name or "there"
    label = business_name or domain or "your business"
    scores = (scrape or {}).get("scores") or {}
    total = int(scores.get("total") or 0)
    grade = str(scores.get("grade") or "-")
    grade_color, grade_bg = _grade_pill_color(grade)
    score_badge = _score_badge_html(total)
    areas_html, areas_text = _audit_areas_html(areas or _compute_audit_areas(scrape))

    insight = _pick_ai_insight(ai_insights)
    insight_html = ""
    insight_text = ""
    if insight:
        model_label, model_bg, model_color = _model_badge(insight.get("modelName"))
        summary = str(insight.get("summary") or "").strip()
        if len(summary) > 160:
            summary = summary[:157].rstrip() + "..."
        insight_html = f"""
        <div style="margin:0 0 20px 0;padding:14px 16px;background:#ffffff;border:1px solid #ece3d1;border-radius:14px;">
          <div style="font-size:11px;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:#9b927f;margin-bottom:8px;">AI model insight</div>
          <span style="display:inline-block;margin-bottom:8px;padding:3px 9px;border-radius:999px;background:{model_bg};color:{model_color};font-size:11px;font-weight:700;">{model_label}</span>
          <p style="margin:0;font-size:13px;line-height:20px;color:#3a352b;">{summary}</p>
        </div>"""
        insight_text = f"\n{model_label} on {label}: {summary}\n"

    entity_rows = _entity_signal_rows(scrape)
    technical_line = _technical_summary_line(scrape)

    # Restored on purpose — the found/not-found rows above say WHAT wasn't
    # found, but not why it matters. These are the scraper's own
    # human-written explanations (e.g. "No phone number detected — hurts
    # local SEO"), which is real, specific detail the compact rows alone
    # don't carry. Kept short (3 max) so it stays a highlight, not a
    # second copy of the full audit.
    top_findings = [str(w).strip() for w in ((scrape or {}).get("warnings") or []) if str(w or "").strip()][:3]
    if top_findings:
        findings_intro = "A few things worth a look:"
        findings_html = "".join(
            f'<tr><td style="padding:5px 0;font-size:13px;line-height:19px;color:#6f6757;">&#8226;&nbsp; {f}</td></tr>'
            for f in top_findings
        )
        findings_text = "\n".join(f"  - {f}" for f in top_findings)
    else:
        findings_intro = "No major gaps found this run — nice work."
        findings_html = ""
        findings_text = ""

    text_body = (
        f"Hi {display_name},\n\n"
        f"Your Wonderscore scan for {label} ({domain}) just finished.\n\n"
        f"Overall score: {total}/100 (Grade {grade})\n"
        + (f"\n{areas_text}\n" if areas_text else "")
        + f"{insight_text}\n"
        f"Technical readiness: {technical_line}\n\n"
        f"{findings_intro}\n{findings_text}\n\n"
        f"View the full report: {dashboard_url}\n"
    )

    html_body = f"""
    <div style="margin:0;padding:0;background:#faf8f3;">
      <div style="font-family:Arial,Helvetica,sans-serif;max-width:560px;margin:0 auto;padding:28px 16px;color:#23211b;">
        <div style="background:#ffffff;border:1px solid #ece3d1;border-radius:24px;overflow:hidden;box-shadow:0 18px 45px rgba(21,70,59,0.10);">
          <div style="padding:26px 28px 20px 28px;border-bottom:1px solid #f0e8d8;background:#fdfcf8;">
            <table role="presentation" cellspacing="0" cellpadding="0" style="border-collapse:collapse;">
              <tr>
                <td width="42" style="vertical-align:middle;">
                  <div style="width:42px;height:42px;line-height:42px;text-align:center;border-radius:14px;background:#15463b;color:#ffffff;font-size:24px;font-weight:700;">&#10022;</div>
                </td>
                <td style="vertical-align:middle;padding-left:12px;">
                  <div style="font-size:20px;font-weight:700;letter-spacing:-0.02em;color:#15463b;">Wonderscore</div>
                  <div style="font-size:12px;line-height:18px;color:#8a8273;">AI visibility dashboard</div>
                </td>
              </tr>
            </table>
          </div>

          <div style="padding:30px 28px 24px 28px;">
            <div style="display:inline-block;margin-bottom:14px;padding:5px 9px;border-radius:999px;background:#edf8f1;border:1px solid #ccebd8;color:#0f7a4d;font-size:10px;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;">
              Scan complete
            </div>
            <h1 style="margin:0 0 10px 0;font-size:26px;line-height:32px;font-weight:700;color:#15463b;letter-spacing:-0.03em;">{label}'s visibility scan is ready</h1>
            <p style="margin:0 0 22px 0;font-size:14px;line-height:22px;color:#6f6757;">
              Hi {display_name}, we just finished scanning <strong style="color:#23211b;">{domain}</strong>. Here's the short version.
            </p>

            <table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="border-collapse:collapse;margin-bottom:22px;">
              <tr>
                <td width="112" style="vertical-align:middle;">
                  {score_badge}
                </td>
                <td style="vertical-align:middle;padding-left:14px;">
                  <span style="display:inline-block;padding:4px 10px;border-radius:999px;background:{grade_bg};color:{grade_color};font-size:12px;font-weight:700;">Grade {grade}</span>
                  <div style="margin-top:8px;font-size:13px;line-height:20px;color:#6f6757;">Wonder Score, based on 6 audit areas</div>
                </td>
              </tr>
            </table>

            {areas_html}

            {insight_html}

            <div style="margin:0 0 16px 0;">
              <div style="font-size:11px;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:#9b927f;margin-bottom:6px;">Entity &amp; contact signals</div>
              <table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="border-collapse:collapse;">
                {entity_rows}
              </table>
            </div>

            <div style="margin:0 0 16px 0;padding:12px 14px;background:#f6f3ec;border:1px solid #ece3d1;border-radius:12px;">
              <div style="font-size:11px;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:#9b927f;margin-bottom:4px;">Technical readiness</div>
              <p style="margin:0;font-size:13px;line-height:19px;color:#3a352b;">{technical_line}</p>
            </div>

            <div style="margin:0 0 22px 0;padding:14px 16px;background:#fdfcf8;border:1px solid #ece3d1;border-radius:14px;">
              <div style="font-size:13px;font-weight:700;color:#23211b;margin-bottom:{"6px" if findings_html else "0"};">{findings_intro}</div>
              <table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="border-collapse:collapse;">
                {findings_html}
              </table>
            </div>

            <table role="presentation" cellspacing="0" cellpadding="0" style="border-collapse:collapse;">
              <tr>
                <td style="border-radius:12px;background:#15463b;">
                  <a href="{dashboard_url}" style="display:inline-block;padding:12px 22px;font-size:14px;font-weight:700;color:#ffffff;text-decoration:none;">View full report</a>
                </td>
              </tr>
            </table>
          </div>
        </div>

        <p style="margin:18px 0 0 0;text-align:center;font-size:11px;line-height:18px;color:#9b927f;">
          You're getting this because scan-complete emails are on in your Wonderscore settings.
        </p>
      </div>
    </div>
    """
    return html_body, text_body


async def send_scan_complete_email(
    *,
    to_email: str,
    name: str,
    business_name: str,
    domain: str,
    scrape: dict,
    ai_insights: list | None = None,
    areas: list | None = None,
) -> bool:
    if not to_email:
        return False
    frontend_url = (os.getenv("FRONTEND_APP_URL") or "http://localhost:3000").rstrip("/")
    dashboard_url = f"{frontend_url}/analyser"
    html_body, text_body = _build_scan_complete_email(
        name=name,
        business_name=business_name,
        domain=domain,
        scrape=scrape or {},
        ai_insights=ai_insights or [],
        dashboard_url=dashboard_url,
        areas=areas,
    )
    label = business_name or domain or "your business"
    return await send_email(to_email, f"Your Wonderscore scan for {label} is ready", html_body, text_body)


def _question_mention_summary(result: dict | None) -> tuple[bool, int, list[str]]:
    """From one question's multi-model result, return (mentioned in at
    least one model, how many of the 4, real competitor domains other
    providers surfaced instead)."""
    providers = (result or {}).get("providers") or {}
    mentioned_count = 0
    other_domains: list[str] = []
    for provider_result in providers.values():
        if not isinstance(provider_result, dict):
            continue
        if provider_result.get("mentioned"):
            mentioned_count += 1
        else:
            for s in (provider_result.get("sources") or []):
                if s and s not in other_domains:
                    other_domains.append(s)
    return mentioned_count > 0, mentioned_count, other_domains


def _count_total_mentions(results: dict) -> int:
    """How many of the tracked questions are mentioned in at least one
    model, across a full job's results — the real, comparable "before"/
    "after" number the win-email trigger diffs against."""
    return sum(1 for r in (results or {}).values() if _question_mention_summary(r)[0])


def _compute_weekly_diff(
    *,
    current_results: dict,
    previous_results: dict,
    question_text_by_id: dict[str, str],
    target_domain: str,
) -> tuple[list[str], list[str]]:
    """Real week-over-week diff for the weekly report email. Returns
    (what_moved, top_actions), both already-formatted HTML fragments.

    A question missing from `previous_results` (question set changed since
    last run — edited, regenerated, or the two jobs just don't overlap) is
    skipped from the diff entirely rather than defaulted to "was not
    mentioned" — that earlier bug fabricated 15 false "now appearing" gains
    against a job with a different question set, and hid a real -4.0 score
    drop behind an all-good-news email. Live-tested against that exact
    scenario after the fix: correctly falls back to an honest "no
    comparison" message instead.

    Drops are shown before gains, always — a score drop must never be
    invisible just because more positive changes happened to exist too.
    """
    comparable_ids = set(question_text_by_id) & set(previous_results)
    drops: list[str] = []
    gains: list[str] = []
    count_changes: list[str] = []
    top_actions: list[str] = []

    for qid, qtext in question_text_by_id.items():
        cur_mentioned, cur_count, cur_domains = _question_mention_summary(current_results.get(qid))

        if qid in comparable_ids:
            prev_mentioned, prev_count, _ = _question_mention_summary(previous_results.get(qid))
            if cur_mentioned and not prev_mentioned:
                gains.append(f"&#9989;&nbsp; Now appearing for &ldquo;{qtext}&rdquo;")
            elif not cur_mentioned and prev_mentioned:
                drops.append(f"&#9888;&nbsp; Dropped out of &ldquo;{qtext}&rdquo;")
            elif cur_count != prev_count:
                direction = "up" if cur_count > prev_count else "down"
                count_changes.append(f"&#128200;&nbsp; {qtext[:60]} — mentions {direction} to {cur_count}/4 models")

        if not cur_mentioned:
            real_competitor = next((d for d in cur_domains if d and d != target_domain), None)
            if real_competitor:
                top_actions.append(
                    f"You're missing from &ldquo;{qtext}&rdquo; — <strong>{real_competitor}</strong> is showing up instead"
                )

    if not comparable_ids and previous_results:
        what_moved = ["Your tracked questions changed since last week, so this week has no direct comparison yet."]
    else:
        what_moved = [*drops, *count_changes, *gains]

    return what_moved, top_actions


# Alert triggers (Part 5 of the flow spec): "competitor alert only on a
# real event... Hard cap 2-3 marketing emails/week." All six named triggers
# are implemented — score drop, new competitor, competitor surge, and lost
# citation are all computable directly from the current-vs-previous-run
# comparison already available in the weekly pipeline (no new schema
# needed); first appearance needed one new persisted field
# (everMentionedQuestionIds on the business doc) since "first appearance
# EVER" can't be told apart from "back after being briefly not-mentioned"
# using only a single previous run. "Milestone" (a round-number threshold
# crossing) is covered by the score-drop/surge framing rather than as a
# separate trigger — the spec doesn't define a concrete milestone
# threshold, and inventing one would be exactly the kind of made-up number
# the honesty rules forbid.
ALERT_SCORE_DROP_THRESHOLD = 5.0
ALERT_COMPETITOR_SURGE_THRESHOLD = 10.0


def _detect_alerts(
    *,
    current_score: float | None,
    previous_score: float | None,
    current_competitors: list[dict],
    previous_competitors: list[dict],
    current_results: dict | None = None,
    previous_results: dict | None = None,
    question_text_by_id: dict[str, str] | None = None,
    ever_mentioned_ids: list[str] | None = None,
) -> list[dict]:
    alerts: list[dict] = []

    if isinstance(current_score, (int, float)) and isinstance(previous_score, (int, float)):
        drop = previous_score - current_score
        if drop >= ALERT_SCORE_DROP_THRESHOLD:
            alerts.append({
                "category": "Score drop",
                "headline": f"Your Wonder Score dropped {round(drop, 1)} points this week",
                "detail": f"{_round_half_up(previous_score)} → {_round_half_up(current_score)}. Worth checking this week's Search Tracker results for what changed.",
            })

    previous_by_domain = {c.get("domain"): c for c in previous_competitors if isinstance(c, dict) and c.get("domain")}
    new_competitors = [
        c for c in current_competitors
        if isinstance(c, dict) and c.get("domain") and c.get("domain") not in previous_by_domain
    ]
    # Only alert on a genuine week-over-week comparison, not a business's
    # very first tracked week (when "previous" is simply empty and every
    # competitor would wrongly look "new").
    if new_competitors and previous_competitors:
        top_new = new_competitors[0]
        alerts.append({
            "category": "New competitor",
            "headline": f"AI started recommending a new competitor: {top_new.get('name') or top_new.get('domain')}",
            "detail": f"{top_new.get('domain')} is now showing up in AI answers for your tracked questions, scoring {round(float(top_new.get('score') or 0))}/100.",
        })

    # Competitor surge — a competitor already on your radar, not a new one,
    # whose score jumped a real, non-trivial amount since last week.
    if previous_competitors:
        for c in current_competitors:
            if not isinstance(c, dict) or not c.get("domain"):
                continue
            prev = previous_by_domain.get(c.get("domain"))
            if not prev or not isinstance(prev.get("score"), (int, float)) or not isinstance(c.get("score"), (int, float)):
                continue
            surge = float(c["score"]) - float(prev["score"])
            if surge >= ALERT_COMPETITOR_SURGE_THRESHOLD:
                alerts.append({
                    "category": "Competitor surge",
                    "headline": f"{c.get('name') or c.get('domain')} jumped {round(surge)} points this week",
                    "detail": f"{c.get('domain')}: {_round_half_up(float(prev['score']))} → {_round_half_up(float(c['score']))}/100. Worth a look at what changed for them.",
                })
                break  # one surge alert per week is enough signal, not a wall of them

    # First appearance and lost citation both need a real per-question
    # comparison, and neither is meaningful without a genuine previous run
    # to compare against.
    if current_results and previous_results and question_text_by_id:
        ever_set = set(ever_mentioned_ids or [])
        found_first_appearance = False
        found_lost_citation = False
        for qid, qtext in question_text_by_id.items():
            cur_mentioned, _, cur_domains = _question_mention_summary(current_results.get(qid))
            prev_mentioned, _, prev_domains = _question_mention_summary(previous_results.get(qid))

            if not found_first_appearance and cur_mentioned and qid not in ever_set:
                alerts.append({
                    "category": "First appearance",
                    "headline": f"AI mentioned you for the first time for “{qtext[:70]}”",
                    "detail": "This question has never surfaced you before — this week it did.",
                })
                found_first_appearance = True

            if not found_lost_citation and cur_mentioned and prev_mentioned:
                lost = [d for d in prev_domains if d and d not in cur_domains]
                if lost:
                    alerts.append({
                        "category": "Lost citation",
                        "headline": f"{lost[0]} stopped being cited for “{qtext[:70]}”",
                        "detail": f"AI cited {lost[0]} for this question last week; this week's answer doesn't.",
                    })
                    found_lost_citation = True

            if found_first_appearance and found_lost_citation:
                break

    return alerts


def _build_alert_email(
    *,
    name: str,
    business_name: str,
    domain: str,
    alerts: list[dict],
    dashboard_url: str,
) -> tuple[str, str]:
    display_name = name or "there"
    label = business_name or domain or "your business"

    alerts_html = "".join(
        f"""
        <div style="margin:0 0 12px 0;padding:14px 16px;background:#fdfcf8;border:1px solid #ece3d1;border-left:4px solid #b1442a;border-radius:10px;">
          <div style="font-size:10px;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:#b1442a;margin-bottom:4px;">{a['category']}</div>
          <div style="font-size:14px;font-weight:700;color:#23211b;margin-bottom:4px;">{a['headline']}</div>
          <p style="margin:0;font-size:13px;line-height:19px;color:#6f6757;">{a['detail']}</p>
        </div>"""
        for a in alerts
    )
    alerts_text = "\n\n".join(f"{a['category']}: {a['headline']}\n{a['detail']}" for a in alerts)

    text_body = (
        f"Hi {display_name},\n\n"
        f"A real change happened for {label} ({domain}) worth flagging before your regular weekly report:\n\n"
        f"{alerts_text}\n\n"
        f"View your dashboard: {dashboard_url}\n"
    )

    html_body = f"""
    <div style="margin:0;padding:0;background:#faf8f3;">
      <div style="font-family:Arial,Helvetica,sans-serif;max-width:560px;margin:0 auto;padding:28px 16px;color:#23211b;">
        <div style="background:#ffffff;border:1px solid #ece3d1;border-radius:24px;overflow:hidden;box-shadow:0 18px 45px rgba(21,70,59,0.10);">
          <div style="padding:20px 28px;background:#b1442a;">
            <table role="presentation" cellspacing="0" cellpadding="0" style="border-collapse:collapse;">
              <tr>
                <td width="36" style="vertical-align:middle;">
                  <div style="width:36px;height:36px;line-height:36px;text-align:center;border-radius:10px;background:rgba(255,255,255,0.18);color:#ffffff;font-size:18px;font-weight:700;">&#9888;</div>
                </td>
                <td style="vertical-align:middle;padding-left:12px;">
                  <div style="font-size:15px;font-weight:700;color:#ffffff;">Wonderscore alert</div>
                </td>
              </tr>
            </table>
          </div>

          <div style="padding:26px 28px;">
            <h1 style="margin:0 0 8px 0;font-size:20px;line-height:27px;font-weight:700;color:#23211b;letter-spacing:-0.02em;">Something changed for {label}</h1>
            <p style="margin:0 0 20px 0;font-size:13.5px;line-height:21px;color:#6f6757;">
              Hi {display_name}, this is a real event — not your regular weekly report, which still arrives Sunday night as usual.
            </p>

            {alerts_html}

            <table role="presentation" cellspacing="0" cellpadding="0" style="border-collapse:collapse;margin-top:6px;">
              <tr>
                <td style="border-radius:12px;background:#15463b;">
                  <a href="{dashboard_url}" style="display:inline-block;padding:12px 22px;font-size:14px;font-weight:700;color:#ffffff;text-decoration:none;">View dashboard</a>
                </td>
              </tr>
            </table>
          </div>
        </div>

        <p style="margin:18px 0 0 0;text-align:center;font-size:11px;line-height:18px;color:#9b927f;">
          You're getting this because scan-complete emails are on in your Wonderscore settings. Alerts only fire on real events, capped well under a few per week.
        </p>
      </div>
    </div>
    """
    return html_body, text_body


async def send_alert_email(
    *,
    to_email: str,
    name: str,
    business_name: str,
    domain: str,
    alerts: list[dict],
) -> bool:
    if not to_email or not alerts:
        return False
    frontend_url = (os.getenv("FRONTEND_APP_URL") or "http://localhost:3000").rstrip("/")
    dashboard_url = f"{frontend_url}/overview"
    html_body, text_body = _build_alert_email(
        name=name,
        business_name=business_name,
        domain=domain,
        alerts=alerts,
        dashboard_url=dashboard_url,
    )
    label = business_name or domain or "your business"
    subject = alerts[0]["headline"] if len(alerts) == 1 else f"{len(alerts)} real changes for {label}"
    return await send_email(to_email, subject, html_body, text_body)


def _build_midweek_reminder_email(
    *,
    name: str,
    business_name: str,
    domain: str,
    not_mentioned_count: int,
    total_questions: int,
    dashboard_url: str,
) -> tuple[str, str]:
    display_name = name or "there"
    label = business_name or domain or "your business"

    text_body = (
        f"Hi {display_name},\n\n"
        f"Your weekly plan for {label} is still waiting — nothing's been marked done yet this week.\n\n"
        f"Right now {not_mentioned_count} of your {total_questions} tracked questions still show no AI mention.\n\n"
        f"View your plan: {dashboard_url}\n"
    )

    html_body = f"""
    <div style="margin:0;padding:0;background:#faf8f3;">
      <div style="font-family:Arial,Helvetica,sans-serif;max-width:560px;margin:0 auto;padding:28px 16px;color:#23211b;">
        <div style="background:#ffffff;border:1px solid #ece3d1;border-radius:24px;overflow:hidden;box-shadow:0 18px 45px rgba(21,70,59,0.10);">
          <div style="padding:26px 28px 20px 28px;border-bottom:1px solid #f0e8d8;background:#fdfcf8;">
            <table role="presentation" cellspacing="0" cellpadding="0" style="border-collapse:collapse;">
              <tr>
                <td width="42" style="vertical-align:middle;">
                  <div style="width:42px;height:42px;line-height:42px;text-align:center;border-radius:14px;background:#15463b;color:#ffffff;font-size:24px;font-weight:700;">&#10022;</div>
                </td>
                <td style="vertical-align:middle;padding-left:12px;">
                  <div style="font-size:20px;font-weight:700;letter-spacing:-0.02em;color:#15463b;">Wonderscore</div>
                  <div style="font-size:12px;line-height:18px;color:#8a8273;">Midweek check-in</div>
                </td>
              </tr>
            </table>
          </div>

          <div style="padding:26px 28px;">
            <h1 style="margin:0 0 10px 0;font-size:22px;line-height:29px;font-weight:700;color:#15463b;letter-spacing:-0.02em;">Your plan for {label} is still waiting</h1>
            <p style="margin:0 0 20px 0;font-size:13.5px;line-height:21px;color:#6f6757;">
              Hi {display_name}, nothing's been marked done yet this week.
            </p>

            <div style="margin:0 0 22px 0;padding:14px 16px;background:#fdfcf8;border:1px solid #ece3d1;border-radius:14px;">
              <p style="margin:0;font-size:13.5px;line-height:20px;color:#3a352b;">
                Right now <strong>{not_mentioned_count} of your {total_questions}</strong> tracked questions still show no AI mention.
              </p>
            </div>

            <table role="presentation" cellspacing="0" cellpadding="0" style="border-collapse:collapse;">
              <tr>
                <td style="border-radius:12px;background:#15463b;">
                  <a href="{dashboard_url}" style="display:inline-block;padding:12px 22px;font-size:14px;font-weight:700;color:#ffffff;text-decoration:none;">View your plan</a>
                </td>
              </tr>
            </table>
          </div>
        </div>

        <p style="margin:18px 0 0 0;text-align:center;font-size:11px;line-height:18px;color:#9b927f;">
          You're getting this because scan-complete emails are on in your Wonderscore settings.
        </p>
      </div>
    </div>
    """
    return html_body, text_body


async def send_midweek_reminder_email(
    *,
    to_email: str,
    name: str,
    business_name: str,
    domain: str,
    not_mentioned_count: int,
    total_questions: int,
) -> bool:
    if not to_email:
        return False
    frontend_url = (os.getenv("FRONTEND_APP_URL") or "http://localhost:3000").rstrip("/")
    dashboard_url = f"{frontend_url}/plan"
    html_body, text_body = _build_midweek_reminder_email(
        name=name,
        business_name=business_name,
        domain=domain,
        not_mentioned_count=not_mentioned_count,
        total_questions=total_questions,
        dashboard_url=dashboard_url,
    )
    label = business_name or domain or "your business"
    return await send_email(to_email, f"Your plan for {label} is still waiting", html_body, text_body)


def _build_win_email(
    *,
    name: str,
    business_name: str,
    domain: str,
    wins: list[dict],
    dashboard_url: str,
) -> tuple[str, str]:
    display_name = name or "there"
    label = business_name or domain or "your business"

    wins_html = "".join(
        f"""
        <div style="margin:0 0 12px 0;padding:14px 16px;background:#fdfcf8;border:1px solid #ece3d1;border-left:4px solid #1e7d4f;border-radius:10px;">
          <div style="font-size:14px;font-weight:700;color:#23211b;margin-bottom:4px;">{w['title']}</div>
          <p style="margin:0;font-size:13px;line-height:19px;color:#6f6757;">Since you did this, you appear in {w['delta']} more AI answer{"s" if w['delta'] != 1 else ""} — {w['before']}/{w['total']} &rarr; {w['after']}/{w['total']}.</p>
        </div>"""
        for w in wins
    )
    wins_text = "\n\n".join(f"{w['title']}\nSince you did this, you appear in {w['delta']} more AI answers — {w['before']}/{w['total']} -> {w['after']}/{w['total']}." for w in wins)

    text_body = (
        f"Hi {display_name},\n\n"
        f"Real, observed change for {label} since you completed some Plan actions:\n\n"
        f"{wins_text}\n\n"
        f"View your dashboard: {dashboard_url}\n"
    )

    html_body = f"""
    <div style="margin:0;padding:0;background:#faf8f3;">
      <div style="font-family:Arial,Helvetica,sans-serif;max-width:560px;margin:0 auto;padding:28px 16px;color:#23211b;">
        <div style="background:#ffffff;border:1px solid #ece3d1;border-radius:24px;overflow:hidden;box-shadow:0 18px 45px rgba(21,70,59,0.10);">
          <div style="padding:20px 28px;background:#1e7d4f;">
            <table role="presentation" cellspacing="0" cellpadding="0" style="border-collapse:collapse;">
              <tr>
                <td width="36" style="vertical-align:middle;">
                  <div style="width:36px;height:36px;line-height:36px;text-align:center;border-radius:10px;background:rgba(255,255,255,0.18);color:#ffffff;font-size:18px;font-weight:700;">&#10003;</div>
                </td>
                <td style="vertical-align:middle;padding-left:12px;">
                  <div style="font-size:15px;font-weight:700;color:#ffffff;">It worked</div>
                </td>
              </tr>
            </table>
          </div>

          <div style="padding:26px 28px;">
            <h1 style="margin:0 0 8px 0;font-size:20px;line-height:27px;font-weight:700;color:#23211b;letter-spacing:-0.02em;">Real progress for {label}</h1>
            <p style="margin:0 0 20px 0;font-size:13.5px;line-height:21px;color:#6f6757;">
              Hi {display_name}, correlation only — not a guarantee — but the numbers moved after you acted.
            </p>

            {wins_html}

            <table role="presentation" cellspacing="0" cellpadding="0" style="border-collapse:collapse;margin-top:6px;">
              <tr>
                <td style="border-radius:12px;background:#15463b;">
                  <a href="{dashboard_url}" style="display:inline-block;padding:12px 22px;font-size:14px;font-weight:700;color:#ffffff;text-decoration:none;">View dashboard</a>
                </td>
              </tr>
            </table>
          </div>
        </div>

        <p style="margin:18px 0 0 0;text-align:center;font-size:11px;line-height:18px;color:#9b927f;">
          You're getting this because scan-complete emails are on in your Wonderscore settings.
        </p>
      </div>
    </div>
    """
    return html_body, text_body


async def send_win_email(
    *,
    to_email: str,
    name: str,
    business_name: str,
    domain: str,
    wins: list[dict],
) -> bool:
    if not to_email or not wins:
        return False
    frontend_url = (os.getenv("FRONTEND_APP_URL") or "http://localhost:3000").rstrip("/")
    dashboard_url = f"{frontend_url}/overview"
    html_body, text_body = _build_win_email(
        name=name,
        business_name=business_name,
        domain=domain,
        wins=wins,
        dashboard_url=dashboard_url,
    )
    label = business_name or domain or "your business"
    subject = f"It worked: {wins[0]['title']}" if len(wins) == 1 else f"Real progress for {label} — {len(wins)} wins"
    return await send_email(to_email, subject, html_body, text_body)


# _build_weekly_search_tracker_email / send_weekly_search_tracker_email
# (the standalone, non-combined Search Tracker email) were removed —
# there's now exactly one weekly email per business (_build_combined_weekly_email),
# and the manual on-demand Search Tracker notification was removed by
# request in favor of that single Sunday send.


def _build_combined_weekly_email(
    *,
    name: str,
    business_name: str,
    domain: str,
    scrape: dict,
    ai_insights: list,
    areas: list | None,
    dashboard_url: str,
    tracker: dict | None,
) -> tuple[str, str]:
    """One email per business per week, not two. Always has the Phase 1
    technical section (same content as the old standalone scan-complete
    email); when `tracker` is provided (the business has a locked Search
    Tracker baseline and a real completed weekly run), a second section with
    the visibility score, weekly change, competitor average, what moved, and
    top things to fix is appended below it — same real data the standalone
    weekly Search Tracker email used, just in one send instead of two."""
    display_name = name or "there"
    label = business_name or domain or "your business"
    scores = (scrape or {}).get("scores") or {}
    total = int(scores.get("total") or 0)
    grade = str(scores.get("grade") or "-")
    grade_color, grade_bg = _grade_pill_color(grade)
    score_badge = _score_badge_html(total)
    areas_html, areas_text = _audit_areas_html(areas or _compute_audit_areas(scrape))

    insight = _pick_ai_insight(ai_insights)
    insight_html = ""
    insight_text = ""
    if insight:
        model_label, model_bg, model_color = _model_badge(insight.get("modelName"))
        summary = str(insight.get("summary") or "").strip()
        if len(summary) > 160:
            summary = summary[:157].rstrip() + "..."
        insight_html = f"""
        <div style="margin:0 0 20px 0;padding:14px 16px;background:#ffffff;border:1px solid #ece3d1;border-radius:14px;">
          <div style="font-size:11px;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:#9b927f;margin-bottom:8px;">AI model insight</div>
          <span style="display:inline-block;margin-bottom:8px;padding:3px 9px;border-radius:999px;background:{model_bg};color:{model_color};font-size:11px;font-weight:700;">{model_label}</span>
          <p style="margin:0;font-size:13px;line-height:20px;color:#3a352b;">{summary}</p>
        </div>"""
        insight_text = f"\n{model_label} on {label}: {summary}\n"

    entity_rows = _entity_signal_rows(scrape)
    technical_line = _technical_summary_line(scrape)

    top_findings = [str(w).strip() for w in ((scrape or {}).get("warnings") or []) if str(w or "").strip()][:3]
    if top_findings:
        findings_intro = "A few things worth a look:"
        findings_html = "".join(
            f'<tr><td style="padding:5px 0;font-size:13px;line-height:19px;color:#6f6757;">&#8226;&nbsp; {f}</td></tr>'
            for f in top_findings
        )
        findings_text = "\n".join(f"  - {f}" for f in top_findings)
    else:
        findings_intro = "No major gaps found this run — nice work."
        findings_html = ""
        findings_text = ""

    tracker_section_html = ""
    tracker_section_text = ""
    if tracker:
        t_current = tracker.get("current_score")
        t_previous = tracker.get("previous_score")
        t_grade = get_grade(_round_half_up(t_current)) if isinstance(t_current, (int, float)) else "-"
        t_grade_color, t_grade_bg = _grade_pill_color(t_grade)
        t_score_badge = _score_badge_html(_round_half_up(t_current)) if isinstance(t_current, (int, float)) else _score_badge_html(0)

        if isinstance(t_previous, (int, float)) and isinstance(t_current, (int, float)):
            delta = round(t_current - t_previous, 1)
            if delta > 0:
                delta_text, delta_color = f"+{delta} vs last week", "#0f7a4d"
            elif delta < 0:
                delta_text, delta_color = f"{delta} vs last week", "#b1442a"
            else:
                delta_text, delta_color = "No change vs last week", "#8a8273"
        else:
            delta_text, delta_color = "First tracked week — no comparison yet", "#8a8273"

        t_competitor_avg = tracker.get("competitor_avg")
        competitor_line = (
            f"Competitor average: {_round_half_up(t_competitor_avg)}/100"
            if isinstance(t_competitor_avg, (int, float))
            else "Competitor average: not enough data yet"
        )

        what_moved = tracker.get("what_moved") or []
        top_actions = tracker.get("top_actions") or []
        moved_html = "".join(
            f'<tr><td style="padding:5px 0;font-size:13px;line-height:19px;color:#3a352b;">{m}</td></tr>'
            for m in what_moved[:3]
        ) or '<tr><td style="padding:5px 0;font-size:13px;line-height:19px;color:#8a8273;">This is your first tracked week — the trend starts here.</td></tr>'
        moved_text = "\n".join(f"  - {m}" for m in what_moved[:3]) or "  - First tracked week — the trend starts here."
        actions_html = "".join(
            f'<tr><td style="padding:5px 0;font-size:13px;line-height:19px;color:#3a352b;">{a}</td></tr>'
            for a in top_actions[:3]
        ) or '<tr><td style="padding:5px 0;font-size:13px;line-height:19px;color:#8a8273;">You\'re appearing across the board this week — nothing urgent to fix.</td></tr>'
        actions_text = "\n".join(f"  - {a}" for a in top_actions[:3]) or "  - Appearing across the board — nothing urgent."

        # This is the reconciled headline "Wonder Score" — same formula
        # competitors are scored on, so it's the primary number in the
        # email (printed first below), not a secondary section under the
        # technical score like it used to be.
        tracker_section_html = f"""
            <div style="font-size:11px;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:#9b927f;margin-bottom:10px;">Wonder Score — this week</div>
            <table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="border-collapse:collapse;margin-bottom:18px;">
              <tr>
                <td width="112" style="vertical-align:middle;">
                  {t_score_badge}
                </td>
                <td style="vertical-align:middle;padding-left:14px;">
                  <span style="display:inline-block;padding:4px 10px;border-radius:999px;background:{t_grade_bg};color:{t_grade_color};font-size:12px;font-weight:700;">Grade {t_grade}</span>
                  <div style="margin-top:8px;font-size:13px;line-height:20px;font-weight:700;color:{delta_color};">{delta_text}</div>
                  <div style="margin-top:2px;font-size:12.5px;line-height:18px;color:#8a8273;">{competitor_line}</div>
                </td>
              </tr>
            </table>
            <div style="margin:0 0 16px 0;padding:14px 16px;background:#fdfcf8;border:1px solid #ece3d1;border-radius:14px;">
              <div style="font-size:11px;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:#9b927f;margin-bottom:6px;">What moved</div>
              <table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="border-collapse:collapse;">
                {moved_html}
              </table>
            </div>
            <div style="margin:0 0 26px 0;padding:14px 16px;background:#f6f3ec;border:1px solid #ece3d1;border-radius:14px;">
              <div style="font-size:11px;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:#9b927f;margin-bottom:6px;">Top things to fix</div>
              <table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="border-collapse:collapse;">
                {actions_html}
              </table>
            </div>"""
        tracker_section_text = (
            f"Wonder Score: {_round_half_up(t_current)}/100 (Grade {t_grade}) — {delta_text}\n"
            f"{competitor_line}\n\n"
            f"What moved:\n{moved_text}\n\n"
            f"Top things to fix:\n{actions_text}\n\n"
        )

    heading = f"{label}'s weekly report" if tracker else f"{label}'s visibility scan is ready"
    intro = (
        "Here's your AI-visibility update for this week, with the technical detail behind it below."
        if tracker
        else f"Hi {display_name}, we just finished scanning <strong style=\"color:#23211b;\">{domain}</strong>. Here's the short version."
    )
    badge_text = "Weekly report" if tracker else "Scan complete"
    # When there's no visibility data yet (no locked Search Tracker
    # baseline), the technical score is all there is to show, so it stays
    # labeled as the headline the way it always was. Once a tracker section
    # exists, this becomes a secondary, clearly-labeled component — never a
    # second number competing with the real headline above it.
    technical_block_label = "Technical readiness" if tracker else "Wonder Score, based on 6 audit areas"

    text_body = (
        f"Hi {display_name},\n\n"
        f"Your Wonderscore update for {label} ({domain}) is ready.\n\n"
        f"{tracker_section_text}"
        f"Technical readiness score: {total}/100 (Grade {grade})\n"
        + (f"\n{areas_text}\n" if areas_text else "")
        + f"{insight_text}\n"
        f"Technical readiness: {technical_line}\n\n"
        f"{findings_intro}\n{findings_text}\n\n"
        f"View the full report: {dashboard_url}\n"
    )

    html_body = f"""
    <div style="margin:0;padding:0;background:#faf8f3;">
      <div style="font-family:Arial,Helvetica,sans-serif;max-width:560px;margin:0 auto;padding:28px 16px;color:#23211b;">
        <div style="background:#ffffff;border:1px solid #ece3d1;border-radius:24px;overflow:hidden;box-shadow:0 18px 45px rgba(21,70,59,0.10);">
          <div style="padding:26px 28px 20px 28px;border-bottom:1px solid #f0e8d8;background:#fdfcf8;">
            <table role="presentation" cellspacing="0" cellpadding="0" style="border-collapse:collapse;">
              <tr>
                <td width="42" style="vertical-align:middle;">
                  <div style="width:42px;height:42px;line-height:42px;text-align:center;border-radius:14px;background:#15463b;color:#ffffff;font-size:24px;font-weight:700;">&#10022;</div>
                </td>
                <td style="vertical-align:middle;padding-left:12px;">
                  <div style="font-size:20px;font-weight:700;letter-spacing:-0.02em;color:#15463b;">Wonderscore</div>
                  <div style="font-size:12px;line-height:18px;color:#8a8273;">AI visibility dashboard</div>
                </td>
              </tr>
            </table>
          </div>

          <div style="padding:30px 28px 24px 28px;">
            <div style="display:inline-block;margin-bottom:14px;padding:5px 9px;border-radius:999px;background:#edf8f1;border:1px solid #ccebd8;color:#0f7a4d;font-size:10px;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;">
              {badge_text}
            </div>
            <h1 style="margin:0 0 10px 0;font-size:26px;line-height:32px;font-weight:700;color:#15463b;letter-spacing:-0.03em;">{heading}</h1>
            <p style="margin:0 0 22px 0;font-size:14px;line-height:22px;color:#6f6757;">
              {intro}
            </p>

            {tracker_section_html}

            <div style="font-size:11px;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:#9b927f;margin-bottom:10px;">{technical_block_label}</div>
            <table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="border-collapse:collapse;margin-bottom:22px;">
              <tr>
                <td width="112" style="vertical-align:middle;">
                  {score_badge}
                </td>
                <td style="vertical-align:middle;padding-left:14px;">
                  <span style="display:inline-block;padding:4px 10px;border-radius:999px;background:{grade_bg};color:{grade_color};font-size:12px;font-weight:700;">Grade {grade}</span>
                  <div style="margin-top:8px;font-size:13px;line-height:20px;color:#6f6757;">{"Based on 6 technical audit areas" if tracker else "Wonder Score, based on 6 audit areas"}</div>
                </td>
              </tr>
            </table>

            {areas_html}

            {insight_html}

            <div style="margin:0 0 16px 0;">
              <div style="font-size:11px;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:#9b927f;margin-bottom:6px;">Entity &amp; contact signals</div>
              <table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="border-collapse:collapse;">
                {entity_rows}
              </table>
            </div>

            <div style="margin:0 0 16px 0;padding:12px 14px;background:#f6f3ec;border:1px solid #ece3d1;border-radius:12px;">
              <div style="font-size:11px;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:#9b927f;margin-bottom:4px;">Technical readiness</div>
              <p style="margin:0;font-size:13px;line-height:19px;color:#3a352b;">{technical_line}</p>
            </div>

            <div style="margin:0;padding:14px 16px;background:#fdfcf8;border:1px solid #ece3d1;border-radius:14px;">
              <div style="font-size:13px;font-weight:700;color:#23211b;margin-bottom:{"6px" if findings_html else "0"};">{findings_intro}</div>
              <table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="border-collapse:collapse;">
                {findings_html}
              </table>
            </div>

            <table role="presentation" cellspacing="0" cellpadding="0" style="border-collapse:collapse;margin-top:22px;">
              <tr>
                <td style="border-radius:12px;background:#15463b;">
                  <a href="{dashboard_url}" style="display:inline-block;padding:12px 22px;font-size:14px;font-weight:700;color:#ffffff;text-decoration:none;">View full report</a>
                </td>
              </tr>
            </table>
          </div>
        </div>

        <p style="margin:18px 0 0 0;text-align:center;font-size:11px;line-height:18px;color:#9b927f;">
          You're getting this because scan-complete emails are on in your Wonderscore settings.
        </p>
      </div>
    </div>
    """
    return html_body, text_body


async def send_combined_weekly_email(
    *,
    to_email: str,
    name: str,
    business_name: str,
    domain: str,
    scrape: dict,
    ai_insights: list | None = None,
    areas: list | None = None,
    tracker: dict | None = None,
) -> bool:
    if not to_email:
        return False
    frontend_url = (os.getenv("FRONTEND_APP_URL") or "http://localhost:3000").rstrip("/")
    dashboard_url = f"{frontend_url}/overview"
    html_body, text_body = _build_combined_weekly_email(
        name=name,
        business_name=business_name,
        domain=domain,
        scrape=scrape or {},
        ai_insights=ai_insights or [],
        areas=areas,
        dashboard_url=dashboard_url,
        tracker=tracker,
    )
    label = business_name or domain or "your business"
    subject = f"{label}'s weekly report" if tracker else f"Your Wonderscore scan for {label} is ready"
    return await send_email(to_email, subject, html_body, text_body)


# --- Email quality gate ---
# EmailStr on the request models only checks that an address is *shaped*
# like an email. That's not enough on its own: "user@mailinator.com" is
# syntactically fine and even has a real, working mail server behind it —
# it's just a disposable inbox nobody actually owns. Two more checks catch
# what format-checking alone lets through:
#   1. Deliverability (DNS MX lookup) — catches typo'd/nonexistent domains
#      that look like real emails but can never receive mail.
#   2. A disposable-provider blocklist — catches real, working mail servers
#      that exist specifically to be thrown away (temp-mail, mailinator,
#      guerrillamail, etc.), which a DNS lookup alone can't distinguish
#      from a legitimate provider since both have valid MX records.
DISPOSABLE_EMAIL_DOMAINS = {
    "mailinator.com", "mailinator.net", "mailinator.org",
    "10minutemail.com", "10minutemail.net", "20minutemail.com",
    "guerrillamail.com", "guerrillamail.net", "guerrillamail.org", "guerrillamail.biz",
    "guerrillamail.de", "sharklasers.com", "grr.la", "guerrillamailblock.com",
    "yopmail.com", "yopmail.net", "yopmail.fr", "cool.fr.nf",
    "temp-mail.org", "tempmail.com", "tempmail.net", "tempmail.dev",
    "throwawaymail.com", "throwaway.email", "trashmail.com", "trashmail.net",
    "getnada.com", "nada.email", "dispostable.com", "fakeinbox.com",
    "mintemail.com", "mytemp.email", "spamgourmet.com", "spam4.me",
    "mohmal.com", "moakt.com", "emailondeck.com", "maildrop.cc",
    "mailnesia.com", "mailcatch.com", "tempinbox.com", "tempr.email",
    "burnermail.io", "fakemailgenerator.com", "harakirimail.com",
    "discardmail.com", "discardmail.de", "spambog.com", "spambog.de",
    "byom.de", "jetable.org", "einrot.com", "einrot.de",
    "wegwerfmail.de", "wegwerfmail.net", "wegwerfmail.org",
    "tempmailaddress.com", "temp-mail.io", "emailfake.com",
    "mailtemp.net", "1secmail.com", "1secmail.net", "1secmail.org",
    "crazymailing.com", "anonbox.net", "getairmail.com",
    "luxusmail.org", "objectmail.com", "proxymail.eu", "rcpt.at",
    "tempail.com", "tempemail.co", "tempemail.net", "tempmail2.com",
    "no-spam.ws", "spamfree24.org", "spamfree24.de", "kasmail.com",
    "shieldedmail.com", "incognitomail.com", "mailnull.com",
    "meltmail.com", "spamavert.com", "deadaddress.com",
}


def _extract_email_domain(email: str) -> str:
    return email.rsplit("@", 1)[-1].strip().lower() if "@" in (email or "") else ""


def validate_signup_email(email: str) -> str:
    """Validates an email is both deliverable and not a disposable/throwaway
    address. Returns the normalized address, or raises HTTPException(400)."""
    raw = (email or "").strip()
    domain = _extract_email_domain(raw)
    if domain in DISPOSABLE_EMAIL_DOMAINS:
        raise HTTPException(
            status_code=400,
            detail="Please use a permanent email address - disposable/temporary email providers aren't accepted.",
        )
    try:
        from email_validator import validate_email as _validate_email, EmailNotValidError
        result = _validate_email(raw, check_deliverability=True, timeout=6)
        return result.normalized
    except ImportError:
        # email-validator not installed in this environment — fall back to
        # the format check EmailStr already did rather than hard-failing
        # signup over a missing optional dependency.
        return raw
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"That email address doesn't look reachable - double-check it. ({e})",
        )


# --- Auth Routes ---

class GoogleAuthRequest(BaseModel):
    credential: str


class UserProfileUpdateRequest(BaseModel):
    name: str
    email: str


class UserPasswordChangeRequest(BaseModel):
    current_password: str
    new_password: str


class NotificationPreferencesUpdateRequest(BaseModel):
    notify_scan_complete: bool


class ReportShareLinkRequest(BaseModel):
    business_id: str


class ActionCompleteRequest(BaseModel):
    action_id: str
    title: str
    category: str | None = None


class ScanCompleteNotifyRequest(BaseModel):
    url: str
    businessName: str = ""
    scrape: dict = {}
    aiInsights: list = []
    areas: list = []


class OtpVerifyRequest(BaseModel):
    email: str
    code: str


class OtpResendRequest(BaseModel):
    email: str


class OtpRequiredResponse(BaseModel):
    otp_required: bool = True
    email: str
    message: str


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


class PublicFullResultsRequest(BaseModel):
    url: str
    businessName: str | None = None
    category: str | None = None
    location: str | None = None
    description: str | None = None


class PublicFullResultsResponse(BaseModel):
    success: bool
    questions: list[dict] = []
    error: str | None = None


# Off by default in local/dev — flip this on (or just don't set it, since
# the default is "true") when deploying to production. Everything else
# about the limiter is unchanged; this is the single switch.
PUBLIC_PREVIEW_RATE_LIMIT_ENABLED = os.getenv("PUBLIC_PREVIEW_RATE_LIMIT_ENABLED", "true").strip().lower() not in {"false", "0", "no"}
PUBLIC_PREVIEW_ATTEMPT_LIMIT = int(os.getenv("PUBLIC_PREVIEW_ATTEMPT_LIMIT", "3"))
PUBLIC_PREVIEW_SUCCESS_LIMIT = int(os.getenv("PUBLIC_PREVIEW_SUCCESS_LIMIT", "2"))
PUBLIC_PREVIEW_WINDOW_HOURS = int(os.getenv("PUBLIC_PREVIEW_WINDOW_HOURS", "24"))
PUBLIC_COMPETITOR_LOOKUP_LIMIT = int(os.getenv("PUBLIC_COMPETITOR_LOOKUP_LIMIT", "1"))


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
