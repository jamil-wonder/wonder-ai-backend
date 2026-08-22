# Generated from the former backend/main.py lines 1271-1673.
#
# Neither signup nor login issues a token directly anymore. Both end the
# same way: an OTP gets emailed, and POST /api/auth/otp/verify is the only
# place a JWT actually gets minted. This means there is no such thing as an
# authenticated-but-unverified session — you cannot hold a valid token
# without having proven you can receive mail at that address, whether that
# proof happened at signup or (for 2FA purposes) on every subsequent login.
@app.post("/api/auth/signup", response_model=OtpRequiredResponse)
async def api_signup(user: UserCreate):
    try:
        # Reject disposable/throwaway providers and addresses whose domain
        # can't actually receive mail, on top of EmailStr's format-only check.
        normalized_email = validate_signup_email(user.email)

        existing_user = await users_col.find_one({"email": normalized_email})
        if existing_user:
            raise HTTPException(status_code=400, detail="Email already registered")

        total_users = await users_col.count_documents({})
        assigned_role = "admin" if total_users == 0 else "user"

        hashed_password = get_password_hash(user.password)
        new_user = {
            "name": user.name,
            "email": normalized_email,
            "hashed_password": hashed_password,
            "role": assigned_role,
            "status": "active",
            "email_verified": False,
            "created_at": datetime.utcnow().isoformat()
        }

        result = await users_col.insert_one(new_user)
        user_id = str(result.inserted_id)

        sent = await send_verification_email(normalized_email, user.name, user_id, purpose="signup")
        if not sent:
            # Don't leave a stuck, unreachable account behind — if we can't
            # even send the first code, there's no way for them to ever get
            # in, so surface that clearly instead of a silent dead end.
            raise HTTPException(
                status_code=502,
                detail="Account created, but the verification email couldn't be sent. Try again shortly, or contact support.",
            )

        return OtpRequiredResponse(
            email=normalized_email,
            message="We sent a verification code to your email.",
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"[API] ERROR during signup: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/api/auth/login", response_model=OtpRequiredResponse)
async def api_login(request: LoginRequest):
    try:
        user = await users_col.find_one({"email": request.email})
        if not user:
            raise HTTPException(status_code=401, detail="Invalid email or password")

        if not verify_password(request.password, user["hashed_password"]):
            raise HTTPException(status_code=401, detail="Invalid email or password")

        if user.get("status") == "banned":
            raise HTTPException(status_code=403, detail="Your account has been restricted.")

        user_id = str(user["_id"])
        sent = await send_verification_email(user.get("email", ""), user.get("name", ""), user_id, purpose="login")
        if not sent:
            raise HTTPException(
                status_code=502,
                detail="Couldn't send a sign-in code right now. Try again shortly.",
            )

        return OtpRequiredResponse(
            email=user["email"],
            message="We sent a sign-in code to your email.",
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"[API] ERROR during login: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/api/auth/otp/verify", response_model=Token)
async def api_otp_verify(request: OtpVerifyRequest):
    if email_verifications_col is None:
        raise HTTPException(status_code=503, detail="Verification storage unavailable")

    email = (request.email or "").strip().lower()
    code = (request.code or "").strip()
    if not email or not code:
        raise HTTPException(status_code=400, detail="Email and code are required")

    user = await users_col.find_one({"email": email})
    if not user:
        raise HTTPException(status_code=400, detail="Incorrect code")
    user_id = str(user["_id"])

    now = datetime.utcnow()
    record = await email_verifications_col.find_one(
        {"user_id": user_id, "used_at": None, "expires_at": {"$gt": now}},
        sort=[("created_at", -1)],
    )
    if not record:
        raise HTTPException(status_code=400, detail="No pending code for this email — request a new one.")

    if record.get("attempts", 0) >= EMAIL_OTP_MAX_ATTEMPTS:
        await email_verifications_col.update_one({"_id": record["_id"]}, {"$set": {"used_at": now}})
        raise HTTPException(status_code=429, detail="Too many incorrect attempts — request a new code.")

    code_hash = hashlib.sha256(code.encode("utf-8")).hexdigest()
    if code_hash != record.get("code_hash"):
        updated = await email_verifications_col.find_one_and_update(
            {"_id": record["_id"]},
            {"$inc": {"attempts": 1}},
            return_document=ReturnDocument.AFTER,
        )
        remaining = max(0, EMAIL_OTP_MAX_ATTEMPTS - (updated.get("attempts", 0) if updated else EMAIL_OTP_MAX_ATTEMPTS))
        raise HTTPException(status_code=400, detail=f"Incorrect code. {remaining} attempt(s) left.")

    await email_verifications_col.update_one({"_id": record["_id"]}, {"$set": {"used_at": now}})

    if user.get("status") == "banned":
        raise HTTPException(status_code=403, detail="Your account has been restricted.")

    # Receiving and entering this code is proof of ownership regardless of
    # whether it was sent for signup or a routine login — mark verified
    # either way so an old unverified account gets cleared the first time
    # its owner successfully logs back in, not just at original signup.
    await users_col.update_one({"_id": ObjectId(user_id)}, {"$set": {"email_verified": True}})

    # Welcome email fires exactly once per account — only when the OTP
    # being verified was the one sent by signup, never on a routine login's
    # 2FA code (send_verification_email tags each record with its purpose
    # at send time; see api_signup). Fire-and-forget so a slow SMTP send
    # never delays the login response.
    if record.get("purpose") == "signup":
        asyncio.create_task(send_welcome_email(user["email"], user.get("name", "")))

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["email"], "id": user_id}, expires_delta=access_token_expires
    )
    user_response = UserResponse(
        id=user_id,
        name=user.get("name", ""),
        email=user["email"],
        created_at=user.get("created_at", datetime.utcnow().isoformat()),
        role=user.get("role", "user"),
        status=user.get("status", "active"),
        email_verified=True,
        notify_scan_complete=user.get("notify_scan_complete", True),
    )
    return Token(access_token=access_token, token_type="bearer", user=user_response)


@app.post("/api/auth/otp/resend")
async def api_otp_resend(request: OtpResendRequest):
    if email_verifications_col is None:
        raise HTTPException(status_code=503, detail="Verification storage unavailable")

    email = (request.email or "").strip().lower()
    if not email:
        raise HTTPException(status_code=400, detail="Email is required")

    user = await users_col.find_one({"email": email})
    if not user:
        # Don't reveal whether this email has an account — same response
        # shape either way.
        return {"success": True, "message": "If that email has a pending code, a new one was sent."}

    user_id = str(user["_id"])
    cooldown_cutoff = datetime.utcnow() - timedelta(seconds=EMAIL_VERIFICATION_RESEND_COOLDOWN_SECONDS)
    recent = await email_verifications_col.find_one({
        "user_id": user_id,
        "created_at": {"$gt": cooldown_cutoff},
    })
    if recent:
        raise HTTPException(status_code=429, detail="Please wait a bit before requesting another code")

    sent = await send_verification_email(email, user.get("name", ""), user_id, purpose="resend")
    if not sent:
        raise HTTPException(
            status_code=502,
            detail="Couldn't send the code — the mail server rejected it or isn't configured. Try again shortly.",
        )
    return {"success": True, "message": "A new code was sent"}


@app.post("/api/auth/handoff")
async def api_auth_handoff(current_user: dict = Depends(get_current_user)):
    if auth_handoffs_col is None:
        raise HTTPException(status_code=503, detail="Auth handoff storage unavailable")

    code = secrets.token_urlsafe(48)
    code_hash = hashlib.sha256(code.encode("utf-8")).hexdigest()
    now = datetime.utcnow()
    expires_at = now + timedelta(minutes=5)
    await auth_handoffs_col.insert_one({
        "code_hash": code_hash,
        "user_id": current_user["id"],
        "user_email": current_user.get("email"),
        "created_at": now,
        "expires_at": expires_at,
        "used_at": None,
    })
    return {
        "code": code,
        "expires_at": expires_at.isoformat() + "Z",
    }


@app.post("/api/auth/handoff/exchange", response_model=Token)
async def api_auth_handoff_exchange(request: AuthHandoffExchangeRequest):
    if auth_handoffs_col is None:
        raise HTTPException(status_code=503, detail="Auth handoff storage unavailable")

    code = (request.code or "").strip()
    if not code:
        raise HTTPException(status_code=400, detail="Missing handoff code")

    code_hash = hashlib.sha256(code.encode("utf-8")).hexdigest()
    now = datetime.utcnow()
    handoff = await auth_handoffs_col.find_one_and_update(
        {
            "code_hash": code_hash,
            "used_at": None,
            "expires_at": {"$gt": now},
        },
        {"$set": {"used_at": now}},
        return_document=ReturnDocument.AFTER,
    )
    if not handoff:
        raise HTTPException(status_code=401, detail="Invalid or expired handoff code")

    try:
        user = await users_col.find_one({"_id": ObjectId(handoff["user_id"])})
    except Exception:
        user = None
    if not user:
        raise HTTPException(status_code=401, detail="User for handoff no longer exists")
    if user.get("status") == "banned":
        raise HTTPException(status_code=403, detail="Your account has been restricted.")

    user_id = str(user["_id"])
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["email"], "id": user_id}, expires_delta=access_token_expires
    )
    user_response = UserResponse(
        id=user_id,
        name=user.get("name", ""),
        email=user["email"],
        created_at=user.get("created_at", datetime.utcnow().isoformat()),
        role=user.get("role", "user"),
        status=user.get("status", "active"),
        notify_scan_complete=user.get("notify_scan_complete", True),
    )
    return Token(access_token=access_token, token_type="bearer", user=user_response)


@app.get("/api/user/profile", response_model=UserResponse)
async def api_user_profile(current_user: dict = Depends(get_current_user)):
    user = await users_col.find_one({"_id": ObjectId(current_user["id"])})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Sliding expiration: there's no separate refresh-token flow, so instead
    # every successful profile check (the frontend calls this on every app
    # load/tab reopen) quietly re-mints a fresh 7-day token. An actively
    # used session effectively never hits the hard expiry and never gets
    # bounced to OTP — only someone who doesn't open the app at all for 7
    # straight days does. The frontend swaps this in when present.
    fresh_token = create_access_token(
        data={"sub": user["email"], "id": str(user["_id"])},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )

    return UserResponse(
        id=str(user["_id"]),
        name=user.get("name", ""),
        email=user.get("email", ""),
        created_at=user.get("created_at", datetime.utcnow().isoformat()),
        role=user.get("role", "user"),
        status=user.get("status", "active"),
        email_verified=user.get("email_verified", False),
        notify_scan_complete=user.get("notify_scan_complete", True),
        access_token=fresh_token,
    )


@app.put("/api/user/profile", response_model=UserResponse)
async def api_user_profile_update(request: UserProfileUpdateRequest, current_user: dict = Depends(get_current_user)):
    name = (request.name or "").strip()
    email = (request.email or "").strip().lower()

    if not name:
        raise HTTPException(status_code=400, detail="Name is required")
    if not email:
        raise HTTPException(status_code=400, detail="Email is required")

    current = await users_col.find_one({"_id": ObjectId(current_user["id"])})
    email_changed = bool(current) and current.get("email") != email

    # Only run the (network-cost) deliverability/disposable-domain check
    # when the email is actually changing — no need to re-validate an
    # address that was already accepted, on every unrelated profile save.
    if email_changed:
        email = validate_signup_email(email)

    existing = await users_col.find_one({"email": email})
    if existing and str(existing.get("_id")) != current_user["id"]:
        raise HTTPException(status_code=400, detail="Email already in use")

    update_fields = {
        "name": name,
        "email": email,
        "updated_at": datetime.utcnow().isoformat(),
    }
    if email_changed:
        update_fields["email_verified"] = False

    updated = await users_col.find_one_and_update(
        {"_id": ObjectId(current_user["id"])},
        {"$set": update_fields},
        return_document=ReturnDocument.AFTER,
    )

    if not updated:
        raise HTTPException(status_code=404, detail="User not found")

    if email_changed:
        try:
            await send_verification_email(email, name, current_user["id"])
        except Exception as e:
            print(f"[API] Failed to send verification email to {email}: {e}")

    return UserResponse(
        id=str(updated["_id"]),
        name=updated.get("name", ""),
        email=updated.get("email", ""),
        created_at=updated.get("created_at", datetime.utcnow().isoformat()),
        role=updated.get("role", "user"),
        status=updated.get("status", "active"),
        email_verified=updated.get("email_verified", False),
        notify_scan_complete=updated.get("notify_scan_complete", True),
    )


@app.put("/api/user/notification-preferences", response_model=UserResponse)
async def api_update_notification_preferences(
    request: NotificationPreferencesUpdateRequest,
    current_user: dict = Depends(get_current_user),
):
    updated = await users_col.find_one_and_update(
        {"_id": ObjectId(current_user["id"])},
        {"$set": {"notify_scan_complete": bool(request.notify_scan_complete)}},
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
        email_verified=updated.get("email_verified", False),
        notify_scan_complete=updated.get("notify_scan_complete", True),
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


@app.get("/api/user/businesses", response_model=list[BusinessResponse])
async def api_user_businesses(current_user: dict = Depends(get_current_user)):
    if businesses_col is None:
        raise HTTPException(status_code=503, detail="business storage unavailable")

    cursor = businesses_col.find({"user_id": current_user["id"]}).sort("updated_at", -1).limit(100)
    docs = await cursor.to_list(length=100)
    return [_public_business_doc(doc) for doc in docs if _public_business_doc(doc)]


@app.post("/api/user/businesses", response_model=BusinessResponse)
async def api_user_business_upsert(
    request: BusinessUpsertRequest,
    current_user: dict = Depends(get_current_user),
):
    if businesses_col is None:
        raise HTTPException(status_code=503, detail="business storage unavailable")

    normalized_domain = _normalize_site(request.url)
    if not normalized_domain:
        raise HTTPException(status_code=400, detail="A valid business URL is required")

    existing = await businesses_col.find_one({
        "user_id": current_user["id"],
        "normalized_domain": normalized_domain,
    })
    if not existing and not request.business_id:
        current_count = await businesses_col.count_documents({"user_id": current_user["id"]})
        if current_count >= 3:
            raise HTTPException(status_code=400, detail="You can save up to 3 business profiles for now")

    # A manual scan (Analyser page) saves its result here via
    # latest_scrape_result, but this endpoint never used to forward its
    # score into latest_phase1_score/weekly_scores — only the Sunday
    # scheduler did. That meant a manual scan's score only ever lived in
    # the browser's localStorage cache, so it vanished on logout or a
    # different device. Deriving it here (not trusting a client-sent
    # score field) makes every save — manual or scheduled — update the
    # same persisted score history.
    phase1_score = None
    scrape_scores = (request.latest_scrape_result or {}).get("scores")
    if isinstance(scrape_scores, dict) and isinstance(scrape_scores.get("total"), (int, float)):
        phase1_score = int(scrape_scores["total"])

    business = await _upsert_user_business(
        current_user=current_user,
        url=request.url,
        category=request.category,
        location=request.location,
        business_name=request.businessName,
        logo_url=request.logoUrl,
        business_description=request.businessDescription,
        ai_description=request.aiDescription,
        services=request.services,
        target_audience=request.targetAudience,
        question_generation=request.questionGeneration,
        tracked_questions=request.trackedQuestions,
        competitors=request.competitors,
        system_competitors=request.systemCompetitors,
        tracked_pages=request.trackedPages,
        business_id=request.business_id,
        scrape_result=request.latest_scrape_result,
        phase1_score=phase1_score,
    )
    public = _public_business_doc(business)
    if not public:
        raise HTTPException(status_code=400, detail="Could not save business")
    try:
        asyncio.create_task(_build_weekly_blogs_for_business(
            business_doc=business,
            current_user=current_user,
            force=False,
        ))
    except Exception as e:
        print(f"[Blogs] failed to queue initial weekly blogs: {e}")
    return public


@app.post("/api/user/businesses/{business_id}/lock-questions", response_model=BusinessResponse)
async def api_user_business_lock_questions(
    business_id: str,
    current_user: dict = Depends(get_current_user),
):
    # Locking the core 20 is what makes the Wonder Score comparable week to
    # week — same basis for every customer (Part 3 / Part 6 rule 3 of the
    # platform-flow spec). One-way from here: unlocking isn't exposed at
    # all, only editing-with-warning after the fact (frontend gates that).
    if businesses_col is None:
        raise HTTPException(status_code=503, detail="business storage unavailable")
    try:
        oid = ObjectId(business_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid business ID format")

    existing = await businesses_col.find_one({"_id": oid, "user_id": current_user["id"]})
    if not existing:
        raise HTTPException(status_code=404, detail="Business not found")
    if existing.get("questionsLocked"):
        return _public_business_doc(existing)

    now_iso = datetime.utcnow().isoformat() + "Z"
    await businesses_col.update_one(
        {"_id": oid, "user_id": current_user["id"]},
        {"$set": {"questionsLocked": True, "questionsLockedAt": now_iso, "updated_at": now_iso}},
    )
    updated = await businesses_col.find_one({"_id": oid, "user_id": current_user["id"]})
    public = _public_business_doc(updated)
    if not public:
        raise HTTPException(status_code=500, detail="Could not lock questions")
    return public


@app.delete("/api/user/businesses/{business_id}")
async def api_user_business_delete(
    business_id: str,
    current_user: dict = Depends(get_current_user),
):
    if businesses_col is None:
        raise HTTPException(status_code=503, detail="business storage unavailable")

    try:
        oid = ObjectId(business_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid business ID format")

    result = await businesses_col.delete_one({
        "_id": oid,
        "user_id": current_user["id"],
    })

    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Business profile not found")

    return {"message": "Business profile deleted successfully"}


@app.post("/api/reports/share-link")
async def api_reports_share_link(
    request: ReportShareLinkRequest,
    current_user: dict = Depends(get_current_user),
):
    """The PDF spec's "Share report" (Part 4): a shareable weekly summary
    link, manager → owner or agency → client, that the recipient can open
    with no account. Generates a stable random token once per business (a
    second call for the same business returns the SAME link, not a new
    one) and stores it on the business doc — the public report endpoint
    below looks a business up by this token, nothing else."""
    if businesses_col is None:
        raise HTTPException(status_code=503, detail="business storage unavailable")

    try:
        oid = ObjectId(request.business_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid business ID format")

    business = await businesses_col.find_one({"_id": oid, "user_id": current_user["id"]})
    if not business:
        raise HTTPException(status_code=404, detail="Business profile not found")

    token = business.get("report_share_token")
    if not token:
        token = secrets.token_urlsafe(24)
        await businesses_col.update_one({"_id": oid}, {"$set": {"report_share_token": token}})

    frontend_url = (os.getenv("FRONTEND_APP_URL") or "http://localhost:3000").rstrip("/")
    return {"token": token, "url": f"{frontend_url}/report/{token}"}


@app.get("/api/public/reports/{token}")
async def api_public_report(token: str):
    """No auth — this is the recipient's side of "Share report." Per the
    spec ("Dashboard — the weekly briefing... rank vs rivals... where AI
    sees you (per-model x/20)... what moved"), this is the real weekly
    briefing, not just the technical audit — score, grade, weekly change,
    rank vs named competitors, per-model appearance out of 20, real
    what-moved lines, and the technical breakdown underneath. Never the
    account email or billing details, since this link goes to someone with
    no account at all — but the actual AI-visibility performance is the
    entire point of sharing this with an owner or client, so it's included
    the same way the account owner already sees it on their own Dashboard."""
    if businesses_col is None:
        raise HTTPException(status_code=503, detail="report storage unavailable")

    business = await businesses_col.find_one({"report_share_token": token})
    if not business:
        raise HTTPException(status_code=404, detail="Report not found")

    visibility_score = business.get("latest_phase5_score")
    technical_score = business.get("latest_phase1_score")
    headline_score = visibility_score if isinstance(visibility_score, (int, float)) else technical_score

    visibility_history = business.get("visibility_weekly_scores") or []
    previous_score = None
    if isinstance(visibility_history, list) and len(visibility_history) >= 2:
        prev_entry = visibility_history[-2]
        if isinstance(prev_entry.get("score"), (int, float)):
            previous_score = prev_entry["score"]

    system_competitors = business.get("systemCompetitors") or []
    competitor_scores = [
        float(c.get("score")) for c in system_competitors
        if isinstance(c, dict) and isinstance(c.get("score"), (int, float))
    ]
    competitor_avg = (sum(competitor_scores) / len(competitor_scores)) if competitor_scores else None

    # Rank vs named competitors — real ranking, not just an average.
    named_competitors = sorted(
        [
            {"name": c.get("name") or c.get("domain"), "domain": c.get("domain"), "score": round(float(c.get("score")))}
            for c in system_competitors
            if isinstance(c, dict) and c.get("domain") and isinstance(c.get("score"), (int, float))
        ],
        key=lambda c: c["score"],
        reverse=True,
    )[:5]
    rank = None
    if isinstance(headline_score, (int, float)) and named_competitors:
        rank = 1 + sum(1 for c in named_competitors if c["score"] > round(headline_score))

    # Real Search Tracker data: per-model appearance and what-moved, from
    # the business's most recent completed multi-model run (and the one
    # before it, for a real diff — never a guessed number).
    per_model: list[dict] = []
    what_moved: list[str] = []
    business_id = str(business.get("_id"))
    if phase5_jobs_col is not None:
        latest_job = await phase5_jobs_col.find_one(
            {"business_id": business_id, "job_type": "core", "model": "multi", "status": "completed"},
            sort=[("created_at", -1)],
        )
        if latest_job:
            provider_scores = latest_job.get("provider_scores") or {}
            model_labels = {"chatgpt": "ChatGPT", "perplexity": "Perplexity", "claude": "Claude", "gemini": "Gemini"}
            for key, label in model_labels.items():
                stats = provider_scores.get(key)
                if isinstance(stats, dict) and isinstance(stats.get("total"), (int, float)):
                    per_model.append({"model": label, "mentioned": int(stats.get("mentioned") or 0), "total": int(stats["total"])})

            previous_job = await phase5_jobs_col.find_one(
                {
                    "business_id": business_id,
                    "job_type": "core",
                    "model": "multi",
                    "status": "completed",
                    "job_id": {"$ne": latest_job.get("job_id")},
                },
                sort=[("created_at", -1)],
            )
            question_text_by_id = {str(q.get("id")): str(q.get("text") or "") for q in (latest_job.get("questions") or [])}
            what_moved, _ = _compute_weekly_diff(
                current_results=latest_job.get("results") or {},
                previous_results=(previous_job or {}).get("results") or {},
                question_text_by_id=question_text_by_id,
                target_domain=_normalize_domain(business.get("url") or ""),
            )
            # Strip the HTML entities/markup _compute_weekly_diff uses for
            # the email template — this is a plain-JSON API, the frontend
            # renders its own styling.
            what_moved = [re.sub(r"&#\d+;|&\w+;", "", m).strip() for m in what_moved[:3]]

    scrape = business.get("latest_scrape_result") or {}
    audit_areas = _compute_audit_areas(scrape) if scrape else []

    return {
        "businessName": business.get("businessName") or "",
        "domain": _normalize_site(business.get("url") or ""),
        "score": round(headline_score) if isinstance(headline_score, (int, float)) else None,
        "grade": get_grade(round(headline_score)) if isinstance(headline_score, (int, float)) else None,
        "previousScore": round(previous_score) if isinstance(previous_score, (int, float)) else None,
        "competitorAverage": round(competitor_avg) if isinstance(competitor_avg, (int, float)) else None,
        "rank": rank,
        "competitors": named_competitors,
        "perModel": per_model,
        "whatMoved": what_moved,
        "auditAreas": audit_areas,
        "updatedAt": business.get("latest_phase5_at") or business.get("latest_phase1_at") or business.get("updated_at"),
    }


@app.get(
    "/api/user/businesses/{business_id}/competitor-tracking",
    response_model=CompetitorTrackingStatusResponse,
)
async def api_competitor_tracking_status(
    business_id: str,
    current_user: dict = Depends(get_current_user),
):
    if businesses_col is None or competitor_tracking_runs_col is None:
        raise HTTPException(status_code=503, detail="competitor tracking storage unavailable")
    try:
        oid = ObjectId(business_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid business ID format")

    business = await businesses_col.find_one({"_id": oid, "user_id": current_user["id"]})
    if not business:
        raise HTTPException(status_code=404, detail="Business profile not found")

    cursor = competitor_tracking_runs_col.find(
        {"business_id": business_id, "user_id": current_user["id"]},
        {"_id": 1, "run_id": 1, "business_id": 1, "status": 1, "url": 1, "target_domain": 1, "tracked_competitors": 1, "competitors": 1, "questions": 1, "created_at": 1, "completed_at": 1, "updated_at": 1, "error": 1},
    ).sort("created_at", -1).limit(10)
    docs = await cursor.to_list(length=10)
    public_docs = [_public_tracking_run_doc(doc) for doc in docs]
    public_docs = [doc for doc in public_docs if doc]
    return {
        "success": True,
        "latest": public_docs[0] if public_docs else None,
        "history": public_docs,
    }


@app.post(
    "/api/user/businesses/{business_id}/competitor-tracking/run",
    response_model=CompetitorTrackingRunResponse,
)
async def api_competitor_tracking_run(
    business_id: str,
    request: CompetitorTrackingRunRequest,
    current_user: dict = Depends(get_current_user),
):
    if businesses_col is None or competitor_tracking_runs_col is None:
        raise HTTPException(status_code=503, detail="competitor tracking storage unavailable")
    try:
        oid = ObjectId(business_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid business ID format")

    business = await businesses_col.find_one({"_id": oid, "user_id": current_user["id"]})
    if not business:
        raise HTTPException(status_code=404, detail="Business profile not found")

    if not request.force:
        running = await competitor_tracking_runs_col.find_one({
            "business_id": business_id,
            "user_id": current_user["id"],
            "status": "running",
        })
        if running:
            return {
                "success": True,
                "run": _public_tracking_run_doc(running),
                "business": _public_business_doc(business),
            }

    run = await _run_competitor_tracking_for_business(
        business_doc=business,
        current_user=current_user,
    )
    fresh_business = await businesses_col.find_one({"_id": oid, "user_id": current_user["id"]})
    return {
        "success": True,
        "run": run,
        "business": _public_business_doc(fresh_business),
    }
