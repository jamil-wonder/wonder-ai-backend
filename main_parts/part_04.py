# Generated from the former backend/main.py lines 1271-1673.
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
    )
    return Token(access_token=access_token, token_type="bearer", user=user_response)


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
        competitors=request.competitors,
        system_competitors=request.systemCompetitors,
        tracked_pages=request.trackedPages,
        business_id=request.business_id,
        scrape_result=request.latest_scrape_result,
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
