# Generated from the former backend/main.py lines 1676-2111.
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
                if "claude" in providers:
                    models_ran.append("claude")

            if not models_ran:
                model_name = str(run.get("model") or "").strip().lower()
                if model_name in {"openai", "chatgpt"}:
                    models_ran = ["chatgpt"]
                elif model_name in {"perplexity", "gemini", "claude"}:
                    models_ran = [model_name]
                elif model_name == "multi":
                    models_ran = ["perplexity", "chatgpt", "claude"]

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

    now_iso = datetime.utcnow().isoformat() + "Z" 
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
            if "claude" in providers:
                models_ran.append("claude")
        if not models_ran:
            model_name = str(run.get("model") or "").strip().lower()
            if model_name in {"openai", "chatgpt"}:
                models_ran = ["chatgpt"]
            elif model_name in {"perplexity", "gemini", "claude"}:
                models_ran = [model_name]
            elif model_name == "multi":
                models_ran = ["perplexity", "chatgpt", "claude"]

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
async def api_scrape(
    request: ScrapeRequest,
    http_request: Request,
    current_user: dict = Depends(get_current_user_optional),
):
    public_preview_context = await _reserve_public_preview_attempt(
        request=http_request,
        current_user=current_user,
        url=request.url,
    )
    try:
        started_at = datetime.utcnow()
        print(f"[API] /api/scrape started: {request.url}")
        try:
            business = await _upsert_user_business(
                current_user=current_user,
                url=request.url,
                category=request.category,
                location=request.location,
                business_id=request.business_id,
            )
            doc = {
                "url": request.url, 
                "phase": "phase1", 
                "timestamp": datetime.utcnow().isoformat()
            }
            if current_user:
                doc["user_id"] = current_user["id"]
                doc["user_email"] = current_user["email"]
            if request.category:
                doc["category"] = request.category
            if request.location:
                doc["location"] = request.location
            if business:
                doc["business_id"] = str(business.get("_id"))
            await urls_col.insert_one(doc)
        except:
            pass
        scrape_timeout_seconds = int(os.getenv("PHASE1_SCRAPE_TIMEOUT_SECONDS", "420"))
        print(f"[API] /api/scrape timeout budget: {scrape_timeout_seconds}s")
        result = await asyncio.wait_for(
            asyncio.to_thread(_run_scrape_worker, request.url),
            timeout=scrape_timeout_seconds,
        )
        try:
            if isinstance(result, dict):
                business = await _upsert_user_business(
                    current_user=current_user,
                    url=request.url,
                    category=request.category,
                    location=request.location,
                    business_name=result.get("businessName"),
                    logo_url=result.get("logoUrl"),
                    phase1_score=((result.get("scores") or {}).get("total") if isinstance(result.get("scores"), dict) else None),
                    business_id=request.business_id,
                    scrape_result=result,
                )
                public_business = _public_business_doc(business)
                if public_business:
                    result["businessId"] = public_business["id"]
                    result["businessProfile"] = public_business
        except Exception as business_error:
            print(f"[Business] phase1 upsert failed: {business_error}")

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
        await _mark_public_preview_success(public_preview_context)
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
                try:
                    business = await _upsert_user_business(
                        current_user=current_user,
                        url=request.url,
                        category=request.category,
                        location=request.location,
                        business_name=fallback.get("businessName"),
                        logo_url=fallback.get("logoUrl"),
                        phase1_score=((fallback.get("scores") or {}).get("total") if isinstance(fallback.get("scores"), dict) else None),
                        business_id=request.business_id,
                        scrape_result=fallback,
                    )
                    public_business = _public_business_doc(business)
                    if public_business:
                        fallback["businessId"] = public_business["id"]
                        fallback["businessProfile"] = public_business
                except Exception as business_error:
                    print(f"[Business] phase1 fallback upsert failed: {business_error}")
            print(f"[API] /api/scrape reduced fallback returned: {request.url}")
            await _mark_public_preview_success(public_preview_context)
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
