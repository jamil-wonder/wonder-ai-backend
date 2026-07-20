# Generated from the former backend/main.py lines 3000-3376.
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
async def api_phase5_generate_questions(
    req: Phase5QuestionsRequest,
    current_user: dict = Depends(get_current_user_optional),
):
    try:
        business_doc = None
        if current_user and businesses_col is not None and req.business_id:
            try:
                business_doc = await businesses_col.find_one({
                    "_id": ObjectId(req.business_id),
                    "user_id": current_user["id"],
                })
            except Exception:
                business_doc = None

        def _first_text(*values):
            for value in values:
                text = str(value or "").strip()
                if text:
                    return text
            return ""

        business_context = {
            "name": _first_text(
                req.businessName,
                business_doc.get("businessName") if business_doc else "",
                business_doc.get("name") if business_doc else "",
                business_doc.get("normalized_domain") if business_doc else "",
            ),
            "category": _first_text(
                req.category,
                business_doc.get("category") if business_doc else "",
            ),
            "location": _first_text(
                req.location,
                business_doc.get("location") if business_doc else "",
            ),
            "description": _first_text(
                req.description,
                business_doc.get("businessDescription") if business_doc else "",
                business_doc.get("aiDescription") if business_doc else "",
                business_doc.get("description") if business_doc else "",
            ),
            "services": (
                req.services
                or (business_doc.get("services") if business_doc else [])
                or []
            ),
        }
        question_generation = _normalize_question_generation_settings(
            req.questionGeneration
            or (business_doc.get("questionGeneration") if business_doc else None)
        )

        questions = await generate_brand_questions(
            req.url,
            business_context=business_context,
            question_counts=question_generation,
        )
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
                "branded_questions_count": question_generation["branded"],
                "non_branded_questions_count": question_generation["nonBranded"],
                "local_seo_questions_count": question_generation["localSeo"],
                "broad_seo_questions_count": question_generation["broadSeo"],
                "business_id": req.business_id,
                "has_saved_business_context": bool(business_doc),
                "has_category": bool(business_context.get("category")),
                "has_location": bool(business_context.get("location")),
            },
        })
        return {"questions": questions}
    except ValueError as e:
        print(f"[Phase5] generate-questions validation failed: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail=str(e),
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=503,
            detail="Questions cannot be generated at this moment. Please try again.",
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
