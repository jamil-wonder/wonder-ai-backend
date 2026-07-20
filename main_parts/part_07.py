# Generated from the former backend/main.py lines 2554-2998.
async def _build_weekly_blogs_for_business(
    *,
    business_doc: dict,
    current_user: dict,
    force: bool = False,
    voice: str | None = None,
    keywords: list[str] | None = None,
) -> dict:
    if weekly_blog_suggestions_col is None or businesses_col is None:
        raise HTTPException(status_code=503, detail="weekly blog storage unavailable")

    business_id = str(business_doc["_id"])
    week_id = _blog_week_id()
    existing = await weekly_blog_suggestions_col.find_one({
        "business_id": business_id,
        "user_id": current_user["id"],
        "week_id": week_id,
    })
    if existing and not force:
        return existing

    now_iso = datetime.utcnow().isoformat() + "Z" 
    blog_voice = _clean_optional_text(voice) or _clean_optional_text(business_doc.get("blogVoice")) or _clean_optional_text(business_doc.get("aiDescription")) or _clean_optional_text(business_doc.get("businessDescription"))
    preferred_keywords = _clean_blog_list(keywords, 12) or _clean_blog_list(business_doc.get("blogKeywords") if isinstance(business_doc.get("blogKeywords"), list) else [], 12)
    services = business_doc.get("services") if isinstance(business_doc.get("services"), list) else []
    business_name = business_doc.get("businessName") or business_doc.get("normalized_domain") or "Business"

    ideas_payload = await generate_weekly_blog_ideas(
        business_name=business_name,
        category=business_doc.get("category"),
        location=business_doc.get("location"),
        services=services,
        business_voice=blog_voice,
        existing_keywords=preferred_keywords,
        selected_model="claude",
    )
    suggested_keywords = _clean_blog_list(preferred_keywords + ideas_payload.get("keywords", []), 12)

    ideas = ideas_payload.get("ideas") if isinstance(ideas_payload.get("ideas"), list) else []
    if len(ideas) < 2:
        category = business_doc.get("category") or "business"
        location = business_doc.get("location") or "local customers"
        ideas = [
            {
                "title": f"How to choose the right {category} in {location}",
                "primaryKeyword": f"{category} in {location}",
                "audience": "People comparing local options",
                "angle": "High-intent local search topic",
            },
            {
                "title": f"What makes {business_name} useful for {location} customers",
                "primaryKeyword": f"{business_name} {category}",
                "audience": "Potential customers researching the brand",
                "angle": "Brand and service clarity topic",
            },
        ]

    drafts = []
    for idx, idea in enumerate(ideas[:2]):
        generated = await generate_seo_blog(
            title=str(idea.get("title") or f"Weekly blog idea {idx + 1}"),
            target_words=1100,
            primary_keyword=str(idea.get("primaryKeyword") or (suggested_keywords[0] if suggested_keywords else "")),
            audience=str(idea.get("audience") or "Potential customers"),
            tone=blog_voice or "clear, helpful, human, specific",
            key_features=services[:8],
            selling_points=[
                str(idea.get("angle") or ""),
                str(business_doc.get("category") or ""),
                str(business_doc.get("location") or ""),
            ],
            internal_links=business_doc.get("trackedPages") if isinstance(business_doc.get("trackedPages"), list) else [],
            selected_model="claude",
        )
        drafts.append({
            **generated,
            "id": f"{week_id}-blog-{idx + 1}",
            "idea": idea,
            "humanizedScore": _humanized_blog_score(generated, blog_voice),
            "humanizeNote": "Edit a few details, add a real customer example, and adjust wording to match the business voice before publishing.",
        })
        try:
            await _log_ai_usage_event({
                "feature": "blog_seo_generator",
                "endpoint": "/api/blogs/weekly/ensure",
                "user_id": current_user.get("id"),
                "user_email": current_user.get("email"),
                "model_name": generated.get("modelUsed") or "claude",
                "model_provider": generated.get("provider") or "claude",
                "ai_calls_estimate": 1,
                "details": {
                    "title": generated.get("title"),
                    "word_count": generated.get("wordCount"),
                    "target_words": 1100,
                },
            })
        except Exception as e:
            print(f"[Weekly Suggestions] log usage event failed: {e}")

    doc = {
        "user_id": current_user["id"],
        "user_email": current_user.get("email"),
        "business_id": business_id,
        "business_name": business_name,
        "week_id": week_id,
        "voice": blog_voice,
        "keywords": suggested_keywords,
        "drafts": drafts,
        "modelUsed": "claude",
        "created_at": now_iso,
        "updated_at": now_iso,
    }
    await weekly_blog_suggestions_col.update_one(
        {"business_id": business_id, "user_id": current_user["id"], "week_id": week_id},
        {"$set": doc, "$setOnInsert": {"first_created_at": now_iso}},
        upsert=True,
    )
    saved = await weekly_blog_suggestions_col.find_one({
        "business_id": business_id,
        "user_id": current_user["id"],
        "week_id": week_id,
    })
    updates = {
        "blogVoice": blog_voice,
        "blogKeywords": suggested_keywords,
        "latest_weekly_blog_at": now_iso,
        "updated_at": now_iso,
    }
    if force:
        updates["last_manually_refreshed_at"] = now_iso

    await businesses_col.update_one(
        {"_id": ObjectId(business_id), "user_id": current_user["id"]},
        {"$set": updates},
    )
    return saved or doc


def _public_weekly_blog_doc(doc: dict | None) -> dict | None:
    if not doc:
        return None
    out = dict(doc)
    out["id"] = str(out.pop("_id", ""))
    return out


@app.get("/api/blogs/usage", response_model=BlogUsageResponse)
async def api_blogs_usage(current_user: dict = Depends(get_current_user)):
    return await _blog_generation_usage(current_user)


@app.get("/api/blogs/weekly")
async def api_blogs_weekly(
    business_id: str,
    current_user: dict = Depends(get_current_user),
):
    if businesses_col is None or weekly_blog_suggestions_col is None:
        raise HTTPException(status_code=503, detail="weekly blog storage unavailable")
    try:
        business = await businesses_col.find_one({
            "_id": ObjectId(business_id),
            "user_id": current_user["id"],
        })
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid business ID format")
    if not business:
        raise HTTPException(status_code=404, detail="Business profile not found")
    week_id = _blog_week_id()
    doc = await weekly_blog_suggestions_col.find_one({
        "business_id": business_id,
        "user_id": current_user["id"],
        "week_id": week_id,
    })
    return {
        "success": True,
        "weekId": week_id,
        "business": _public_business_doc(business),
        "weekly": _public_weekly_blog_doc(doc),
        "needsSetup": not bool(business.get("blogVoice")),
        "voice": business.get("blogVoice") or "",
        "keywords": business.get("blogKeywords") if isinstance(business.get("blogKeywords"), list) else [],
    }


@app.post("/api/blogs/weekly/setup")
async def api_blogs_weekly_setup(
    request: BlogWeeklySetupRequest,
    current_user: dict = Depends(get_current_user),
):
    if businesses_col is None:
        raise HTTPException(status_code=503, detail="business storage unavailable")
    try:
        business = await businesses_col.find_one({
            "_id": ObjectId(request.business_id),
            "user_id": current_user["id"],
        })
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid business ID format")
    if not business:
        raise HTTPException(status_code=404, detail="Business profile not found")
    now_iso = datetime.utcnow().isoformat() + "Z" 
    keywords = _clean_blog_list(request.keywords, 12)
    await businesses_col.update_one(
        {"_id": ObjectId(request.business_id), "user_id": current_user["id"]},
        {
            "$set": {
                "blogVoice": _clean_optional_text(request.voice) or "",
                "blogKeywords": keywords,
                "updated_at": now_iso,
            }
        },
    )
    updated = await businesses_col.find_one({"_id": ObjectId(request.business_id), "user_id": current_user["id"]})
    return {
        "success": True,
        "business": _public_business_doc(updated),
    }


@app.post("/api/blogs/weekly/ensure")
async def api_blogs_weekly_ensure(
    request: BlogWeeklyEnsureRequest,
    current_user: dict = Depends(get_current_user),
):
    if businesses_col is None or weekly_blog_suggestions_col is None:
        raise HTTPException(status_code=503, detail="weekly blog storage unavailable")
    try:
        business = await businesses_col.find_one({
            "_id": ObjectId(request.business_id),
            "user_id": current_user["id"],
        })
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid business ID format")
    if not business:
        raise HTTPException(status_code=404, detail="Business profile not found")
    doc = await _build_weekly_blogs_for_business(
        business_doc=business,
        current_user=current_user,
        force=bool(request.force),
        voice=request.voice,
        keywords=request.keywords,
    )
    # Fetch the updated business doc
    updated_business = await businesses_col.find_one({
        "_id": ObjectId(request.business_id),
        "user_id": current_user["id"],
    })
    return {
        "success": True,
        "weekly": _public_weekly_blog_doc(doc),
        "business": _public_business_doc(updated_business),
    }


@app.post("/api/blogs/generate", response_model=BlogGenerateResponse)
async def api_blogs_generate(request: BlogGenerateRequest, current_user: dict = Depends(get_current_user)):
    usage = await _blog_generation_usage(current_user)
    if usage["remaining"] <= 0:
        return BlogGenerateResponse(success=False, usage=usage, error="You have used both blog generations for this week.")

    title = str(request.title or "").strip()
    if len(title) < 4:
        return BlogGenerateResponse(success=False, usage=usage, error="A clear blog title is required.")

    selected_model = (request.selected_model or "chatgpt").strip().lower()
    if selected_model not in {"chatgpt", "perplexity", "claude"}:
        raise HTTPException(status_code=400, detail="Choose chatgpt, perplexity, or claude.")

    target_words = max(1000, min(int(request.target_words or 1200), 1500))
    try:
        generated = await generate_seo_blog(
            title=title,
            target_words=target_words,
            primary_keyword=_clean_optional_text(request.primary_keyword),
            audience=_clean_optional_text(request.audience),
            tone=_clean_optional_text(request.tone),
            key_features=_clean_blog_list(request.key_features, 10),
            selling_points=_clean_blog_list(request.selling_points, 10),
            internal_links=_clean_blog_list(request.internal_links, 8),
            selected_model=selected_model,
        )
    except Exception as e:
        print(f"[API] blog generation error: {e}")
        return BlogGenerateResponse(success=False, usage=usage, error=str(e) or "Blog generation failed.")

    sections = [
        BlogSection(
            id=str(section.get("id") or f"section-{idx + 1}"),
            label=str(section.get("label") or section.get("heading") or f"Section {idx + 1}"),
            heading=str(section.get("heading") or f"Section {idx + 1}"),
            content=str(section.get("content") or ""),
        )
        for idx, section in enumerate(generated.get("sections") or [])
        if str(section.get("content") or "").strip()
    ]
    if not sections:
        return BlogGenerateResponse(success=False, usage=usage, error="The selected model did not return usable blog sections.")

    result = BlogGenerateResult(
        title=str(generated.get("title") or title),
        metaTitle=str(generated.get("metaTitle") or title),
        metaDescription=str(generated.get("metaDescription") or ""),
        slug=str(generated.get("slug") or ""),
        excerpt=str(generated.get("excerpt") or ""),
        keywords=[str(k) for k in generated.get("keywords", []) if str(k).strip()],
        sections=sections,
        wordCount=int(generated.get("wordCount") or 0),
        modelUsed=str(generated.get("modelUsed") or selected_model),
    )

    try:
        await _log_ai_usage_event({
            "feature": "blog_seo_generator",
            "endpoint": "/api/blogs/generate",
            "user_id": current_user.get("id"),
            "user_email": current_user.get("email"),
            "model_name": result.modelUsed,
            "model_provider": generated.get("provider") or selected_model,
            "ai_calls_estimate": 1,
            "details": {
                "title": result.title,
                "word_count": result.wordCount,
                "target_words": target_words,
            },
        })
    except Exception as e:
        print(f"[API] blog generation usage logging failed: {e}")

    next_usage = await _blog_generation_usage(current_user)
    return BlogGenerateResponse(success=True, result=result, usage=next_usage)


@app.post("/api/blogs/rewrite-section", response_model=BlogRewriteSectionResponse)
async def api_blogs_rewrite_section(request: BlogRewriteSectionRequest, current_user: dict = Depends(get_current_user)):
    selected_model = (request.selected_model or "chatgpt").strip().lower()
    if selected_model not in {"chatgpt", "perplexity", "claude"}:
        raise HTTPException(status_code=400, detail="Choose chatgpt, perplexity, or claude.")

    try:
        rewritten = await rewrite_blog_section(
            title=str(request.title or "SEO blog"),
            section=request.section.model_dump(),
            full_blog_context=[section.model_dump() for section in request.full_blog_context],
            selected_model=selected_model,
            instruction=_clean_optional_text(request.instruction),
            target_words=request.target_words,
        )
    except Exception as e:
        print(f"[API] blog section rewrite error: {e}")
        return BlogRewriteSectionResponse(success=False, error=str(e) or "Section rewrite failed.")

    section = BlogSection(
        id=str(rewritten.get("id") or request.section.id),
        label=str(rewritten.get("label") or request.section.label),
        heading=str(rewritten.get("heading") or request.section.heading),
        content=str(rewritten.get("content") or request.section.content),
    )

    try:
        await _log_ai_usage_event({
            "feature": "blog_section_rewrite",
            "endpoint": "/api/blogs/rewrite-section",
            "user_id": current_user.get("id"),
            "user_email": current_user.get("email"),
            "model_name": rewritten.get("modelUsed"),
            "model_provider": rewritten.get("provider") or selected_model,
            "ai_calls_estimate": 1,
            "details": {
                "title": request.title,
                "section_id": section.id,
            },
        })
    except Exception as e:
        print(f"[API] blog rewrite usage logging failed: {e}")

    return BlogRewriteSectionResponse(success=True, section=section, modelUsed=str(rewritten.get("modelUsed") or selected_model))
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
