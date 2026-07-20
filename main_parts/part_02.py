# Generated from the former backend/main.py lines 439-857.
async def _get_public_limit_doc(request: Request) -> dict:
    now = datetime.utcnow()
    key = _public_limit_key(request)
    reset_at = now + timedelta(hours=PUBLIC_PREVIEW_WINDOW_HOURS)
    default_doc = {
        "key": key,
        "attempt_count": 0,
        "preview_success_count": 0,
        "allowed_scan_ids": [],
        "endpoint_counts": {},
        "created_at": now,
        "reset_at": reset_at,
        "last_ip": _get_public_client_ip(request),
        "last_device_id": _get_public_device_id(request),
    }

    if public_rate_limits_col is None:
        return default_doc

    doc = await public_rate_limits_col.find_one({"key": key})
    existing_reset_at = doc.get("reset_at") if doc else None
    if not doc or not isinstance(existing_reset_at, datetime) or existing_reset_at <= now:
        await public_rate_limits_col.replace_one({"key": key}, default_doc, upsert=True)
        return default_doc
    return doc


def _raise_public_limit(doc: dict):
    reset_at = doc.get("reset_at")
    reset_text = reset_at.isoformat() if isinstance(reset_at, datetime) else None
    raise HTTPException(
        status_code=429,
        detail={
            "code": "public_preview_limit_reached",
            "message": "You have used today's free Wonder Score preview. Please sign up or try again after the limit resets.",
            "resetAt": reset_text,
        },
    )


async def _reserve_public_preview_attempt(
    *,
    request: Request,
    current_user: dict | None,
    url: str,
) -> dict | None:
    if current_user:
        return None

    doc = await _get_public_limit_doc(request)
    if (
        int(doc.get("attempt_count") or 0) >= PUBLIC_PREVIEW_ATTEMPT_LIMIT
        or int(doc.get("preview_success_count") or 0) >= PUBLIC_PREVIEW_SUCCESS_LIMIT
    ):
        _raise_public_limit(doc)

    scan_id = _get_public_scan_id(request)
    domain = _normalize_site(url)
    if public_rate_limits_col is not None:
        await public_rate_limits_col.update_one(
            {"key": doc["key"]},
            {
                "$inc": {"attempt_count": 1},
                "$set": {
                    "last_attempt_at": datetime.utcnow(),
                    "last_domain": domain,
                    "last_ip": _get_public_client_ip(request),
                    "last_device_id": _get_public_device_id(request),
                },
                "$addToSet": {"attempted_domains": domain},
            },
        )
    return {"key": doc["key"], "scan_id": scan_id}


async def _mark_public_preview_success(context: dict | None):
    if not context or public_rate_limits_col is None:
        return
    await public_rate_limits_col.update_one(
        {"key": context["key"]},
        {
            "$inc": {"preview_success_count": 1},
            "$set": {"last_success_at": datetime.utcnow()},
            "$addToSet": {"allowed_scan_ids": context["scan_id"]},
        },
    )


async def _enforce_public_child_call(
    *,
    request: Request,
    current_user: dict | None,
    endpoint_key: str,
    max_calls: int = 1,
):
    if current_user:
        return

    doc = await _get_public_limit_doc(request)
    scan_id = _get_public_scan_id(request)
    allowed = doc.get("allowed_scan_ids") if isinstance(doc.get("allowed_scan_ids"), list) else []
    if scan_id not in allowed:
        _raise_public_limit(doc)

    endpoint_counts = doc.get("endpoint_counts") if isinstance(doc.get("endpoint_counts"), dict) else {}
    count_key = f"{endpoint_key}_{scan_id}"
    if int(endpoint_counts.get(count_key) or 0) >= max_calls:
        _raise_public_limit(doc)

    if public_rate_limits_col is not None:
        await public_rate_limits_col.update_one(
            {"key": doc["key"]},
            {
                "$inc": {f"endpoint_counts.{count_key}": 1},
                "$set": {"last_child_call_at": datetime.utcnow()},
            },
        )


def _clean_optional_text(value) -> str | None:
    text = str(value or "").strip()
    return text or None


DEFAULT_PHASE5_QUESTION_GENERATION = {
    "branded": 5,
    "nonBranded": 0,
    "localSeo": 15,
    "broadSeo": 0,
}


def _normalize_question_generation_settings(value) -> dict:
    source = value if isinstance(value, dict) else {}
    def _safe_count(key: str) -> int:
        try:
            return max(0, int(source.get(key, DEFAULT_PHASE5_QUESTION_GENERATION[key]) or 0))
        except Exception:
            return DEFAULT_PHASE5_QUESTION_GENERATION[key]
    normalized = {
        "branded": _safe_count("branded"),
        "nonBranded": _safe_count("nonBranded"),
        "localSeo": _safe_count("localSeo"),
        "broadSeo": _safe_count("broadSeo"),
    }
    total = sum(normalized.values())
    if total <= 0:
        return dict(DEFAULT_PHASE5_QUESTION_GENERATION)
    if total > 20:
        overflow = total - 20
        for key in ["broadSeo", "localSeo", "nonBranded", "branded"]:
            remove = min(normalized[key], overflow)
            normalized[key] -= remove
            overflow -= remove
            if overflow <= 0:
                break
    elif total < 20:
        normalized["localSeo"] += 20 - total
    return normalized


def _public_business_doc(doc: dict | None) -> dict | None:
    if not doc:
        return None
    return {
        "id": str(doc.get("_id")),
        "user_id": str(doc.get("user_id") or ""),
        "url": doc.get("url") or "",
        "normalized_domain": doc.get("normalized_domain") or "",
        "businessName": doc.get("businessName"),
        "category": doc.get("category"),
        "location": doc.get("location"),
        "logoUrl": doc.get("logoUrl"),
        "businessDescription": doc.get("businessDescription"),
        "aiDescription": doc.get("aiDescription"),
        "services": doc.get("services") if isinstance(doc.get("services"), list) else [],
        "targetAudience": doc.get("targetAudience"),
        "blogVoice": doc.get("blogVoice"),
        "blogKeywords": doc.get("blogKeywords") if isinstance(doc.get("blogKeywords"), list) else [],
        "questionGeneration": _normalize_question_generation_settings(doc.get("questionGeneration")),
        "competitors": doc.get("competitors") if isinstance(doc.get("competitors"), list) else [],
        "systemCompetitors": doc.get("systemCompetitors") if isinstance(doc.get("systemCompetitors"), list) else [],
        "trackedPages": doc.get("trackedPages") if isinstance(doc.get("trackedPages"), list) else [],
        "latest_phase1_score": doc.get("latest_phase1_score"),
        "latest_phase5_score": doc.get("latest_phase5_score"),
        "latest_weekly_blog_at": doc.get("latest_weekly_blog_at"),
        "last_manually_refreshed_at": doc.get("last_manually_refreshed_at"),
        "created_at": doc.get("created_at"),
        "updated_at": doc.get("updated_at"),
        "latest_scrape_result": doc.get("latest_scrape_result"),
        "scores_history": doc.get("scores_history") or [],
    }


def _public_tracking_run_doc(doc: dict | None) -> dict | None:
    if not doc:
        return None
    out = dict(doc)
    out["id"] = str(out.pop("_id", ""))
    out["business_id"] = str(out.get("business_id") or "")
    out["competitors"] = out.get("competitors") if isinstance(out.get("competitors"), list) else []
    out["tracked_competitors"] = out.get("tracked_competitors") if isinstance(out.get("tracked_competitors"), list) else []
    out["questions"] = out.get("questions") if isinstance(out.get("questions"), list) else []
    return out


def _build_tracking_fallback_questions(business_doc: dict) -> list[dict]:
    name = str(business_doc.get("businessName") or business_doc.get("normalized_domain") or "this business").strip()
    category = str(business_doc.get("category") or "business").strip()
    location = str(business_doc.get("location") or "local area").strip()
    services = business_doc.get("services") if isinstance(business_doc.get("services"), list) else []
    service_text = ", ".join([str(s).strip() for s in services if str(s).strip()][:3]) or category
    raw_questions = [
        f"What are the best {category} options in {location}?",
        f"Which {category} in {location} is most recommended?",
        f"Who offers {service_text} in {location}?",
        f"Is {name} recommended for {category} in {location}?",
        f"What do reviews say about {name}?",
        f"Which businesses compete with {name} for {category} customers in {location}?",
    ]
    return [{"id": f"track-{idx + 1}", "text": text} for idx, text in enumerate(raw_questions)]


async def _run_competitor_tracking_for_business(
    *,
    business_doc: dict,
    current_user: dict,
) -> dict:
    if competitor_tracking_runs_col is None:
        raise HTTPException(status_code=503, detail="competitor tracking storage unavailable")

    business_id = str(business_doc["_id"])
    url = business_doc.get("url") or business_doc.get("normalized_domain") or ""
    target_domain = _normalize_site(url)
    if not target_domain:
        raise HTTPException(status_code=400, detail="Business URL is required before tracking competitors")

    tracked_competitors = []
    seen_tracked: set[str] = set()
    for item in business_doc.get("competitors") or []:
        domain = _normalize_site(str(item or ""))
        if domain and domain != target_domain and domain not in seen_tracked:
            seen_tracked.add(domain)
            tracked_competitors.append(domain)
        if len(tracked_competitors) >= 5:
            break

    business_context = {
        "name": business_doc.get("businessName") or business_doc.get("normalized_domain") or target_domain,
        "category": business_doc.get("category") or "",
        "location": business_doc.get("location") or "",
        "description": business_doc.get("aiDescription") or business_doc.get("businessDescription") or "",
        "services": business_doc.get("services") if isinstance(business_doc.get("services"), list) else [],
    }

    questions: list[dict]
    try:
        generated_questions = await generate_brand_questions(url, business_context=business_context)
        questions = [
            {"id": f"track-{idx + 1}", "text": str(text)}
            for idx, text in enumerate((generated_questions or [])[:20])
            if str(text or "").strip()
        ]
    except Exception as e:
        print(f"[CompetitorTracking] question generation fallback business_id={business_id}: {e}")
        questions = _build_tracking_fallback_questions(business_doc)

    started_at = datetime.utcnow().isoformat()
    run_doc = {
        "run_id": uuid.uuid4().hex,
        "business_id": business_id,
        "user_id": current_user["id"],
        "user_email": current_user.get("email"),
        "url": url,
        "target_domain": target_domain,
        "status": "running",
        "tracked_competitors": tracked_competitors,
        "questions": questions,
        "created_at": started_at,
        "updated_at": started_at,
    }
    insert_result = await competitor_tracking_runs_col.insert_one(run_doc)

    try:
        discovered = await generate_deep_competitor_scores(
            url=url,
            questions=questions,
            seed_results={},
        )

        previous = await competitor_tracking_runs_col.find_one(
            {
                "business_id": business_id,
                "user_id": current_user["id"],
                "status": "completed",
                "_id": {"$ne": insert_result.inserted_id},
            },
            sort=[("created_at", -1)],
        )
        previous_scores = {
            _normalize_site(str(item.get("domain") or item.get("url") or "")): int(float(item.get("score") or 0))
            for item in (previous or {}).get("competitors", [])
            if isinstance(item, dict)
        }

        entries: dict[str, dict] = {}
        for item in discovered or []:
            if not isinstance(item, dict):
                continue
            domain = _normalize_site(str(item.get("domain") or item.get("url") or ""))
            if not domain:
                continue
            score = int(max(0, min(100, round(float(item.get("score") or 0)))))
            entries[domain] = {
                "domain": domain,
                "url": item.get("url") or f"https://{domain}/",
                "name": _clean_optional_text(item.get("name")) or domain,
                "score": score,
                "position": item.get("position"),
                "status": "Your site" if domain == target_domain else (_clean_optional_text(item.get("confidence")) or "AI evidence"),
                "evidence": _clean_optional_text(item.get("evidence")) or "",
                "isUser": domain == target_domain,
                "isTracked": domain in tracked_competitors,
            }

        for domain in tracked_competitors:
            if domain not in entries:
                entries[domain] = {
                    "domain": domain,
                    "url": f"https://{domain}/",
                    "name": domain.split(".")[0].replace("-", " ").title(),
                    "score": 0,
                    "position": None,
                    "status": "Not found this run",
                    "evidence": "This saved competitor was not returned by the latest AI competitor intelligence run.",
                    "isUser": False,
                    "isTracked": True,
                }
            else:
                entries[domain]["isTracked"] = True

        if target_domain not in entries:
            entries[target_domain] = {
                "domain": target_domain,
                "url": f"https://{target_domain}/",
                "name": business_doc.get("businessName") or target_domain,
                "score": int(float(business_doc.get("latest_phase5_score") or business_doc.get("latest_phase1_score") or 0)),
                "position": None,
                "status": "Your site",
                "evidence": "Your current saved visibility score was used because this tracking run did not return a target score.",
                "isUser": True,
                "isTracked": False,
            }

        competitor_rows = []
        for entry in sorted(entries.values(), key=lambda item: float(item.get("score") or 0), reverse=True):
            domain = entry["domain"]
            previous_score = previous_scores.get(domain)
            entry["weeklyChange"] = None if previous_score is None else int(entry["score"]) - previous_score
            competitor_rows.append(entry)

        completed_at = datetime.utcnow().isoformat()
        await competitor_tracking_runs_col.update_one(
            {"_id": insert_result.inserted_id},
            {
                "$set": {
                    "status": "completed",
                    "competitors": competitor_rows,
                    "completed_at": completed_at,
                    "updated_at": completed_at,
                }
            },
        )

        external_system_competitors = [
            {
                "domain": item["domain"],
                "url": item["url"],
                "name": item["name"],
                "score": item["score"],
                "status": item["status"],
                "evidence": item["evidence"],
            }
            for item in competitor_rows
            if not item.get("isUser") and int(item.get("score") or 0) > 0
        ][:4]
        await _upsert_user_business(
            current_user=current_user,
            url=url,
            business_id=business_id,
            system_competitors=external_system_competitors,
        )
        await businesses_col.update_one(
            {"_id": ObjectId(business_id), "user_id": current_user["id"]},
            {
                "$set": {
                    "latest_competitor_tracking_at": completed_at,
                    "latest_competitor_tracking_run_id": str(insert_result.inserted_id),
                    "updated_at": completed_at,
                }
            },
        )

        saved = await competitor_tracking_runs_col.find_one({"_id": insert_result.inserted_id})
        return _public_tracking_run_doc(saved)
    except Exception as e:
        failed_at = datetime.utcnow().isoformat()
        await competitor_tracking_runs_col.update_one(
            {"_id": insert_result.inserted_id},
            {
                "$set": {
                    "status": "failed",
                    "error": str(e),
                    "updated_at": failed_at,
                }
            },
        )
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Competitor tracking failed: {str(e)}")
