# Generated from the former backend/main.py lines 2113-2551.
@app.post("/api/ai-insights", response_model=AiInsightsResult)
async def api_ai_insights(
    request: AiInsightsRequest,
    http_request: Request,
    current_user: dict = Depends(get_current_user_optional),
):
    try:
        await _enforce_public_child_call(
            request=http_request,
            current_user=current_user,
            endpoint_key="ai_insights",
            max_calls=1,
        )
        started_at = datetime.utcnow()
        print(f"[API] /api/ai-insights started: {request.url}")
        insights = await asyncio.wait_for(get_ai_insights_multi(request.businessName, request.url), timeout=120)
        if not isinstance(insights, list):
            insights = []

        for insight in insights:
            model_name = insight.get("modelName") if isinstance(insight, dict) else None
            lowered_model = str(model_name or "").lower()
            if any(k in lowered_model for k in ["gpt", "o1", "o3", "o4"]):
                provider_hint = "openai"
            elif any(k in lowered_model for k in ["pplx", "sonar", "perplexity"]):
                provider_hint = "perplexity"
            elif "claude" in lowered_model:
                provider_hint = "anthropic"
            else:
                provider_hint = "gemini"
            await _log_ai_usage_event({
                "feature": "phase1_ai_insights",
                "endpoint": "/api/ai-insights",
                "url": request.url,
                "user_id": current_user.get("id") if current_user else None,
                "user_email": current_user.get("email") if current_user else None,
                "model_name": model_name,
                "model_provider": provider_hint,
                "ai_calls_estimate": 1,
                "details": {
                    "isKnown": bool(insight.get("isKnown", False)) if isinstance(insight, dict) else None,
                    "platform_count": len(insight.get("platforms", [])) if isinstance(insight, dict) and isinstance(insight.get("platforms"), list) else 0,
                },
            })
        elapsed = (datetime.utcnow() - started_at).total_seconds()
        print(f"[API] /api/ai-insights completed in {elapsed:.1f}s: {request.url}")
        return AiInsightsResult(success=True, insights=insights)
    except asyncio.TimeoutError:
        print(f"[API] /api/ai-insights timeout: {request.url}")
        return AiInsightsResult(success=False, insights=[], error="AI insights timed out. Please retry.")
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        return AiInsightsResult(success=False, insights=[], error=str(e))


def _build_public_competitor_questions(request: PublicCompetitorsRequest) -> list[dict]:
    category = (request.category or "business").strip()
    location = (request.location or "nearby").strip()
    name = (request.businessName or "").strip()
    description = (request.description or "").strip()
    context_hint = f"{category} in {location}".strip()

    raw_questions = [
        f"Best {context_hint}?",
        f"Top rated {context_hint}?",
        f"Where should I book a {category} in {location}?",
        f"Recommended {category} near {location}?",
        f"Popular {category} for customers in {location}?",
        f"Which {category} has the best reviews in {location}?",
        f"Alternatives to {name}?" if name else f"Best alternatives for a {category} in {location}?",
        f"{description[:90]} competitors?" if description else f"Trusted {category} options in {location}?",
    ]

    questions = []
    seen = set()
    for text in raw_questions:
        cleaned = re.sub(r"\s+", " ", text).strip()
        key = cleaned.lower().rstrip("?.!")
        if len(cleaned) < 8 or key in seen:
            continue
        seen.add(key)
        questions.append({"id": f"public_competitor_{len(questions) + 1}", "text": cleaned if cleaned.endswith("?") else f"{cleaned}?"})
    return questions[:8]


@app.post("/api/public/competitors", response_model=PublicCompetitorsResponse)
async def api_public_competitors(
    request: PublicCompetitorsRequest,
    http_request: Request,
    current_user: dict = Depends(get_current_user_optional),
):
    try:
        await _enforce_public_child_call(
            request=http_request,
            current_user=current_user,
            endpoint_key="competitors",
            max_calls=3,
        )
        target_domain = _normalize_domain(request.url)
        questions = _build_public_competitor_questions(request)
        print(f"[API] /api/public/competitors started: {request.url} questions={len(questions)}")

        public_competitor_timeout_seconds = 300
        raw_competitors = await asyncio.wait_for(
            generate_public_competitor_suggestions(
                url=request.url,
                questions=questions,
                business_name=request.businessName or "",
                category=request.category or "",
                location=request.location or "",
                description=request.description or "",
                desired_count=4,
            ),
            timeout=public_competitor_timeout_seconds,
        )

        competitors: list[dict] = []
        seen = set()
        for item in raw_competitors if isinstance(raw_competitors, list) else []:
            if not isinstance(item, dict):
                continue
            domain = _normalize_domain(str(item.get("domain") or ""))
            evidence = str(item.get("evidence") or "").strip()
            confidence = str(item.get("confidence") or "").strip().lower()
            if (
                not domain
                or domain == target_domain
                or domain in seen
                or confidence == "generated"
                or "auto-generated fallback" in evidence.lower()
            ):
                continue
            seen.add(domain)
            competitors.append({
                "domain": domain,
                "position": item.get("position"),
                "score": item.get("score"),
                "evidence": evidence,
                "confidence": confidence or "validated",
                "faviconUrl": f"https://www.google.com/s2/favicons?domain={domain}&sz=64",
            })
            if len(competitors) >= 4:
                break

        await _log_ai_usage_event({
            "feature": "public_competitor_lookup",
            "endpoint": "/api/public/competitors",
            "url": request.url,
            "user_id": current_user.get("id") if current_user else None,
            "user_email": current_user.get("email") if current_user else None,
            "model_provider": "claude",
            "ai_calls_estimate": 1,
            "details": {
                "question_count": len(questions),
                "competitor_count": len(competitors),
            },
        })

        print(f"[API] /api/public/competitors completed: {request.url} competitors={len(competitors)}")
        return PublicCompetitorsResponse(success=True, competitors=competitors)
    except asyncio.TimeoutError:
        print(f"[API] /api/public/competitors timeout: {request.url}")
        return PublicCompetitorsResponse(success=False, competitors=[], error="Competitor lookup took too long. Please retry or add domains manually.")
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        return PublicCompetitorsResponse(success=False, competitors=[], error=str(e))


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


@app.post("/api/content/page-generator", response_model=ContentPageGeneratorResponse)
async def api_content_page_generator(
    request: ContentPageGeneratorRequest,
    current_user: dict = Depends(get_current_user),
):
    if not request.url:
        raise HTTPException(status_code=400, detail="A business URL is required")

    saved_business = None
    if request.business_id:
        if businesses_col is None:
            raise HTTPException(status_code=503, detail="business storage unavailable")
        try:
            saved_business = await businesses_col.find_one({
                "_id": ObjectId(request.business_id),
                "user_id": current_user["id"],
            })
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid business ID format")
        if not saved_business:
            raise HTTPException(status_code=404, detail="Business profile not found")

    latest_scan = request.scanData or {}
    if saved_business and isinstance(saved_business.get("latest_scrape_result"), dict):
        latest_scan = {**saved_business.get("latest_scrape_result", {}), **latest_scan}

    business_context = {
        "business_id": request.business_id,
        "url": request.url or (saved_business or {}).get("url"),
        "businessName": request.businessName or (saved_business or {}).get("businessName") or latest_scan.get("businessName"),
        "category": request.category or (saved_business or {}).get("category"),
        "location": request.location or (saved_business or {}).get("location"),
        "businessDescription": request.businessDescription or (saved_business or {}).get("businessDescription"),
        "aiDescription": request.aiDescription or (saved_business or {}).get("aiDescription"),
        "services": request.services or (saved_business or {}).get("services") or [],
        "targetAudience": request.targetAudience or (saved_business or {}).get("targetAudience"),
        "competitors": request.competitors or (saved_business or {}).get("competitors") or [],
        "scanData": latest_scan,
        "promptContext": request.promptContext or {},
    }

    model = (request.model or "claude").strip().lower()
    if model not in {"claude", "chatgpt", "openai"}:
        raise HTTPException(status_code=400, detail="Use claude or chatgpt for page generation")

    try:
        generated = await generate_content_page(
            business_context=business_context,
            selected_model="chatgpt" if model == "openai" else model,
        )
    except Exception as e:
        print(f"[API] content page generation error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=502, detail="Could not generate content page right now")

    generated_at = datetime.utcnow().isoformat()
    doc = {
        "user_id": current_user["id"],
        "user_email": current_user.get("email"),
        "business_id": request.business_id,
        "url": business_context["url"],
        "business_context": business_context,
        "result": generated,
        "created_at": generated_at,
        "updated_at": generated_at,
    }
    if generated_content_pages_col is not None:
        try:
            await generated_content_pages_col.insert_one(doc)
        except Exception as e:
            print(f"[API] failed to save generated content page: {e}")

    return ContentPageGeneratorResponse(
        success=True,
        pageTitle=str(generated.get("pageTitle") or ""),
        metaTitle=str(generated.get("metaTitle") or ""),
        metaDescription=str(generated.get("metaDescription") or ""),
        sections=generated.get("sections") or [],
        faqs=generated.get("faqs") or [],
        factsUsed=generated.get("factsUsed") or [],
        warnings=generated.get("warnings") or [],
        outputText=str(generated.get("outputText") or ""),
        modelUsed=generated.get("modelUsed"),
        provider=generated.get("provider"),
        generatedAt=generated_at,
    )


@app.post("/api/blogs/analyze", response_model=BlogAnalysisResponse)
async def api_blogs_analyze(request: BlogAnalyzeRequest, current_user: dict = Depends(get_current_user_optional)):
    text = str(request.text or "").strip()
    if not text:
        return BlogAnalysisResponse(success=False, error="Blog text is required.")

    model = (request.model or "perplexity").strip().lower()
    if model != "perplexity":
        raise HTTPException(status_code=400, detail="Only Perplexity is supported for now.")

    base = _build_blog_base_analysis(text, max(0, int(request.attachment_count or 0)))
    metrics = base.get("metrics", {})

    try:
        llm = await get_blog_analysis_perplexity(text, metrics)
    except Exception as e:
        print(f"[API] blog analysis error: {e}")
        llm = {}

    strengths = llm.get("strengths") if isinstance(llm.get("strengths"), list) else base["strengths"]
    suggestions = llm.get("suggestions") if isinstance(llm.get("suggestions"), list) else base["suggestions"]
    weak_spots = llm.get("weakSpots") if isinstance(llm.get("weakSpots"), list) else [
        s for s in suggestions if "Expand" in s or "H2 and H3" in s or "Split long walls" in s
    ]
    improvements = llm.get("improvements") if isinstance(llm.get("improvements"), list) else [
        s for s in suggestions if s not in weak_spots
    ]
    overview = str(llm.get("overview") or base["overview"])
    summary = str(llm.get("summary") or base["summary"])

    result = BlogAnalysis(
        score=int(metrics.get("score", 0)),
        grade=str(metrics.get("grade", "Needs Work")),
        overview=overview,
        summary=summary,
        wordCount=int(metrics.get("wordCount", 0)),
        paragraphCount=int(metrics.get("paragraphCount", 0)),
        headingCount=int(metrics.get("headingCount", 0)),
        listCount=int(metrics.get("listCount", 0)),
        linkCount=int(metrics.get("linkCount", 0)),
        ctaCount=int(metrics.get("ctaCount", 0)),
        readability=int(metrics.get("readability", 0)),
        structure=int(metrics.get("structure", 0)),
        seo=int(metrics.get("seo", 0)),
        engagement=int(metrics.get("engagement", 0)),
        strengths=[str(s) for s in strengths if str(s).strip()],
        weakSpots=[str(s) for s in weak_spots if str(s).strip()],
        improvements=[str(s) for s in improvements if str(s).strip()],
        suggestions=[str(s) for s in suggestions if str(s).strip()],
    )

    try:
        await _log_ai_usage_event({
            "feature": "blog_seo_analyzer",
            "endpoint": "/api/blogs/analyze",
            "user_id": current_user.get("id") if current_user else None,
            "user_email": current_user.get("email") if current_user else None,
            "model_name": llm.get("modelUsed") if isinstance(llm, dict) else None,
            "model_provider": "perplexity",
            "ai_calls_estimate": 1,
            "details": {
                "word_count": result.wordCount,
                "attachment_count": int(metrics.get("attachmentCount", 0)),
            },
        })
    except Exception as e:
        print(f"[API] blog usage logging failed: {e}")

    return BlogAnalysisResponse(success=True, result=result)


BLOG_WEEKLY_LIMIT = 2


def _blog_week_window(now: datetime | None = None) -> tuple[datetime, datetime]:
    current = now or datetime.utcnow()
    start = current - timedelta(days=current.weekday())
    start = start.replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + timedelta(days=7)
    return start, end


async def _blog_generation_usage(current_user: dict) -> dict:
    start, end = _blog_week_window()
    used = 0
    if ai_usage_col is not None:
        used = await ai_usage_col.count_documents({
            "user_id": current_user["id"],
            "feature": "blog_seo_generator",
            "timestamp": {"$gte": start, "$lt": end},
        })
    remaining = max(0, BLOG_WEEKLY_LIMIT - int(used or 0))
    return {
        "limit": BLOG_WEEKLY_LIMIT,
        "used": int(used or 0),
        "remaining": remaining,
        "periodStart": start.isoformat(),
        "periodEnd": end.isoformat(),
    }


def _clean_blog_list(values: list[str] | None, max_items: int = 8) -> list[str]:
    if not isinstance(values, list):
        return []
    cleaned = []
    seen = set()
    for value in values:
        text = str(value or "").strip()
        if text:
            lower_text = text.lower()
            if lower_text not in seen:
                seen.add(lower_text)
                cleaned.append(text[:220])
        if len(cleaned) >= max_items:
            break
    return cleaned


def _blog_week_id(now: datetime | None = None) -> str:
    start, _ = _blog_week_window(now)
    return start.strftime("%G-W%V")


def _blog_text_from_generated(result: dict) -> str:
    parts = [str(result.get("title") or ""), str(result.get("excerpt") or "")]
    for section in result.get("sections") or []:
        if isinstance(section, dict):
            parts.append(str(section.get("heading") or ""))
            parts.append(str(section.get("content") or ""))
    return "\n\n".join(parts)


def _humanized_blog_score(result: dict, voice: str | None = None) -> int:
    text = _blog_text_from_generated(result)
    words = [w for w in re.split(r"\s+", text or "") if w.strip()]
    sentences = _blog_split_sentences(text)
    avg_sentence = (len(words) / len(sentences)) if sentences else 0
    score = 82
    if voice and len(voice.strip()) > 20:
        score += 8
    if 10 <= avg_sentence <= 22:
        score += 5
    elif avg_sentence > 30:
        score -= 10
    ai_phrases = [
        "in today's digital landscape",
        "look no further",
        "game-changer",
        "unlock the power",
        "comprehensive guide",
        "it is important to note",
    ]
    lowered = text.lower()
    score -= sum(6 for phrase in ai_phrases if phrase in lowered)
    score += min(5, len(set(w.lower().strip(".,!?;:") for w in words)) // 180)
    return _blog_clamp(score, 35, 98)
