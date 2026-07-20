# Generated from the former backend/main.py lines 4193-4516.
@app.post("/api/phase5/start-job", response_model=Phase5StartJobResponse)
async def api_phase5_start_job(req: Phase5StartJobRequest, current_user: dict = Depends(get_current_user_optional)):
    try:
        if phase5_jobs_col is None:
            raise HTTPException(status_code=503, detail="phase5 job storage unavailable")
        questions_dicts = [q.model_dump() for q in req.questions]
        if not questions_dicts:
            raise HTTPException(status_code=400, detail="questions cannot be empty")
        model_provider = "multi"

        job_id = uuid.uuid4().hex
        now_iso = datetime.utcnow().isoformat() + "Z" 
        business = await _upsert_user_business(
            current_user=current_user,
            url=req.url,
            business_id=getattr(req, "business_id", None),
        )
        public_business = _public_business_doc(business)
        job_doc = {
            "job_id": job_id,
            "job_type": "core",
            "model": model_provider,
            "queue_priority": 0,
            "url": req.url,
            "user_id": current_user.get("id") if current_user else None,
            "user_email": current_user.get("email") if current_user else None,
            "business_id": public_business["id"] if public_business else None,
            "questions": questions_dicts,
            "seed_results": {},
            "status": "queued",
            "total": len(questions_dicts),
            "processed": 0,
            "current_question_id": None,
            "results": {},
            "deep_competitors": [],
            "brand_summary": None,
            "error": None,
            "created_at": now_iso,
            "updated_at": now_iso,
        }
        await phase5_jobs_col.insert_one(job_doc)

        print(
            f"[Phase5] queued job_id={job_id} provider={model_provider} "
            f"type=core questions={len(questions_dicts)}"
        )
        await phase5_jobs_col.update_one(
            {"job_id": job_id},
            {
                "$set": {
                    "status": "running",
                    "worker_id": PHASE5_WORKER_ID,
                    "updated_at": datetime.utcnow().isoformat(),
                }
            },
        )
        print(f"[Phase5] immediate start job_id={job_id} provider={model_provider}")
        running_doc = {**job_doc, "status": "running", "worker_id": PHASE5_WORKER_ID}
        asyncio.create_task(_process_phase5_job(running_doc))

        await _log_ai_usage_event({
            "feature": "phase5_job_started",
            "endpoint": "/api/phase5/start-job",
            "url": req.url,
            "user_id": current_user.get("id") if current_user else None,
            "user_email": current_user.get("email") if current_user else None,
            "model_provider": model_provider,
            "ai_calls_estimate": 0,
            "details": {
                "job_id": job_id,
                "job_type": "core",
                "model_provider": model_provider,
                "providers": ["perplexity", "chatgpt", "claude"],
                "question_count": len(questions_dicts),
            },
        })

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


@app.post("/api/phase5/start-deep-job", response_model=Phase5StartJobResponse)
async def api_phase5_start_deep_job(req: Phase5StartJobRequest, current_user: dict = Depends(get_current_user_optional)):
    try:
        if phase5_jobs_col is None:
            raise HTTPException(status_code=503, detail="phase5 job storage unavailable")
        questions_dicts = [q.model_dump() for q in req.questions]
        if not questions_dicts:
            raise HTTPException(status_code=400, detail="questions cannot be empty")
        model_provider = str(req.model or "perplexity").strip().lower()
        if model_provider == "gemini" and not PHASE5_ENABLE_GEMINI:
            raise HTTPException(status_code=400, detail="Gemini is temporarily disabled for Phase 5")
        if model_provider not in {"openai", "perplexity", "claude", "gemini"}:
            raise HTTPException(status_code=400, detail="unsupported model provider")

        job_id = uuid.uuid4().hex
        now_iso = datetime.utcnow().isoformat() + "Z" 
        business = await _upsert_user_business(
            current_user=current_user,
            url=req.url,
            business_id=getattr(req, "business_id", None),
        )
        public_business = _public_business_doc(business)
        job_doc = {
            "job_id": job_id,
            "job_type": "deep",
            "model": model_provider,
            "queue_priority": 0 if model_provider in {"openai", "perplexity"} else 1,
            "url": req.url,
            "user_id": current_user.get("id") if current_user else None,
            "user_email": current_user.get("email") if current_user else None,
            "business_id": public_business["id"] if public_business else None,
            "questions": questions_dicts,
            "seed_results": req.seed_results or {},
            "status": "queued",
            "total": len(questions_dicts),
            "processed": 0,
            "current_question_id": None,
            "results": {},
            "deep_competitors": [],
            "brand_summary": None,
            "error": None,
            "created_at": now_iso,
            "updated_at": now_iso,
        }
        await phase5_jobs_col.insert_one(job_doc)

        print(
            f"[Phase5] queued job_id={job_id} provider={model_provider} "
            f"type=deep questions={len(questions_dicts)}"
        )
        await phase5_jobs_col.update_one(
            {"job_id": job_id},
            {
                "$set": {
                    "status": "running",
                    "worker_id": PHASE5_WORKER_ID,
                    "updated_at": datetime.utcnow().isoformat(),
                }
            },
        )
        print(f"[Phase5] immediate start job_id={job_id} provider={model_provider}")
        running_doc = {**job_doc, "status": "running", "worker_id": PHASE5_WORKER_ID}
        asyncio.create_task(_process_phase5_job(running_doc))

        await _log_ai_usage_event({
            "feature": "phase5_deep_job_started",
            "endpoint": "/api/phase5/start-deep-job",
            "url": req.url,
            "user_id": current_user.get("id") if current_user else None,
            "user_email": current_user.get("email") if current_user else None,
            "model_provider": model_provider,
            "ai_calls_estimate": 0,
            "details": {
                "job_id": job_id,
                "job_type": "deep",
                "model_provider": model_provider,
                "question_count": len(questions_dicts),
            },
        })

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
        "deep_competitors": job.get("deep_competitors", []),
        "brand_summary": job.get("brand_summary"),
        "error": job.get("error"),
    }


@app.get("/api/phase5/job-stream/{job_id}")
async def api_phase5_job_stream(job_id: str, request: Request):
    """Stream Phase 5 job progress via Server-Sent Events."""
    if phase5_jobs_col is None:
        raise HTTPException(status_code=503, detail="phase5 job storage unavailable")

    async def _event_generator():
        last_processed = -1
        last_status = ""
        last_deep = None
        last_brand = None
        idle_ticks = 0
        heartbeat_ticks = 0

        while True:
            if await request.is_disconnected():
                return

            job = await phase5_jobs_col.find_one(
                {"job_id": job_id},
                {
                    "_id": 0,
                    "job_id": 1,
                    "status": 1,
                    "processed": 1,
                    "total": 1,
                    "current_question_id": 1,
                    "results": 1,
                    "deep_competitors": 1,
                    "brand_summary": 1,
                    "provider_scores": 1,
                    "overall_score": 1,
                    "error": 1,
                },
            )

            if not job:
                yield 'event: error\ndata: {"error":"job not found"}\n\n'
                return

            processed = int(job.get("processed", 0) or 0)
            status = str(job.get("status", "") or "")

            # Also emit when deep_competitors or brand_summary change so clients
            # reliably receive finalization results even if processed/status
            # didn't change.
            deep_serial = json.dumps(job.get("deep_competitors", []), default=str, sort_keys=True)
            brand_now = str(job.get("brand_summary") or "")
            if processed != last_processed or status != last_status or deep_serial != last_deep or brand_now != last_brand:
                payload = {
                    "job_id": job_id,
                    "status": status,
                    "processed": processed,
                    "total": int(job.get("total", 0) or 0),
                    "current_question_id": job.get("current_question_id"),
                    "results": job.get("results", {}),
                    "provider_scores": job.get("provider_scores", {}),
                    "overall_score": job.get("overall_score"),
                    "brand_summary": job.get("brand_summary"),
                    "deep_competitors": job.get("deep_competitors", []),
                    "error": job.get("error"),
                }
                yield f"data: {json.dumps(payload, default=str)}\n\n"
                last_processed = processed
                last_status = status
                last_deep = deep_serial
                last_brand = brand_now
                idle_ticks = 0
                heartbeat_ticks = 0
            else:
                idle_ticks += 1
                heartbeat_ticks += 1
                if heartbeat_ticks >= 30:
                    # SSE comment heartbeat keeps proxies/load balancers from closing the stream.
                    yield ": keep-alive\n\n"
                    heartbeat_ticks = 0

            if status in PHASE5_TERMINAL_STATUSES:
                return

            if idle_ticks >= 600:
                return

            await asyncio.sleep(0.5)

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


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
