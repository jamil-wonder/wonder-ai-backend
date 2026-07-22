# Generated from the former backend/main.py lines 3825-4190.
async def _phase5_worker_loop():
    while True:
        try:
            claimed = await phase5_jobs_col.find_one_and_update(
                {"status": "queued"},
                {
                    "$set": {
                        "status": "running",
                        "worker_id": PHASE5_WORKER_ID,
                        "updated_at": datetime.utcnow().isoformat(),
                    }
                },
                sort=[("queue_priority", 1), ("created_at", 1)],
                return_document=ReturnDocument.AFTER,
            )

            if not claimed:
                await asyncio.sleep(PHASE5_WORKER_POLL_INTERVAL)
                continue

            await _process_phase5_job(claimed)
        except asyncio.CancelledError:
            break
        except Exception:
            traceback.print_exc()
            await asyncio.sleep(PHASE5_WORKER_POLL_INTERVAL)


async def _phase5_kick_job_processing(job_id: str):
    """Best-effort direct claim path so newly queued jobs are not stranded."""
    if phase5_jobs_col is None:
        return
    try:
        claimed = await phase5_jobs_col.find_one_and_update(
            {"job_id": job_id, "status": "queued"},
            {
                "$set": {
                    "status": "running",
                    "worker_id": PHASE5_WORKER_ID,
                    "updated_at": datetime.utcnow().isoformat(),
                }
            },
            return_document=ReturnDocument.AFTER,
        )
        if not claimed:
            return
        print(f"[Phase5] kick start job_id={job_id} provider={claimed.get('model')}")
        await _process_phase5_job(claimed)
    except Exception:
        traceback.print_exc()


async def _phase5_try_start_immediately(job_id: str) -> None:
    """Try to claim a newly queued job right away and process in background."""
    if phase5_jobs_col is None:
        return
    try:
        claimed = await phase5_jobs_col.find_one_and_update(
            {"job_id": job_id, "status": "queued"},
            {
                "$set": {
                    "status": "running",
                    "worker_id": PHASE5_WORKER_ID,
                    "updated_at": datetime.utcnow().isoformat(),
                }
            },
            return_document=ReturnDocument.AFTER,
        )
        if not claimed:
            return
        print(f"[Phase5] immediate start job_id={job_id} provider={claimed.get('model')}")
        asyncio.create_task(_process_phase5_job(claimed))
    except Exception:
        traceback.print_exc()


def seconds_until_next_sunday_4am() -> float:
    now = datetime.now()
    days_until_sunday = 6 - now.weekday()
    if days_until_sunday == 0:
        if now.hour >= 4:
            days_until_sunday = 7
            
    target = datetime(
        now.year, now.month, now.day, 4, 0, 0, 0
    ) + timedelta(days=days_until_sunday)
    
    diff = target - now
    return max(1.0, diff.total_seconds())


async def sunday_analyzer_scheduler():
    print("[Scheduler] Weekly Sunday 4:00 AM analyzer task started")
    await asyncio.sleep(15)
    while True:
        sleep_sec = seconds_until_next_sunday_4am()
        hours_val = round(sleep_sec / 3600.0, 2)
        print(f"[Scheduler] Sleeping for {sleep_sec} seconds (approx {hours_val} hours) until next Sunday 4:00 AM")
        await asyncio.sleep(sleep_sec)
        
        print("[Scheduler] It is Sunday 4:00 AM. Starting sequential Phase 1 crawl queue...")
        try:
            if businesses_col is not None:
                cursor = businesses_col.find({})
                all_businesses = await cursor.to_list(length=10000)
                print(f"[Scheduler] Found {len(all_businesses)} businesses to crawl.")
                
                for biz_doc in all_businesses:
                    url = biz_doc.get("url")
                    user_id = biz_doc.get("user_id")
                    business_id = str(biz_doc.get("_id"))
                    
                    if not url or not user_id:
                        continue
                        
                    user_mock = {
                        "id": user_id,
                        "email": biz_doc.get("user_email") or ""
                    }
                    
                    print(f"[Scheduler] Running auto scrape for {url} (user: {user_id}, business: {business_id})...")
                    try:
                        result = await asyncio.wait_for(
                            asyncio.to_thread(_run_scrape_worker, url),
                            timeout=420
                        )
                        if isinstance(result, dict):
                            await _upsert_user_business(
                                current_user=user_mock,
                                url=url,
                                category=biz_doc.get("category"),
                                location=biz_doc.get("location"),
                                business_name=result.get("businessName"),
                                logo_url=result.get("logoUrl"),
                                phase1_score=((result.get("scores") or {}).get("total") if isinstance(result.get("scores"), dict) else None),
                                business_id=business_id,
                                scrape_result=result,
                            )
                            print(f"[Scheduler] Completed auto scrape for {url}")
                            try:
                                await _build_weekly_blogs_for_business(
                                    business_doc={**biz_doc, "businessName": result.get("businessName") or biz_doc.get("businessName"), "latest_scrape_result": result},
                                    current_user=user_mock,
                                    force=False,
                                )
                                print(f"[Scheduler] Weekly blogs ready for {url}")
                            except Exception as blog_err:
                                print(f"[Scheduler] Weekly blog generation failed for {url}: {blog_err}")
                    except Exception as scrape_err:
                        print(f"[Scheduler] Auto scrape failed for {url}: {scrape_err}")
                        
                    await asyncio.sleep(5)
            else:
                print("[Scheduler] businesses_col is None, skipping cron crawl")
        except Exception as run_err:
            print(f"[Scheduler] Exception in auto crawler queue execution: {run_err}")


@app.on_event("startup")
async def _phase5_worker_startup():
    asyncio.create_task(sunday_analyzer_scheduler())

    if phase5_jobs_col is None:
        app.state.phase5_worker_tasks = []
        return

    loop = asyncio.get_running_loop()
    app.state.phase5_executor = ThreadPoolExecutor(max_workers=max(4, PHASE5_MODEL_MAX_THREADS))
    loop.set_default_executor(app.state.phase5_executor)

    # Indexes tuned to current Phase 5 query/update patterns.
    try:
        # Direct lookups
        await phase5_jobs_col.create_index("job_id", unique=True)

        # Worker claim path: find queued jobs sorted by priority and creation time.
        await phase5_jobs_col.create_index([
            ("status", 1),
            ("queue_priority", 1),
            ("created_at", 1),
        ])

        # User history and trend listing paths.
        await phase5_jobs_col.create_index([
            ("user_id", 1),
            ("created_at", -1),
        ])
        await phase5_jobs_col.create_index([
            ("user_id", 1),
            ("status", 1),
            ("created_at", -1),
        ])

        # Stale job sweeps and startup recovery updates.
        await phase5_jobs_col.create_index([
            ("status", 1),
            ("updated_at", 1),
        ])

        # Related collections used by history endpoints.
        if urls_col is not None:
            await urls_col.create_index([
                ("user_id", 1),
                ("timestamp", -1),
            ])
        if user_history_meta_col is not None:
            await user_history_meta_col.create_index("user_id", unique=True)
        if public_rate_limits_col is not None:
            await public_rate_limits_col.create_index("key", unique=True)
            await public_rate_limits_col.create_index("reset_at")
        if competitor_tracking_runs_col is not None:
            await competitor_tracking_runs_col.create_index([
                ("business_id", 1),
                ("user_id", 1),
                ("created_at", -1),
            ])
            await competitor_tracking_runs_col.create_index([
                ("business_id", 1),
                ("status", 1),
            ])
        if weekly_blog_suggestions_col is not None:
            await weekly_blog_suggestions_col.create_index([
                ("business_id", 1),
                ("user_id", 1),
                ("week_id", 1),
            ], unique=True)
            await weekly_blog_suggestions_col.create_index([
                ("user_id", 1),
                ("created_at", -1),
            ])
        if auth_handoffs_col is not None:
            await auth_handoffs_col.create_index("code_hash", unique=True)
            await auth_handoffs_col.create_index("expires_at", expireAfterSeconds=0)
        if google_integrations_col is not None:
            await google_integrations_col.create_index("user_id", unique=True)
        if analytics_snapshots_col is not None:
            await analytics_snapshots_col.create_index([
                ("user_id", 1),
                ("business_id", 1),
                ("created_at", -1),
            ])
    except Exception:
        print("[Phase5] warning: index creation failed; continuing without blocking startup")
        traceback.print_exc()

    # Cost-safety default: do not auto-resume previously queued/in-progress jobs after restart
    # unless explicitly enabled via env.
    if not PHASE5_RESUME_QUEUED_ON_STARTUP:
        try:
            startup_failed = await phase5_jobs_col.update_many(
                {"status": {"$in": ["queued", "running", "finalizing"]}},
                {
                    "$set": {
                        "status": "failed",
                        "worker_id": None,
                        "current_question_id": None,
                        "error": "startup_queue_reset",
                        "updated_at": datetime.utcnow().isoformat(),
                    }
                },
            )
            if int(startup_failed.modified_count or 0) > 0:
                print(f"[Phase5] startup reset previous queued/running jobs={startup_failed.modified_count}")
        except Exception:
            traceback.print_exc()

    # Hard safety gate: while Gemini is disabled for Phase 5, fail stale Gemini jobs on startup.
    if not PHASE5_ENABLE_GEMINI:
        try:
            startup_gemini_failed = await phase5_jobs_col.update_many(
                {
                    "model": "gemini",
                    "status": {"$in": ["queued", "running", "finalizing"]},
                },
                {
                    "$set": {
                        "status": "failed",
                        "worker_id": None,
                        "current_question_id": None,
                        "error": "gemini_disabled_phase5",
                        "updated_at": datetime.utcnow().isoformat(),
                    }
                },
            )
            if int(startup_gemini_failed.modified_count or 0) > 0:
                print(f"[Phase5] startup failed stale gemini jobs={startup_gemini_failed.modified_count}")
        except Exception:
            traceback.print_exc()

    stale_running_cutoff_iso = (datetime.utcnow() - timedelta(seconds=max(30, PHASE5_STALE_RUNNING_SECONDS))).isoformat()
    stale_queued_cutoff_iso = (datetime.utcnow() - timedelta(seconds=max(120, PHASE5_STALE_QUEUED_SECONDS))).isoformat()
    try:
        if PHASE5_RECOVER_STALE_RUNNING:
            stale_reset = await phase5_jobs_col.update_many(
                {"status": "running", "updated_at": {"$lt": stale_running_cutoff_iso}},
                {
                    "$set": {
                        "status": "queued",
                        "worker_id": None,
                        "current_question_id": None,
                        "updated_at": datetime.utcnow().isoformat(),
                    }
                },
            )
            if int(stale_reset.modified_count or 0) > 0:
                print(f"[Phase5] recovered stale running jobs={stale_reset.modified_count}")
        else:
            stale_failed = await phase5_jobs_col.update_many(
                {"status": "running", "updated_at": {"$lt": stale_running_cutoff_iso}},
                {
                    "$set": {
                        "status": "failed",
                        "worker_id": None,
                        "current_question_id": None,
                        "error": "stale_running_job_recovered_as_failed",
                        "updated_at": datetime.utcnow().isoformat(),
                    }
                },
            )
            if int(stale_failed.modified_count or 0) > 0:
                print(f"[Phase5] marked stale running jobs as failed={stale_failed.modified_count}")

        stale_queued_failed = await phase5_jobs_col.update_many(
            {"status": "queued", "updated_at": {"$lt": stale_queued_cutoff_iso}},
            {
                "$set": {
                    "status": "failed",
                    "error": "stale_queued_job_recovered_as_failed",
                    "updated_at": datetime.utcnow().isoformat(),
                }
            },
        )
        if int(stale_queued_failed.modified_count or 0) > 0:
            print(f"[Phase5] marked stale queued jobs as failed={stale_queued_failed.modified_count}")
    except Exception:
        traceback.print_exc()

    print(
        f"[Phase5] startup workers={max(1, PHASE5_WORKER_CONCURRENCY)} "
        f"parallelism={PHASE5_JOB_PARALLELISM} gemini_enabled={PHASE5_ENABLE_GEMINI} "
        f"timeout_openai={PHASE5_QUESTION_TIMEOUT_OPENAI_SEC}s "
        f"timeout_perplexity={PHASE5_QUESTION_TIMEOUT_PERPLEXITY_SEC}s "
        f"timeout_anthropic={PHASE5_QUESTION_TIMEOUT_ANTHROPIC_SEC}s"
    )
    print(
        f"[Phase5] providers openai_model={(os.getenv('OPENAI_MODEL_PHASE5') or 'unset').strip() or 'unset'} "
        f"perplexity_model={(os.getenv('PERPLEXITY_MODEL_PHASE5') or 'sonar-pro').strip() or 'sonar-pro'} "
        f"anthropic_model={(os.getenv('ANTHROPIC_MODEL_PHASE5') or 'claude-sonnet-4-5').strip() or 'claude-sonnet-4-5'}"
    )
    app.state.phase5_worker_tasks = [
        asyncio.create_task(_phase5_worker_loop())
        for _ in range(max(1, PHASE5_WORKER_CONCURRENCY))
    ]


@app.on_event("shutdown")
async def _phase5_worker_shutdown():
    tasks = getattr(app.state, "phase5_worker_tasks", [])
    for t in tasks:
        t.cancel()
    for t in tasks:
        try:
            await t
        except asyncio.CancelledError:
            pass
        except Exception:
            pass

    executor = getattr(app.state, "phase5_executor", None)
    if executor is not None:
        try:
            executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
