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


async def _run_weekly_business_pipeline(biz_doc: dict, week_id: str) -> None:
    """One business's full weekly pipeline: Phase 1 re-crawl, scan-complete
    email, weekly blogs, and (if it has a locked baseline) the Search
    Tracker re-run + weekly report email. Extracted from the scheduler loop
    so it can run under a concurrency limit instead of strictly one business
    at a time, and so a crash mid-run only loses this one business's
    progress, not the whole week's — see `weekly_run_week_id` below."""
    url = biz_doc.get("url")
    user_id = biz_doc.get("user_id")
    business_id = str(biz_doc.get("_id"))

    if not url or not user_id:
        return

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
        if not isinstance(result, dict):
            return

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

        # AI insights gathered now (right after the scrape, like before) but
        # the actual email is deferred to the very end of this function —
        # one combined send per business per week instead of a separate
        # Analyser email here and a separate Search Tracker email later.
        business_label = result.get("businessName") or biz_doc.get("businessName") or ""
        ai_insights: list = []
        try:
            # One AI-insights pass per business per week (not per visit
            # like the manual dashboard flow) — the scheduler has no other
            # source for this data, so it's worth the bounded weekly cost
            # to include a real model insight instead of leaving that part
            # of the email empty.
            ai_insights = await asyncio.wait_for(
                get_ai_insights_multi(business_label, url),
                timeout=60,
            )
        except Exception as insight_err:
            print(f"[Scheduler] AI insights unavailable for {url}: {insight_err}")

        tracker_email_data: dict | None = None

        try:
            await _build_weekly_blogs_for_business(
                business_doc={**biz_doc, "businessName": result.get("businessName") or biz_doc.get("businessName"), "latest_scrape_result": result},
                current_user=user_mock,
                force=False,
            )
            print(f"[Scheduler] Weekly blogs ready for {url}")
        except Exception as blog_err:
            print(f"[Scheduler] Weekly blog generation failed for {url}: {blog_err}")

        # Weekly Search Tracker re-run (Part 5 of the flow
        # spec): "the analyser runs every Sunday night —
        # all 20 questions, all four models." This was
        # previously missing entirely — the scheduler only
        # ever re-crawled the site, it never re-ran the
        # locked 20 questions Search Tracker shows. Only
        # businesses with a real locked baseline get one;
        # there's nothing stable to re-run otherwise. This
        # phase5 job run is what keeps "Appeared x/20",
        # rank, and the visibility side of the score
        # current — same job pipeline a live Search
        # Tracker run uses, just triggered here instead of
        # by a button click.
        tracked_questions = biz_doc.get("trackedQuestions")
        if biz_doc.get("questionsLocked") and isinstance(tracked_questions, list) and tracked_questions:
            try:
                questions_dicts = [
                    {"id": str(q.get("id")), "text": str(q.get("query") or "").strip()}
                    for q in tracked_questions
                    if isinstance(q, dict) and str(q.get("query") or "").strip()
                ]
                if questions_dicts:
                    # Grabbed BEFORE inserting the new job — this is
                    # "last week" for the email's diff (weekly change,
                    # what-moved lines). Real comparison or none; never
                    # a guessed number.
                    previous_tracker_job = await phase5_jobs_col.find_one(
                        {
                            "business_id": business_id,
                            "job_type": "core",
                            "model": "multi",
                            "status": "completed",
                        },
                        sort=[("created_at", -1)],
                    )
                    tracker_job_id = uuid.uuid4().hex
                    tracker_now_iso = datetime.utcnow().isoformat() + "Z"
                    tracker_job_doc = {
                        "job_id": tracker_job_id,
                        "job_type": "core",
                        "model": "multi",
                        "queue_priority": 0,
                        "url": url,
                        "user_id": user_id,
                        "user_email": biz_doc.get("user_email") or user_mock["email"],
                        "business_id": business_id,
                        "questions": questions_dicts,
                        "seed_results": {},
                        "status": "running",
                        "worker_id": PHASE5_WORKER_ID,
                        "total": len(questions_dicts),
                        "processed": 0,
                        "current_question_id": None,
                        "results": {},
                        "deep_competitors": [],
                        "brand_summary": None,
                        "error": None,
                        "created_at": tracker_now_iso,
                        "updated_at": tracker_now_iso,
                    }
                    await phase5_jobs_col.insert_one(tracker_job_doc)
                    print(
                        f"[Scheduler] Running weekly Search Tracker re-run for {url} "
                        f"({len(questions_dicts)} questions, all 4 models)..."
                    )
                    # 2 questions x 4 models took ~2.2 minutes live-tested;
                    # a real locked-20 run is 10x the question volume, so
                    # this leaves real headroom rather than cutting off a
                    # slow-but-working run overnight.
                    await asyncio.wait_for(_process_phase5_job(tracker_job_doc), timeout=3600)
                    print(f"[Scheduler] Weekly Search Tracker re-run completed for {url}")

                    try:
                        finished_job = await phase5_jobs_col.find_one({"job_id": tracker_job_id})
                        if finished_job and finished_job.get("status") == "completed":
                            current_score = finished_job.get("overall_score")
                            previous_score = (previous_tracker_job or {}).get("overall_score")
                            current_results = finished_job.get("results") or {}
                            previous_results = (previous_tracker_job or {}).get("results") or {}
                            question_text_by_id = {q["id"]: q["text"] for q in questions_dicts}

                            target_domain = _normalize_domain(url)
                            what_moved, top_actions = _compute_weekly_diff(
                                current_results=current_results,
                                previous_results=previous_results,
                                question_text_by_id=question_text_by_id,
                                target_domain=target_domain,
                            )

                            deep_competitors = finished_job.get("deep_competitors") or []
                            competitor_scores = [
                                float(c.get("score")) for c in deep_competitors
                                if isinstance(c, dict) and isinstance(c.get("score"), (int, float))
                            ]
                            competitor_avg = (sum(competitor_scores) / len(competitor_scores)) if competitor_scores else None

                            if isinstance(current_score, (int, float)):
                                # Handed to the single combined send below
                                # instead of emailing separately here — see
                                # tracker_email_data.
                                tracker_email_data = {
                                    "current_score": float(current_score),
                                    "previous_score": float(previous_score) if isinstance(previous_score, (int, float)) else None,
                                    "competitor_avg": competitor_avg,
                                    "what_moved": what_moved,
                                    "top_actions": top_actions,
                                }

                                # Alerts are a SEPARATE send from the weekly
                                # report — "competitor alert only on a real
                                # event," not bundled into the calm weekly
                                # cadence. Only fires when something real
                                # crossed the threshold, so this stays rare
                                # (well under the spec's 2-3/week cap) rather
                                # than firing every run.
                                try:
                                    alerts = _detect_alerts(
                                        current_score=float(current_score),
                                        previous_score=float(previous_score) if isinstance(previous_score, (int, float)) else None,
                                        current_competitors=deep_competitors,
                                        previous_competitors=(previous_tracker_job or {}).get("deep_competitors") or [],
                                    )
                                    if alerts:
                                        alert_user_doc = await users_col.find_one({"_id": ObjectId(user_id)}) if users_col is not None else None
                                        if alert_user_doc and alert_user_doc.get("notify_scan_complete", True):
                                            alert_sent = await send_alert_email(
                                                to_email=alert_user_doc.get("email", "") or biz_doc.get("user_email") or "",
                                                name=alert_user_doc.get("name", ""),
                                                business_name=biz_doc.get("businessName") or "",
                                                domain=target_domain,
                                                alerts=alerts,
                                            )
                                            print(f"[Scheduler] Alert email for {url}: {'sent' if alert_sent else 'not sent (SMTP unavailable)'} ({len(alerts)} alert(s))")
                                except Exception as alert_err:
                                    print(f"[Scheduler] Alert detection/send failed for {url}: {alert_err}")
                    except Exception as weekly_email_err:
                        print(f"[Scheduler] Weekly Search Tracker data build failed for {url}: {weekly_email_err}")
            except Exception as tracker_err:
                print(f"[Scheduler] Weekly Search Tracker re-run failed for {url}: {tracker_err}")

        try:
            user_doc = await users_col.find_one({"_id": ObjectId(user_id)}) if users_col is not None else None
            if user_doc and user_doc.get("notify_scan_complete", True):
                sent = await send_combined_weekly_email(
                    to_email=user_doc.get("email", "") or user_mock["email"],
                    name=user_doc.get("name", ""),
                    business_name=business_label,
                    domain=_normalize_site(url),
                    scrape=result,
                    ai_insights=ai_insights,
                    tracker=tracker_email_data,
                )
                print(f"[Scheduler] Combined weekly email for {url}: {'sent' if sent else 'not sent (SMTP unavailable)'}")
        except Exception as email_err:
            print(f"[Scheduler] Failed to send combined weekly email for {url}: {email_err}")

        # Stamped only after the pipeline above actually finished (whether
        # or not individual sub-steps like the email hit a caught error) —
        # this is the crash-resilience marker: a restart mid-Sunday-run
        # re-queries `businesses_col` and skips anything already stamped
        # for this week instead of redoing it or, worse, silently never
        # getting to it because the old code just slept until next Sunday
        # regardless of whether this run actually finished.
        if businesses_col is not None:
            try:
                await businesses_col.update_one(
                    {"_id": biz_doc["_id"]},
                    {"$set": {"weekly_run_week_id": week_id, "weekly_run_completed_at": datetime.utcnow().isoformat() + "Z"}},
                )
            except Exception as stamp_err:
                print(f"[Scheduler] Failed to stamp weekly_run_week_id for {url}: {stamp_err}")
    except Exception as scrape_err:
        print(f"[Scheduler] Auto scrape failed for {url}: {scrape_err}")


async def _run_weekly_pass(week_id: str) -> None:
    """Process every business not yet stamped for `week_id`, bounded to
    SUNDAY_SCHEDULER_CONCURRENCY at a time instead of the old one-at-a-time
    loop. Safe to call more than once for the same week — anything already
    stamped is skipped, so a restart resumes instead of redoing finished
    work or silently skipping the rest of the week."""
    if businesses_col is None:
        print("[Scheduler] businesses_col is None, skipping weekly pass")
        return

    cursor = businesses_col.find({"weekly_run_week_id": {"$ne": week_id}})
    pending = await cursor.to_list(length=10000)
    if not pending:
        print(f"[Scheduler] No businesses pending for week {week_id}")
        return
    print(f"[Scheduler] {len(pending)} businesses pending for week {week_id} (concurrency={SUNDAY_SCHEDULER_CONCURRENCY})")

    semaphore = asyncio.Semaphore(SUNDAY_SCHEDULER_CONCURRENCY)

    async def _run_one(doc: dict) -> None:
        async with semaphore:
            try:
                await _run_weekly_business_pipeline(doc, week_id)
            except Exception as pipeline_err:
                print(f"[Scheduler] Unhandled error processing {doc.get('url')}: {pipeline_err}")

    await asyncio.gather(*[_run_one(doc) for doc in pending])
    print(f"[Scheduler] Weekly pass for {week_id} finished")


async def sunday_analyzer_scheduler():
    print("[Scheduler] Weekly Sunday 4:00 AM analyzer task started")
    await asyncio.sleep(15)

    # Catch-up pass: if the process was down across a Sunday-4am window (or
    # crashed mid-run last time), don't silently wait a full extra week —
    # anything not yet stamped for the CURRENT week gets processed now.
    # `_run_weekly_pass` is idempotent (skips already-stamped businesses),
    # so this is safe even if last week's run actually did finish.
    try:
        now = datetime.now()
        this_weeks_sunday_4am = now - timedelta(days=(now.weekday() - 6) % 7)
        this_weeks_sunday_4am = this_weeks_sunday_4am.replace(hour=4, minute=0, second=0, microsecond=0)
        if now >= this_weeks_sunday_4am:
            print("[Scheduler] Startup catch-up: past this week's Sunday 4am, running catch-up pass now")
            await _run_weekly_pass(_blog_week_id(now))
    except Exception as catchup_err:
        print(f"[Scheduler] Startup catch-up failed: {catchup_err}")

    while True:
        sleep_sec = seconds_until_next_sunday_4am()
        hours_val = round(sleep_sec / 3600.0, 2)
        print(f"[Scheduler] Sleeping for {sleep_sec} seconds (approx {hours_val} hours) until next Sunday 4:00 AM")
        await asyncio.sleep(sleep_sec)

        print("[Scheduler] It is Sunday 4:00 AM. Starting weekly pass...")
        try:
            await _run_weekly_pass(_blog_week_id(datetime.now()))
        except Exception as run_err:
            print(f"[Scheduler] Exception in weekly pass: {run_err}")


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
