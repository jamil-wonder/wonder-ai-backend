# Generated from the former backend/main.py lines 3825-4190.
import signal


async def _manual_run_worker_loop():
    """Same atomic-claim shape as `_phase5_worker_loop`, but for "Run now"
    requests queued by the API (see part_04.py) instead of run inline in the
    web process. Only started by `run_worker()` in the worker process."""
    while True:
        try:
            if manual_run_requests_col is None:
                await asyncio.sleep(PHASE5_WORKER_POLL_INTERVAL)
                continue
            claimed = await manual_run_requests_col.find_one_and_update(
                {"status": "queued"},
                {
                    "$set": {
                        "status": "running",
                        "worker_id": PHASE5_WORKER_ID,
                        "updated_at": datetime.utcnow().isoformat(),
                    }
                },
                sort=[("created_at", 1)],
                return_document=ReturnDocument.AFTER,
            )

            if not claimed:
                await asyncio.sleep(PHASE5_WORKER_POLL_INTERVAL)
                continue

            business_id = claimed.get("business_id")
            week_id = claimed.get("week_id")
            result_status = "completed"
            error = None
            try:
                biz_doc = await businesses_col.find_one({"_id": ObjectId(business_id)}) if businesses_col is not None else None
                if not biz_doc:
                    raise RuntimeError(f"business {business_id} not found")
                await _run_weekly_business_pipeline(biz_doc, week_id)
            except Exception as run_err:
                result_status = "failed"
                error = str(run_err)
                print(f"[Worker] manual run failed for business={business_id}: {run_err}")

            await manual_run_requests_col.update_one(
                {"_id": claimed["_id"]},
                {
                    "$set": {
                        "status": result_status,
                        "error": error,
                        "updated_at": datetime.utcnow().isoformat(),
                    }
                },
            )
        except asyncio.CancelledError:
            break
        except Exception:
            traceback.print_exc()
            await asyncio.sleep(PHASE5_WORKER_POLL_INTERVAL)


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


def seconds_until_next_wednesday_10am() -> float:
    now = datetime.now()
    days_until_wed = (2 - now.weekday()) % 7  # Wednesday = weekday 2
    if days_until_wed == 0 and now.hour >= 10:
        days_until_wed = 7
    target = datetime(now.year, now.month, now.day, 10, 0, 0, 0) + timedelta(days=days_until_wed)
    diff = target - now
    return max(1.0, diff.total_seconds())


async def _run_midweek_reminder_pass():
    """Part 5 of the flow spec: "a mid-week reminder only if tasks are
    undone and they're inactive." Both conditions are checked against real
    data: completedActions this week_id (server-tracked, not the old
    localStorage-only state) and users_col.last_active_at (stamped on every
    authenticated request). Never nags a business with real progress this
    week or a user who's actively in the dashboard regardless."""
    if businesses_col is None:
        return
    cursor = businesses_col.find({"questionsLocked": True})
    businesses = await cursor.to_list(length=10000)
    week_id = _blog_week_id(datetime.now())
    for biz_doc in businesses:
        try:
            user_id = biz_doc.get("user_id")
            if not user_id:
                continue

            completed_this_week = [
                a for a in (biz_doc.get("completedActions") or [])
                if isinstance(a, dict) and a.get("week_id") == week_id
            ]
            if completed_this_week:
                continue

            user_doc = await users_col.find_one({"_id": ObjectId(user_id)}) if users_col is not None else None
            if not user_doc or not user_doc.get("notify_scan_complete", True):
                continue

            last_active_at = user_doc.get("last_active_at")
            is_inactive = True
            if last_active_at:
                try:
                    last_dt = datetime.fromisoformat(str(last_active_at).rstrip("Z"))
                    is_inactive = (datetime.utcnow() - last_dt).total_seconds() >= 3 * 86400
                except Exception:
                    is_inactive = True
            if not is_inactive:
                continue

            latest_job = None
            if phase5_jobs_col is not None:
                latest_job = await phase5_jobs_col.find_one(
                    {"business_id": str(biz_doc.get("_id")), "job_type": "core", "model": "multi", "status": "completed"},
                    sort=[("created_at", -1)],
                )
            if not latest_job:
                continue
            results = latest_job.get("results") or {}
            total_questions = len(results)
            if total_questions == 0:
                continue
            mentioned = _count_total_mentions(results)
            not_mentioned = total_questions - mentioned
            if not_mentioned <= 0:
                continue

            sent = await send_midweek_reminder_email(
                to_email=user_doc.get("email", ""),
                name=user_doc.get("name", ""),
                business_name=biz_doc.get("businessName") or "",
                domain=_normalize_site(biz_doc.get("url") or ""),
                not_mentioned_count=not_mentioned,
                total_questions=total_questions,
            )
            print(f"[Scheduler] Midweek reminder for {biz_doc.get('url')}: {'sent' if sent else 'not sent (SMTP unavailable)'}")
        except Exception as e:
            print(f"[Scheduler] Midweek reminder failed for business {biz_doc.get('_id')}: {e}")


async def wednesday_reminder_scheduler():
    print("[Scheduler] Midweek reminder task started")
    await asyncio.sleep(20)
    while True:
        sleep_sec = seconds_until_next_wednesday_10am()
        hours_val = round(sleep_sec / 3600.0, 2)
        print(f"[Scheduler] Sleeping for {sleep_sec} seconds (approx {hours_val} hours) until next Wednesday 10:00 AM")
        await asyncio.sleep(sleep_sec)
        print("[Scheduler] It is Wednesday 10:00 AM. Running midweek reminder pass...")
        try:
            await _run_midweek_reminder_pass()
        except Exception as e:
            print(f"[Scheduler] Midweek reminder pass failed: {e}")


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
                                    ever_mentioned_ids = biz_doc.get("everMentionedQuestionIds") if isinstance(biz_doc.get("everMentionedQuestionIds"), list) else []
                                    alerts = _detect_alerts(
                                        current_score=float(current_score),
                                        previous_score=float(previous_score) if isinstance(previous_score, (int, float)) else None,
                                        current_competitors=deep_competitors,
                                        previous_competitors=(previous_tracker_job or {}).get("deep_competitors") or [],
                                        current_results=current_results,
                                        previous_results=previous_results,
                                        question_text_by_id=question_text_by_id,
                                        ever_mentioned_ids=ever_mentioned_ids,
                                    )

                                    # Update the "ever mentioned" record with
                                    # this run's real mentions — the only
                                    # source "first appearance" can ever
                                    # check against, so it has to stay
                                    # current regardless of whether an alert
                                    # actually fired this week.
                                    newly_mentioned_ids = {
                                        qid for qid in question_text_by_id
                                        if _question_mention_summary(current_results.get(qid))[0]
                                    }
                                    updated_ever_ids = sorted(set(ever_mentioned_ids) | newly_mentioned_ids)
                                    if updated_ever_ids != sorted(ever_mentioned_ids):
                                        await businesses_col.update_one(
                                            {"_id": biz_doc["_id"]},
                                            {"$set": {"everMentionedQuestionIds": updated_ever_ids}},
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

                                # Win email: real before/after diff against
                                # each completed action's own baseline
                                # (recorded the moment it was marked done),
                                # never a guessed or predicted number —
                                # "observed since you did X" per the spec's
                                # honesty rules, correlation only.
                                try:
                                    completed_actions = biz_doc.get("completedActions") or []
                                    current_mentions = _count_total_mentions(current_results)
                                    win_items: list[dict] = []
                                    actions_changed = False
                                    for act in completed_actions:
                                        if not isinstance(act, dict) or act.get("reported_win"):
                                            continue
                                        baseline = act.get("baseline_mentions")
                                        if not isinstance(baseline, (int, float)):
                                            continue
                                        delta = current_mentions - int(baseline)
                                        if delta > 0:
                                            win_items.append({
                                                "title": act.get("title") or "Your action",
                                                "before": int(baseline),
                                                "after": current_mentions,
                                                "total": len(current_results),
                                                "delta": delta,
                                            })
                                            act["reported_win"] = True
                                            actions_changed = True
                                    if actions_changed:
                                        await businesses_col.update_one(
                                            {"_id": biz_doc["_id"]},
                                            {"$set": {"completedActions": completed_actions}},
                                        )
                                    if win_items:
                                        win_user_doc = await users_col.find_one({"_id": ObjectId(user_id)}) if users_col is not None else None
                                        if win_user_doc and win_user_doc.get("notify_scan_complete", True):
                                            win_sent = await send_win_email(
                                                to_email=win_user_doc.get("email", "") or biz_doc.get("user_email") or "",
                                                name=win_user_doc.get("name", ""),
                                                business_name=biz_doc.get("businessName") or "",
                                                domain=target_domain,
                                                wins=win_items,
                                            )
                                            print(f"[Scheduler] Win email for {url}: {'sent' if win_sent else 'not sent (SMTP unavailable)'} ({len(win_items)} win(s))")
                                except Exception as win_err:
                                    print(f"[Scheduler] Win detection/send failed for {url}: {win_err}")
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


async def _ensure_phase5_indexes():
    """Indexes tuned to current Phase 5 query/update patterns. Idempotent,
    so it's safe to run from both the web process (on every startup) and
    the worker process — whichever container comes up first creates them."""
    if phase5_jobs_col is None:
        return
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


@app.on_event("startup")
async def _phase5_worker_startup():
    """Runs in the web (`api`) process only. Background schedulers, the
    Phase5/manual-run poll loops, and the startup recovery passes now live
    in `run_worker()` (backend/worker.py) instead — see
    docs/infra-diagnosis.html for why a web-process redeploy used to kill
    in-flight jobs. This hook just ensures indexes exist, which is cheap
    and safe to run from either process."""
    await _ensure_phase5_indexes()


async def _phase5_startup_recovery():
    """The stale-job recovery passes that used to run on every web-process
    startup. Now only run from `run_worker()`, since only the worker
    process owns in-flight job state."""
    # Cost-safety default: do not auto-resume previously queued/in-progress jobs after restart
    # unless explicitly enabled via env.
    if phase5_jobs_col is None:
        return
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

async def run_worker():
    """Entrypoint for the `worker` process (backend/worker.py, ROLE=worker).
    Owns everything that used to run inside the web process's FastAPI
    startup/shutdown hooks: the Sunday/Wednesday schedulers, the Phase5 and
    manual-run poll loops, the shared thread pool executor (Playwright
    scrapes run via asyncio.to_thread), and the startup recovery passes over
    in-flight jobs. Splitting this out means a web-tier deploy
    (`docker compose up -d --build api`) never touches this process, so an
    in-flight job survives a hotfix deploy — the incident in
    docs/infra-diagnosis.html."""
    print("[Worker] starting background worker process")
    await _ensure_phase5_indexes()

    loop = asyncio.get_running_loop()
    executor = ThreadPoolExecutor(max_workers=max(4, PHASE5_MODEL_MAX_THREADS))
    loop.set_default_executor(executor)

    await _phase5_startup_recovery()

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

    tasks = [
        asyncio.create_task(sunday_analyzer_scheduler()),
        asyncio.create_task(wednesday_reminder_scheduler()),
        asyncio.create_task(_manual_run_worker_loop()),
    ]
    if phase5_jobs_col is not None:
        tasks.extend(
            asyncio.create_task(_phase5_worker_loop())
            for _ in range(max(1, PHASE5_WORKER_CONCURRENCY))
        )

    stop_event = asyncio.Event()

    def _handle_stop_signal(*_args):
        print("[Worker] shutdown signal received")
        stop_event.set()

    for sig_name in ("SIGTERM", "SIGINT"):
        sig = getattr(signal, sig_name, None)
        if sig is None:
            continue
        try:
            loop.add_signal_handler(sig, _handle_stop_signal)
        except NotImplementedError:
            # Windows dev environments don't support add_signal_handler;
            # Ctrl+C still raises KeyboardInterrupt there, which is fine
            # since this worker only runs in Docker/Linux in production.
            pass

    await stop_event.wait()

    print("[Worker] shutting down: cancelling background tasks")
    for t in tasks:
        t.cancel()
    for t in tasks:
        try:
            await t
        except asyncio.CancelledError:
            pass
        except Exception:
            pass

    try:
        executor.shutdown(wait=False, cancel_futures=True)
    except Exception:
        pass

    print("[Worker] shutdown complete")
