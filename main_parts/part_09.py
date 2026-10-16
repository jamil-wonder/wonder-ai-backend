# Generated from the former backend/main.py lines 3379-3822.
class _Phase5RunIncomplete(Exception):
    """Internal: skips the score/competitor write for an incomplete run."""


def _phase5_compute_provider_scores(results_map: dict) -> tuple[dict, float | None]:
    providers = ["perplexity", "chatgpt", "claude", "gemini"]
    score_map: dict[str, dict] = {}
    for provider_name in providers:
        score_map[provider_name] = compute_provider_score(results_map, provider_name)

    # A model that failed or did not run (credits, rate limit, timeout)
    # has no usable answers: compute_provider_score reports it as
    # available=False and it is left out of the overall score, rather
    # than dragging it down with zeros (A4).
    usable_scores = [
        float(v.get("score", 0))
        for v in score_map.values()
        if isinstance(v, dict) and v.get("available")
    ]
    overall = round(sum(usable_scores) / len(usable_scores), 2) if usable_scores else None
    return score_map, overall


async def _scan_run_tracker_failed(job_doc: dict, error) -> None:
    """The tracker job died or was cancelled: record the leg as failed so the
    run can never be 'complete', and finalize if this job owns the run."""
    run_id = job_doc.get("scan_run_id")
    if not run_id:
        return
    await _scan_run_set_leg(run_id, "tracker", "failed", job_id=job_doc.get("job_id"), error=str(error)[:300])
    if job_doc.get("scan_run_finalize"):
        await _scan_run_finalize(run_id)


# Result-key -> provider name understood by _run_with_backoff
_PHASE5_RETRY_PROVIDER_CALL = {"perplexity": "perplexity", "chatgpt": "openai", "claude": "claude", "gemini": "gemini"}


async def _retry_failed_phase5_items(job_doc: dict) -> None:
    """Re-run ONLY the (question, model) answers that failed in an earlier run
    of this job, then re-decide whether the run is now complete (A4). Runs in
    the worker like any other job, so a web deploy can't interrupt it."""
    import copy
    job_id = job_doc["job_id"]
    url = job_doc["url"]
    questions_by_id = {str(q.get("id")): q for q in (job_doc.get("questions") or [])}
    results = copy.deepcopy(job_doc.get("results") or {})

    targets = [
        (qid, name)
        for qid, row in results.items()
        if isinstance(row, dict) and isinstance(row.get("providers"), dict)
        for name, pdata in row["providers"].items()
        if (provider_result_state(pdata) == "failed" or (isinstance(pdata, dict) and pdata.get("citations_missing")))
        and name in _PHASE5_RETRY_PROVIDER_CALL and str(qid) in questions_by_id
    ]
    print(f"[Phase5] retry-failed job_id={job_id}: {len(targets)} failed answers to re-run")
    run_id = job_doc.get("scan_run_id")
    if run_id:
        await _scan_run_restart(run_id)
        await _scan_run_set_leg(run_id, "tracker", "running", job_id=job_id)
    await phase5_jobs_col.update_one(
        {"job_id": job_id},
        {"$set": {"status": "running", "processed": 0, "total": len(targets), "retry_failed_only": False,
                  "updated_at": datetime.utcnow().isoformat()}},
    )

    per_call_timeout = max(
        PHASE5_QUESTION_TIMEOUT_OPENAI_SEC, PHASE5_QUESTION_TIMEOUT_PERPLEXITY_SEC,
        PHASE5_QUESTION_TIMEOUT_ANTHROPIC_SEC, PHASE5_QUESTION_TIMEOUT_GEMINI_SEC, 10,
    )
    sem = asyncio.Semaphore(max(1, PHASE5_JOB_PARALLELISM))

    async def _one(qid: str, name: str) -> None:
        async with sem:
            try:
                raw = await asyncio.wait_for(
                    _run_with_backoff(url, questions_by_id[str(qid)], model_provider=_PHASE5_RETRY_PROVIDER_CALL[name]),
                    timeout=per_call_timeout,
                )
            except asyncio.TimeoutError:
                raw = RuntimeError(f"Timed out after {per_call_timeout}s")
            except Exception as e:
                raw = e
            fresh = _safe_provider_result(raw)
        results[qid]["providers"][name] = fresh
        await phase5_jobs_col.update_one(
            {"job_id": job_id},
            {"$set": {f"results.{qid}.providers.{name}": fresh, "updated_at": datetime.utcnow().isoformat()},
             "$inc": {"processed": 1}},
        )

    await asyncio.gather(*[_one(qid, name) for qid, name in targets])

    scores, overall = _phase5_compute_provider_scores(results)
    health = compute_run_health(results)
    incomplete = bool(health.get("incomplete"))
    if incomplete:
        overall = None
    else:
        try:
            await _upsert_user_business(
                current_user={"id": job_doc.get("user_id"), "email": job_doc.get("user_email")} if job_doc.get("user_id") else None,
                url=url,
                phase5_score=overall,
                business_id=job_doc.get("business_id"),
            )
        except Exception as business_error:
            print(f"[Business] phase5 retry upsert failed: {business_error}")
    await phase5_jobs_col.update_one(
        {"job_id": job_id},
        {"$set": {
            "status": "incomplete" if incomplete else "completed",
            "current_question_id": None,
            "provider_scores": scores,
            "overall_score": overall,
            "run_health": health,
            "error": (
                "Run incomplete: some AI models could not be reached "
                f"({', '.join(sorted(health['failed_by_provider']))}). No score was recorded."
                if incomplete else None
            ),
            "updated_at": datetime.utcnow().isoformat(),
        }},
    )
    if run_id:
        await _scan_run_set_leg(
            run_id, "tracker", "incomplete" if incomplete else "ok", job_id=job_id,
            mention_rate=None if incomplete else compute_mention_rate(results, job_doc.get("questions") or []),
            mention_rate_margin=None if incomplete else compute_mention_rate_margin(results, job_doc.get("questions") or []),
            health=health,
        )
        await _scan_run_finalize(run_id)  # a retry that succeeds is what finally unlocks the score
    print(f"[Phase5] retry-failed job_id={job_id} -> {'incomplete' if incomplete else 'completed'} (overall={overall})")


async def _process_phase5_job(job_doc: dict):
    job_id = job_doc["job_id"]
    job_type = job_doc.get("job_type", "core")
    model_provider = str(job_doc.get("model", "perplexity") or "perplexity").strip().lower()
    if model_provider == "gemini" and not PHASE5_ENABLE_GEMINI:
        await phase5_jobs_col.update_one(
            {"job_id": job_id},
            {
                "$set": {
                    "status": "failed",
                    "current_question_id": None,
                    "error": "Gemini is temporarily disabled for Phase 5. Use OpenAI, Perplexity, or Claude.",
                    "updated_at": datetime.utcnow().isoformat(),
                }
            },
        )
        return
    if job_doc.get("retry_failed_only"):
        try:
            await _retry_failed_phase5_items(job_doc)
        except Exception as e:
            traceback.print_exc()
            await phase5_jobs_col.update_one(
                {"job_id": job_id},
                {"$set": {"status": "failed", "current_question_id": None, "error": str(e),
                          "retry_failed_only": False, "updated_at": datetime.utcnow().isoformat()}},
            )
            await _scan_run_tracker_failed({**job_doc, "scan_run_finalize": True}, e)
        return
    include_competitors = job_type == "deep"
    print(f"[Phase5] worker start job_id={job_id} type={job_type} provider={model_provider}")
    try:
        async def _is_cancelled() -> bool:
            latest = await phase5_jobs_col.find_one(
                {"job_id": job_id},
                {"status": 1, "_id": 0},
            )
            return not latest or latest.get("status") == "cancelled"

        _compute_provider_scores = _phase5_compute_provider_scores

        questions = list(job_doc.get("questions", []))
        seed_results = job_doc.get("seed_results", {}) or {}
        accumulated_results = {}

        if include_competitors:
            deep_competitors = await generate_deep_competitor_scores(
                url=job_doc["url"],
                questions=questions,
                seed_results=seed_results,
                target_profile=await _competitor_target_profile(job_doc.get("business_id")),
            )

            await phase5_jobs_col.update_one(
                {"job_id": job_id, "status": {"$ne": "cancelled"}},
                {
                    "$set": {
                        "results": seed_results if isinstance(seed_results, dict) else {},
                        "processed": len(questions),
                        "deep_competitors": deep_competitors,
                        "status": "completed",
                        "current_question_id": None,
                        "updated_at": datetime.utcnow().isoformat(),
                    }
                },
            )

            async def _finalize_brand_summary_async():
                try:
                    brand_summary = await generate_brand_perception_summary(
                        url=job_doc["url"],
                        questions=questions,
                        results=seed_results if isinstance(seed_results, dict) else {},
                    )
                    await phase5_jobs_col.update_one(
                        {"job_id": job_id, "status": "completed"},
                        {
                            "$set": {
                                "brand_summary": brand_summary,
                                "updated_at": datetime.utcnow().isoformat(),
                            }
                        },
                    )
                    try:
                        await _upsert_user_business(
                            current_user={"id": job_doc.get("user_id"), "email": job_doc.get("user_email")} if job_doc.get("user_id") else None,
                            url=job_doc["url"],
                            business_id=job_doc.get("business_id"),
                        )
                    except Exception as business_error:
                        print(f"[Business] phase5 deep summary upsert failed: {business_error}")
                except Exception:
                    traceback.print_exc()

            asyncio.create_task(_finalize_brand_summary_async())
            return

        lock = asyncio.Lock()
        queue: asyncio.Queue[dict] = asyncio.Queue()
        for q in questions:
            queue.put_nowait(q)

        # A5: re-use this week's stored answers from all four models where we
        # have them; only ask a model again for an answer we don't have.
        snapshot, snapshot_meta = ({}, None)
        if model_provider == "multi" and job_type == "core":
            snapshot, snapshot_meta = await _load_answer_snapshot(
                job_doc.get("business_id"), questions, fresh=bool(job_doc.get("fresh")),
            )
        snapshot_stats = {"reused": 0, "asked": 0}
        collected_at = (snapshot_meta or {}).get("collected_at") or datetime.utcnow().isoformat() + "Z"
        await phase5_jobs_col.update_one(
            {"job_id": job_id},
            {"$set": {"answers_collected_at": collected_at,
                      "snapshot_from_job": (snapshot_meta or {}).get("job_id")}},
        )
        if snapshot_meta:
            print(f"[Phase5] job_id={job_id} re-using stored answers collected {collected_at}")

        # Background finalization: start it immediately in parallel with workers
        # so it doesn't block question analysis. It will use its own Perplexity
        # probes if seed_results is empty/incomplete.
        always_run_deep = os.getenv("PHASE5_ALWAYS_RUN_DEEP", "false").strip().lower() == "true"
        finalize_task = None
        if always_run_deep or include_competitors:
            async def _finalize_background_task():
                print(f"[Phase5] background competitor finalizing job_id={job_id}")
                try:
                    # Run competitor discovery early using Perplexity probes.
                    # It will return 4 external competitors. Target site is added later.
                    deep_competitors = await generate_deep_competitor_scores(
                        url=job_doc["url"],
                        questions=questions,
                        seed_results={}, # Intentionally empty to force fast probe
                        target_profile=await _competitor_target_profile(job_doc.get("business_id")),
                    )
                    
                    # Update deep competitors immediately so UI has early partial data
                    await phase5_jobs_col.update_one(
                        {"job_id": job_id, "status": {"$ne": "cancelled"}},
                        {
                            "$set": {
                                "deep_competitors": deep_competitors,
                                "updated_at": datetime.utcnow().isoformat(),
                            }
                        },
                    )
                    print(f"[Phase5] background competitor finalizing done job_id={job_id}")
                    return deep_competitors
                except Exception:
                    traceback.print_exc()
                    return []

            finalize_task = asyncio.create_task(_finalize_background_task())

        async def _worker_run():
            while True:
                if await _is_cancelled():
                    break

                try:
                    q = queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

                qid = q["id"]
                print(f"[Phase5] run qid={qid} job_id={job_id} provider={model_provider}")
                await phase5_jobs_col.update_one(
                    {"job_id": job_id, "status": {"$ne": "cancelled"}},
                    {
                        "$set": {
                            "current_question_id": qid,
                            "updated_at": datetime.utcnow().isoformat(),
                        }
                    },
                )

                provider_for_q = str(job_doc.get("model", "perplexity") or "perplexity").strip().lower()
                if provider_for_q == "openai":
                    per_question_timeout = PHASE5_QUESTION_TIMEOUT_OPENAI_SEC
                elif provider_for_q == "perplexity":
                    per_question_timeout = PHASE5_QUESTION_TIMEOUT_PERPLEXITY_SEC
                elif provider_for_q == "claude":
                    per_question_timeout = PHASE5_QUESTION_TIMEOUT_ANTHROPIC_SEC
                elif provider_for_q == "multi":
                    per_question_timeout = max(
                        PHASE5_QUESTION_TIMEOUT_OPENAI_SEC,
                        PHASE5_QUESTION_TIMEOUT_PERPLEXITY_SEC,
                        PHASE5_QUESTION_TIMEOUT_ANTHROPIC_SEC,
                        PHASE5_QUESTION_TIMEOUT_GEMINI_SEC,
                    )
                else:
                    per_question_timeout = PHASE5_QUESTION_TIMEOUT_GEMINI_SEC

                try:
                    if provider_for_q == "multi":
                        cached_row = snapshot.get(str(q.get("text") or "").strip())
                        missing = _snapshot_missing_providers(cached_row) if cached_row else None
                        if cached_row and not missing:
                            result, reused_n = _merge_snapshot_row(q, cached_row, None)
                            snapshot_stats["reused"] += reused_n
                        else:
                            fresh_row = await asyncio.wait_for(
                                analyze_single_question_multi(
                                    url=job_doc["url"],
                                    question=q,
                                    include_competitors=include_competitors,
                                    **({"only_providers": missing} if missing else {}),
                                ),
                                timeout=max(10, per_question_timeout),
                            )
                            if cached_row:
                                result, reused_n = _merge_snapshot_row(q, cached_row, fresh_row)
                                snapshot_stats["reused"] += reused_n
                            else:
                                result = fresh_row
                            snapshot_stats["asked"] += len(missing) if missing else 4
                    else:
                        result = await asyncio.wait_for(
                            _run_with_backoff(
                                job_doc["url"],
                                q,
                                model_provider=provider_for_q,
                                include_competitors=include_competitors,
                            ),
                            timeout=max(10, per_question_timeout),
                        )
                except asyncio.TimeoutError:
                    print(
                        f"[Phase5] qid={qid} job_id={job_id} provider={provider_for_q} "
                        f"hard-timeout after {per_question_timeout}s"
                    )
                    timeout_msg = f"Timed out after {per_question_timeout}s"
                    result = {
                        "id": qid,
                        "fetch_status": "failed",
                        "error": timeout_msg,
                        "status": "Not Mentioned",
                        "position": None,
                        "sources": [],
                        "source_urls": [],
                        "references": [],
                        "competitors": [],
                        "competitor_scores": [],
                        "reasoning": timeout_msg,
                    }
                    if provider_for_q == "multi":
                        # A timeout is missing data, not "not mentioned" (A4).
                        def _timed_out() -> dict:
                            return {
                                "mentioned": False, "position": None, "sources": [], "cited": False,
                                "status": "Not Mentioned", "fetch_status": "failed", "error": timeout_msg,
                            }
                        result = {
                            "id": qid,
                            "providers": {
                                "perplexity": _timed_out(),
                                "chatgpt": _timed_out(),
                                "claude": _timed_out(),
                                "gemini": _timed_out(),
                            },
                        }

                async with lock:
                    accumulated_results[qid] = result

                running_scores, running_overall = _compute_provider_scores(accumulated_results) if provider_for_q == "multi" else ({}, None)

                print(
                    f"[Phase5] done qid={qid} job_id={job_id} provider={model_provider} "
                    f"status={result.get('status', 'multi')} pos={result.get('position')}"
                )

                await phase5_jobs_col.update_one(
                    {"job_id": job_id, "status": {"$ne": "cancelled"}},
                    {
                        "$set": {
                            f"results.{qid}": result,
                            "provider_scores": running_scores,
                            "overall_score": running_overall,
                            "updated_at": datetime.utcnow().isoformat(),
                        },
                        "$inc": {"processed": 1},
                    },
                )
                queue.task_done()

        workers = [
            asyncio.create_task(_worker_run())
            for _ in range(max(1, PHASE5_JOB_PARALLELISM))
        ]
        await asyncio.gather(*workers)
        
        deep_competitors = []
        # Ensure the background finalization task (started earlier) is finished
        if finalize_task:
            try:
                # Do not cancel competitor discovery if it needs more time. If
                # it is still running after this wait, fall back below using the
                # completed prompt results instead of completing with only the
                # target site.
                deep_competitors = await asyncio.wait_for(asyncio.shield(finalize_task), timeout=40)
            except asyncio.TimeoutError:
                print(f"[Phase5] background competitor finalization still running job_id={job_id}; using fallback generation")
            except Exception as e:
                print(f"[Phase5] background finalization wait error: {e}")

        external_deep_competitors = [
            c for c in (deep_competitors or [])
            if isinstance(c, dict) and c.get("domain") and c.get("confidence") != "target"
        ]
        if (always_run_deep or include_competitors) and not external_deep_competitors:
            try:
                print(f"[Phase5] final competitor fallback generation job_id={job_id}")
                fallback_competitors = await asyncio.wait_for(
                    generate_deep_competitor_scores(
                        url=job_doc["url"],
                        questions=questions,
                        seed_results=accumulated_results,
                        target_profile=await _competitor_target_profile(job_doc.get("business_id")),
                    ),
                    timeout=75,
                )
                if isinstance(fallback_competitors, list) and fallback_competitors:
                    deep_competitors = fallback_competitors
            except Exception as e:
                print(f"[Phase5] final competitor fallback failed job_id={job_id}: {e}")

        processed_count = len(accumulated_results)

        if await _is_cancelled():
            await phase5_jobs_col.update_one(
                {"job_id": job_id},
                {
                    "$set": {
                        "current_question_id": None,
                        "updated_at": datetime.utcnow().isoformat(),
                    }
                },
            )
            await _scan_run_tracker_failed(job_doc, "cancelled")
            return

        # Calculate the target score now that all questions are finished (replaced by the
        # measured one below when a market ladder could be built)
        target_score = _estimate_target_visibility_score(accumulated_results)
        domain = _normalize_domain(job_doc["url"])
        
        # Remove any existing target site entry from deep_competitors
        deep_competitors = [c for c in (deep_competitors or []) if c.get("domain") != domain]
        
        # Append target site with the fully accurate score
        deep_competitors.append({
            "domain": domain,
            "url": f"https://{domain}/" if domain else job_doc["url"],
            "name": domain.split(".")[0].replace("-", " ").title() if domain else "Your site",
            "position": None,
            "score": target_score,
            "evidence": "Your site score from visibility and rank consistency across analyzed prompts.",
            "confidence": "target",
        })

        # A2: score every competitor the same way as the business - from the same
        # questions and the same stored answers - instead of an estimate. Without a
        # measured comparison there is no ladder (the old estimated scores are not shown).
        ladder = {"rows": [], "position": None, "measured": False}
        if model_provider == "multi":
            try:
                ladder = build_market_ladder(
                    accumulated_results, questions, domain, None,
                    [c for c in deep_competitors if isinstance(c, dict) and c.get("confidence") != "target"],
                )
            except Exception as ladder_error:
                print(f"[Phase5] market ladder failed job_id={job_id}: {ladder_error}")
        ladder_by_domain = {r["domain"]: r for r in ladder["rows"]}
        for c in deep_competitors:
            row = ladder_by_domain.get(_normalize_domain(str(c.get("domain") or ""))) if ladder["measured"] else None
            if row:
                c["score"] = row["score"]
                c["evidence"] = market_evidence_text(row)
                c["score_basis"] = "measured"
                c["appeared"] = row["appeared"]
                c["answered"] = row["answered"]
            elif c.get("confidence") != "target":
                c["score_basis"] = "estimated"

        # Generate accurate brand summary now that questions are finished
        brand_summary = None
        if always_run_deep or include_competitors:
            try:
                brand_summary = await generate_brand_perception_summary(
                    url=job_doc["url"],
                    questions=questions,
                    results=accumulated_results,
                )
            except Exception as e:
                print(f"[Phase5] brand summary error: {e}")

        # Finalize job status
        final_scores, final_overall = _compute_provider_scores(accumulated_results) if model_provider == "multi" else ({}, None)

        # J6/A4: if too many model answers failed (credits, rate limits,
        # timeouts) the run is "incomplete": no score is written anywhere,
        # so a half-measured number can never reach a client or the trend.
        run_health = compute_run_health(accumulated_results) if model_provider == "multi" else None
        run_incomplete = bool(run_health and run_health.get("incomplete"))
        if run_incomplete:
            final_overall = None
            print(
                f"[Phase5] job_id={job_id} INCOMPLETE: {run_health['failed']} of "
                f"{run_health['ok'] + run_health['failed']} model answers failed "
                f"{run_health['failed_by_provider']}; no score written"
            )

        # Merge into the business's persisted systemCompetitors BEFORE
        # marking the job "completed" — the frontend refetches
        # /api/user/businesses the instant it observes "completed", so if
        # this ran afterward instead, that refetch could easily land in the
        # gap and see stale data (this was exactly that race).
        try:
            if run_incomplete:
                raise _Phase5RunIncomplete()
            external_system_competitors = [
                {
                    "domain": c.get("domain"),
                    "url": c.get("url"),
                    "name": c.get("name"),
                    # A2: only a measured score is stored as a score
                    "score": c.get("score") if c.get("score_basis") == "measured" else None,
                    "score_basis": "measured" if c.get("score_basis") == "measured" else None,
                    "status": c.get("confidence") or "AI evidence",
                    "evidence": c.get("evidence") or "",
                }
                for c in sorted(
                    [
                        c for c in (deep_competitors or [])
                        if isinstance(c, dict) and c.get("domain") and c.get("domain") != domain
                    ],
                    key=lambda item: float(item.get("score") or 0),
                    reverse=True,
                )[:4]
            ]
            await _upsert_user_business(
                current_user={"id": job_doc.get("user_id"), "email": job_doc.get("user_email")} if job_doc.get("user_id") else None,
                url=job_doc["url"],
                phase5_score=final_overall,
                business_id=job_doc.get("business_id"),
                system_competitors=external_system_competitors,
            )
        except _Phase5RunIncomplete:
            pass
        except Exception as business_error:
            print(f"[Business] phase5 upsert failed: {business_error}")

        await phase5_jobs_col.update_one(
            {"job_id": job_id},
            {
                "$set": {
                    "status": "incomplete" if run_incomplete else "completed",
                    "current_question_id": None,
                    "provider_scores": final_scores,
                    "overall_score": final_overall,
                    "run_health": run_health,
                    "error": (
                        (
                            "Run incomplete: more than half of the AI answers came back without any "
                            "sources, so we can't tell whether your site was cited. No score was recorded."
                            if run_health.get("incomplete_reason") == "citations_missing"
                            else "Run incomplete: some AI models could not be reached "
                                 f"({', '.join(sorted(run_health['failed_by_provider']))}). No score was recorded."
                        )
                        if run_incomplete else None
                    ),
                    "deep_competitors": deep_competitors,
                    "market_ladder": ladder,
                    "brand_summary": brand_summary,
                    "updated_at": datetime.utcnow().isoformat(),
                }
            },
        )
        print(f"[Phase5] worker {'incomplete' if run_incomplete else 'completed'} job_id={job_id} provider={model_provider}")

        scan_run_id = job_doc.get("scan_run_id")
        if scan_run_id and model_provider == "multi":
            if ladder["measured"] and not run_incomplete:
                await _scan_run_set_leg(scan_run_id, "competitors", "ok", market_position=ladder["position"], job_id=job_id)
            else:
                await _scan_run_set_leg(scan_run_id, "competitors", "skipped")
            await _scan_run_set_leg(
                scan_run_id, "tracker", "incomplete" if run_incomplete else "ok", job_id=job_id,
                mention_rate=None if run_incomplete else compute_mention_rate(accumulated_results, questions),
                mention_rate_margin=None if run_incomplete else compute_mention_rate_margin(accumulated_results, questions),
                health=run_health,
            )
            if job_doc.get("scan_run_finalize"):
                await _scan_run_finalize(scan_run_id)

        await _log_ai_usage_event({
            "feature": "phase5_job_completed",
            "endpoint": "/api/phase5/start-job",
            "url": job_doc.get("url"),
            "user_id": job_doc.get("user_id"),
            "user_email": job_doc.get("user_email"),
            "model_provider": model_provider,
            "ai_calls_estimate": int(processed_count),
            "details": {
                "job_id": job_id,
                "job_type": job_type,
                "model_provider": model_provider,
                "processed": processed_count,
                "total": len(questions),
            },
        })
    except Phase5RateLimitError as e:
        processed_count = len(locals().get("accumulated_results", {}) or {})
        await _log_ai_usage_event({
            "feature": "phase5_job_failed_rate_limit",
            "endpoint": "/api/phase5/start-job",
            "url": job_doc.get("url"),
            "user_id": job_doc.get("user_id"),
            "user_email": job_doc.get("user_email"),
            "model_provider": model_provider,
            "ai_calls_estimate": int(processed_count),
            "details": {
                "job_id": job_id,
                "job_type": job_type,
                "model_provider": model_provider,
                "processed": processed_count,
                "error": str(e),
            },
        })
        await phase5_jobs_col.update_one(
            {"job_id": job_id},
            {
                "$set": {
                    "status": "failed",
                    "current_question_id": None,
                    "error": str(e),
                    "updated_at": datetime.utcnow().isoformat(),
                }
            },
        )
        await _scan_run_tracker_failed(job_doc, e)
    except Exception as e:
        processed_count = len(locals().get("accumulated_results", {}) or {})
        await _log_ai_usage_event({
            "feature": "phase5_job_failed",
            "endpoint": "/api/phase5/start-job",
            "url": job_doc.get("url"),
            "user_id": job_doc.get("user_id"),
            "user_email": job_doc.get("user_email"),
            "model_provider": model_provider,
            "ai_calls_estimate": int(processed_count),
            "details": {
                "job_id": job_id,
                "job_type": job_type,
                "model_provider": model_provider,
                "processed": processed_count,
                "error": str(e),
            },
        })
        traceback.print_exc()
        await phase5_jobs_col.update_one(
            {"job_id": job_id},
            {
                "$set": {
                    "status": "failed",
                    "current_question_id": None,
                    "error": str(e),
                    "updated_at": datetime.utcnow().isoformat(),
                }
            },
        )
        await _scan_run_tracker_failed(job_doc, e)
