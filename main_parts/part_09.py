# Generated from the former backend/main.py lines 3379-3822.
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
    include_competitors = job_type == "deep"
    print(f"[Phase5] worker start job_id={job_id} type={job_type} provider={model_provider}")
    try:
        async def _is_cancelled() -> bool:
            latest = await phase5_jobs_col.find_one(
                {"job_id": job_id},
                {"status": 1, "_id": 0},
            )
            return not latest or latest.get("status") == "cancelled"

        def _compute_provider_scores(results_map: dict[str, dict]) -> tuple[dict, float | None]:
            providers = ["perplexity", "chatgpt", "claude"]
            score_map: dict[str, dict] = {}
            for provider_name in providers:
                score_map[provider_name] = compute_provider_score(results_map, provider_name)
            numeric_scores = [float(v.get("score", 0)) for v in score_map.values() if isinstance(v, dict)]
            overall = round(sum(numeric_scores) / len(numeric_scores), 2) if numeric_scores else None
            return score_map, overall

        questions = list(job_doc.get("questions", []))
        seed_results = job_doc.get("seed_results", {}) or {}
        accumulated_results = {}

        if include_competitors:
            deep_competitors = await generate_deep_competitor_scores(
                url=job_doc["url"],
                questions=questions,
                seed_results=seed_results,
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
                    )
                else:
                    per_question_timeout = PHASE5_QUESTION_TIMEOUT_GEMINI_SEC

                try:
                    if provider_for_q == "multi":
                        result = await asyncio.wait_for(
                            analyze_single_question_multi(
                                url=job_doc["url"],
                                question=q,
                                include_competitors=include_competitors,
                            ),
                            timeout=max(10, per_question_timeout),
                        )
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
                    result = {
                        "id": qid,
                        "status": "Not Mentioned",
                        "position": None,
                        "sources": [],
                        "source_urls": [],
                        "references": [],
                        "competitors": [],
                        "competitor_scores": [],
                        "reasoning": f"Timed out after {per_question_timeout}s",
                    }
                    if provider_for_q == "multi":
                        result = {
                            "id": qid,
                            "providers": {
                                "perplexity": {"mentioned": False, "position": None, "sources": [], "cited": False, "status": "Not Mentioned"},
                                "chatgpt": {"mentioned": False, "position": None, "sources": [], "cited": False, "status": "Not Mentioned"},
                                "claude": {"mentioned": False, "position": None, "sources": [], "cited": False, "status": "Not Mentioned"},
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
            return

        # Calculate accurate target score now that all questions are finished
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
        await phase5_jobs_col.update_one(
            {"job_id": job_id},
            {
                "$set": {
                    "status": "completed",
                    "current_question_id": None,
                    "provider_scores": final_scores,
                    "overall_score": final_overall,
                    "deep_competitors": deep_competitors,
                    "brand_summary": brand_summary,
                    "updated_at": datetime.utcnow().isoformat(),
                }
            },
        )
        try:
            external_system_competitors = [
                {
                    "domain": c.get("domain"),
                    "url": c.get("url"),
                    "name": c.get("name"),
                    "score": c.get("score"),
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
        except Exception as business_error:
            print(f"[Business] phase5 upsert failed: {business_error}")
        print(f"[Phase5] worker completed job_id={job_id} provider={model_provider}")

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
