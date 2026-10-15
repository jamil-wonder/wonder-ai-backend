# OFF by default (0): every tracker run asks all four models afresh - that is
# what the Search Tracker is for. Set PHASE5_SNAPSHOT_DAYS=7 only if the owner
# explicitly wants a re-run inside that many days to re-use the stored answers
# (identical score, no AI calls, but it hides what the models say right now).
PHASE5_SNAPSHOT_DAYS = int(os.getenv("PHASE5_SNAPSHOT_DAYS", "0"))
_SNAPSHOT_PROVIDERS = ("perplexity", "chatgpt", "claude", "gemini")


def _parse_iso_utc(value):
    try:
        text = str(value or "").replace("Z", "")
        dt = datetime.fromisoformat(text)
        return dt.replace(tzinfo=None)
    except Exception:
        return None


async def _load_answer_snapshot(business_id, questions: list, *, fresh: bool = False):
    """(snapshot, meta): `snapshot` maps question text -> the stored per-model
    answers of the business's latest COMPLETE tracker run that is still inside
    the snapshot window; `meta` = {"collected_at", "job_id"}. ({}, None) when
    there is none, the questions changed, or a fresh run was requested."""
    if fresh or not business_id or phase5_jobs_col is None or PHASE5_SNAPSHOT_DAYS <= 0:
        return {}, None
    job = await phase5_jobs_col.find_one(
        {"business_id": str(business_id), "job_type": "core", "model": "multi", "status": "completed"},
        sort=[("created_at", -1)],
        projection={"results": 1, "questions": 1, "created_at": 1, "answers_collected_at": 1, "job_id": 1},
    )
    if not job:
        return {}, None
    collected = job.get("answers_collected_at") or job.get("created_at")
    when = _parse_iso_utc(collected)
    if when is None or (datetime.utcnow() - when).total_seconds() > PHASE5_SNAPSHOT_DAYS * 86400:
        return {}, None
    results = job.get("results") or {}
    by_text = {}
    for q in (job.get("questions") or []):
        text = str(q.get("text") or "").strip()
        row = results.get(str(q.get("id")))
        if text and isinstance(row, dict) and isinstance(row.get("providers"), dict):
            by_text[text] = row
    wanted = {str(q.get("text") or "").strip() for q in questions}
    if not wanted or not (wanted & set(by_text)):
        return {}, None
    return by_text, {"collected_at": collected, "job_id": job.get("job_id")}


def _merge_snapshot_row(question: dict, cached_row, fresh_row):
    """Combine stored per-model answers with freshly asked ones. Returns
    (row, reused_count). A stored answer is only used if it was a good one."""
    providers = {}
    reused = 0
    cached = (cached_row or {}).get("providers") or {}
    for key in _SNAPSHOT_PROVIDERS:
        cell = cached.get(key)
        if isinstance(cell, dict) and provider_result_state(cell) == "ok":
            providers[key] = copy.deepcopy(cell)
            reused += 1
        elif fresh_row and key in (fresh_row.get("providers") or {}):
            providers[key] = fresh_row["providers"][key]
    for key, cell in ((fresh_row or {}).get("providers") or {}).items():
        providers.setdefault(key, cell)
    return {"id": question.get("id"), "providers": providers}, reused


def _snapshot_missing_providers(cached_row) -> set:
    cached = (cached_row or {}).get("providers") or {}
    return {k for k in _SNAPSHOT_PROVIDERS if not (isinstance(cached.get(k), dict) and provider_result_state(cached[k]) == "ok")}


async def _competitor_target_profile(business_id) -> dict:
    """J4: what the business says it is (category + sub-type + area). Competitor
    discovery and validation must match all three, not guess from the page."""
    if not business_id or businesses_col is None:
        return {}
    try:
        doc = await businesses_col.find_one({"_id": ObjectId(str(business_id))})
    except Exception:
        return {}
    if not doc:
        return {}
    return {
        "name": doc.get("businessName") or "",
        "category": doc.get("category") or "",
        "sub_type": doc.get("sub_type") or "",
        "location": doc.get("location") or "",
        "description": doc.get("aiDescription") or doc.get("businessDescription") or "",
    }


# Generated from the former backend/main.py lines 860-1269.
async def _upsert_user_business(
    *,
    current_user: dict | None,
    url: str,
    category: str | None = None,
    sub_type: str | None = None,
    location: str | None = None,
    location_postcode: str | None = None,
    business_name: str | None = None,
    logo_url: str | None = None,
    business_description: str | None = None,
    ai_description: str | None = None,
    services: list[str] | None = None,
    target_audience: str | None = None,
    question_generation: dict | None = None,
    tracked_questions: list[dict] | None = None,
    competitors: list[str] | None = None,
    system_competitors: list[dict] | None = None,
    tracked_pages: list[str] | None = None,
    phase1_score: int | None = None,
    phase5_score: float | None = None,
    business_id: str | None = None,
    scrape_result: dict | None = None,
    analysis: dict | None = None,
) -> dict | None:
    if not current_user or businesses_col is None:
        return None

    normalized_domain = _normalize_site(url)
    if not normalized_domain:
        return None

    existing = None
    if business_id:
        try:
            existing = await businesses_col.find_one({"_id": ObjectId(business_id), "user_id": current_user["id"]})
        except:
            pass
    if not existing:
        existing = await businesses_col.find_one({"user_id": current_user["id"], "normalized_domain": normalized_domain})

    if not existing:
        current_count = await businesses_col.count_documents({"user_id": current_user["id"]})
        if current_count >= 3:
            raise HTTPException(status_code=400, detail="You can save up to 3 business profiles for now")

    now_iso = datetime.utcnow().isoformat() + "Z" 
    set_fields = {
        "user_id": current_user["id"],
        "user_email": current_user.get("email"),
        "url": url,
        "normalized_domain": normalized_domain,
        "updated_at": now_iso,
    }

    optional_pairs = {
        "category": _clean_optional_text(category),
        "location": _clean_optional_text(location),
        "businessName": _clean_optional_text(business_name),
        "logoUrl": _clean_optional_text(logo_url),
        "businessDescription": _clean_optional_text(business_description),
        "aiDescription": _clean_optional_text(ai_description),
        "targetAudience": _clean_optional_text(target_audience),
    }
    for key, value in optional_pairs.items():
        if value:
            set_fields[key] = value
    # G3: the location is stored as text AND as structured data. Only a city/area is
    # accepted (not a street address, postcode or URL) and it is Title Cased; a value
    # that fails the check is left out so the previous good location stays.
    if _clean_optional_text(location):
        if not location_input_problem(location):
            parsed_location = parse_location_input(location, location_postcode or "")
            if parsed_location["city"]:
                set_fields["location"] = parsed_location["label"]
                set_fields["location_data"] = {k: parsed_location[k] for k in ("area", "city", "country", "postcode")}
            else:
                set_fields.pop("location", None)
        else:
            set_fields.pop("location", None)     # the generic field code above already queued the raw text
            print(f"[Business] ignored invalid location {location!r}: {location_input_problem(location)}")
    if sub_type is not None:
        # unlike the other optional fields, "" is meaningful here: it clears
        # the sub-type (a business that isn't a niche within its category)
        set_fields["sub_type"] = _clean_optional_text(sub_type) or ""
    if phase1_score is not None:
        set_fields["latest_phase1_score"] = int(phase1_score)
        set_fields["latest_phase1_at"] = now_iso
    if phase5_score is not None:
        set_fields["latest_phase5_score"] = float(phase5_score)
        set_fields["latest_phase5_at"] = now_iso
    if scrape_result is not None:
        set_fields["latest_scrape_result"] = scrape_result
    if isinstance(analysis, dict) and analysis.get("scan"):
        # The exact scorecard/audit/AI-insight set the Analyzer showed for a
        # manual run, so reopening the page (any device, any time) shows what
        # was already analysed instead of asking to run it again.
        set_fields["latest_analysis"] = analysis
    for key, values in {
        "services": services,
        "competitors": competitors,
        "trackedPages": tracked_pages,
    }.items():
        if isinstance(values, list):
            cleaned = [str(v).strip() for v in values if str(v or "").strip()]
            set_fields[key] = cleaned
    if isinstance(question_generation, dict):
        set_fields["questionGeneration"] = _normalize_question_generation_settings(question_generation)
    if isinstance(tracked_questions, list):
        # This is the actual locked/tracked baseline — id/type/label/query
        # only, no live run results (those come from phase5 jobs/query
        # history separately). Saving it here is what makes the baseline
        # real and durable instead of living only in a 2-hour browser cache.
        cleaned_questions = []
        for item in tracked_questions:
            if not isinstance(item, dict):
                continue
            text = str(item.get("query") or "").strip()
            if not text:
                continue
            cleaned_questions.append({
                "id": item.get("id"),
                "type": str(item.get("type") or "").strip(),
                "label": str(item.get("label") or "").strip(),
                "query": text,
            })
        set_fields["trackedQuestions"] = cleaned_questions[:40]
    if isinstance(system_competitors, list):
        # Merge newly discovered competitors into the ones already saved on
        # this business profile, instead of replacing the list each run.
        # A domain found again just gets its score/evidence refreshed; a
        # domain that isn't rediscovered this run is left alone (it may
        # simply not have come up for this particular question set/run).
        MAX_SYSTEM_COMPETITORS = 8
        existing_system_competitors = (
            existing.get("systemCompetitors") if existing and isinstance(existing.get("systemCompetitors"), list) else []
        )

        merged: dict[str, dict] = {}
        for item in existing_system_competitors:
            if not isinstance(item, dict):
                continue
            domain = _normalize_site(str(item.get("domain") or item.get("url") or ""))
            if not domain:
                continue
            merged[domain] = {**item, "domain": domain}

        for item in system_competitors:
            if not isinstance(item, dict):
                continue
            domain = _normalize_site(str(item.get("domain") or item.get("url") or ""))
            if not domain:
                continue
            prior = merged.get(domain)
            merged[domain] = {
                "domain": domain,
                "url": item.get("url") or (prior or {}).get("url") or f"https://{domain}/",
                "name": _clean_optional_text(item.get("name")) or (prior or {}).get("name") or domain,
                "score": int(float(item.get("score") or 0)),
                "status": _clean_optional_text(item.get("status")) or _clean_optional_text(item.get("confidence")) or "AI evidence",
                "evidence": _clean_optional_text(item.get("evidence")) or "",
                "first_seen_at": (prior or {}).get("first_seen_at") or now_iso,
                "updated_at": now_iso,
            }

        cleaned_system_competitors = sorted(
            merged.values(),
            key=lambda entry: float(entry.get("score") or 0),
            reverse=True,
        )[:MAX_SYSTEM_COMPETITORS]
        set_fields["systemCompetitors"] = cleaned_system_competitors

    query: dict
    if business_id:
        try:
            query = {"_id": ObjectId(business_id), "user_id": current_user["id"]}
        except Exception:
            query = {"user_id": current_user["id"], "normalized_domain": normalized_domain}
    else:
        query = {"user_id": current_user["id"], "normalized_domain": normalized_domain}

    # Score history — one point per real scan, not one per ISO week.
    # This used to find-or-create a bucket keyed by week_id and overwrite
    # its score in place. That silently erased history: the Sunday
    # scheduler and a manual re-scan later the same week both land in the
    # same week_id, so the second write just overwrote the first entry's
    # score — only one of the two scans ever survived to show on the trend
    # chart, under whichever timestamp happened to be on that shared entry.
    # Every scan (manual or scheduler) now gets appended as its own point
    # with its own real timestamp, so the chart can show exactly what ran
    # and when instead of collapsing same-week runs into one number.
    if phase1_score is not None:
        dt = datetime.utcnow()
        week_id = f"{dt.isocalendar().year}-W{dt.isocalendar().week:02d}"

        existing = await businesses_col.find_one(query)
        weekly_scores = (existing.get("weekly_scores") if existing else []) or []

        # Guard only against true accidental duplicates (e.g. a frontend
        # double-submit firing the same save twice within seconds) — not
        # against a second real scan later the same day/week.
        last_entry = weekly_scores[-1] if weekly_scores else None
        is_accidental_duplicate = False
        if last_entry and last_entry.get("score") == int(phase1_score) and last_entry.get("created_at"):
            try:
                # now_iso carries a trailing "Z" that fromisoformat() only
                # accepts on Python 3.11+ — stripped here to parse safely
                # regardless of runtime version.
                last_dt = datetime.fromisoformat(str(last_entry["created_at"]).rstrip("Z"))
                is_accidental_duplicate = (dt - last_dt).total_seconds() < 120
            except Exception:
                is_accidental_duplicate = False

        if is_accidental_duplicate:
            last_entry["updated_at"] = now_iso
        else:
            weekly_scores.append({
                "week_id": week_id,
                "score": int(phase1_score),
                "created_at": now_iso,
                "updated_at": now_iso,
            })

        # Cap by point count, not by "weeks", now that a week can hold more
        # than one point — generous enough for even a daily scanner.
        if len(weekly_scores) > 200:
            weekly_scores = weekly_scores[-200:]

        set_fields["weekly_scores"] = weekly_scores

    # Same append-only history pattern as weekly_scores, but for the
    # visibility score (mention rate/position/citation — the same formula
    # competitors are scored on). This is the score reconciliation fix: the
    # Dashboard headline "Wonder Score" now reads from THIS history, not
    # weekly_scores, so a competitor comparison is always apples-to-apples —
    # weekly_scores (Phase 1 technical) becomes a labeled sub-component
    # instead of competing as its own standalone number.
    if phase5_score is not None:
        dt5 = datetime.utcnow()
        week_id5 = f"{dt5.isocalendar().year}-W{dt5.isocalendar().week:02d}"

        # Reuses the `existing` doc already fetched at the top of this
        # function (before any updates were applied) — same source the
        # phase1_score branch above re-fetches fresh for weekly_scores; this
        # one doesn't need to since visibility_weekly_scores can't have
        # changed between that fetch and here within a single call.
        visibility_weekly_scores = (existing.get("visibility_weekly_scores") if existing else []) or []

        last_entry5 = visibility_weekly_scores[-1] if visibility_weekly_scores else None
        is_accidental_duplicate5 = False
        if last_entry5 and last_entry5.get("score") == float(phase5_score) and last_entry5.get("created_at"):
            try:
                last_dt5 = datetime.fromisoformat(str(last_entry5["created_at"]).rstrip("Z"))
                is_accidental_duplicate5 = (dt5 - last_dt5).total_seconds() < 120
            except Exception:
                is_accidental_duplicate5 = False

        if is_accidental_duplicate5:
            last_entry5["updated_at"] = now_iso
        else:
            visibility_weekly_scores.append({
                "week_id": week_id5,
                "score": float(phase5_score),
                "created_at": now_iso,
                "updated_at": now_iso,
            })

        if len(visibility_weekly_scores) > 200:
            visibility_weekly_scores = visibility_weekly_scores[-200:]

        set_fields["visibility_weekly_scores"] = visibility_weekly_scores

    update_doc = {
        "$set": set_fields,
        "$setOnInsert": {
            "created_at": now_iso,
        },
    }

    return await businesses_col.find_one_and_update(
        query,
        update_doc,
        upsert=True,
        return_document=ReturnDocument.AFTER,
    )


def _iso_from_range(range_value: str) -> str | None:
    rv = (range_value or "").strip().lower()
    if rv in {"", "all"}:
        return None
    now = datetime.utcnow()
    if rv == "week":
        return (now - timedelta(days=7)).isoformat()
    if rv == "month":
        return (now - timedelta(days=30)).isoformat()
    if rv == "year":
        return (now - timedelta(days=365)).isoformat()
    return None


def _to_datetime(iso_value: str | None) -> datetime | None:
    if not iso_value:
        return None
    try:
        return datetime.fromisoformat(str(iso_value).replace("Z", "+00:00"))
    except Exception:
        return None


def _blog_clamp(value: int | float, min_value: int = 0, max_value: int = 100) -> int:
    return int(max(min_value, min(max_value, value)))


def _blog_split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"[.!?]+", re.sub(r"\s+", " ", text or "")) if s.strip()]


def _blog_count_matches(text: str, pattern: str) -> int:
    return len(re.findall(pattern, text or "", flags=re.I))


def _build_blog_base_analysis(text: str, attachment_count: int) -> dict:
    cleaned = (text or "").strip()
    words = [w for w in re.split(r"\s+", cleaned) if w]
    word_count = len(words)
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", cleaned) if p.strip()]
    paragraph_count = len(paragraphs)
    lines = [line.strip() for line in cleaned.split("\n") if line.strip()]
    heading_count = len([
        line for line in lines
        if re.match(r"^#{1,3}\s+", line) or re.match(r"^(?:[A-Z][A-Za-z0-9 ,:&-]{5,80})\??$", line)
    ])
    list_count = len([line for line in lines if re.match(r"^(-|\*|\d+\.)\s+", line)])
    link_count = _blog_count_matches(cleaned, r"https?://") + _blog_count_matches(cleaned, r"\[[^\]]+\]\([^\)]+\)")
    cta_count = _blog_count_matches(cleaned.lower(), r"\b(learn more|read more|sign up|subscribe|download|book a demo|get started|contact us|buy now|try it now)\b")

    sentence_lengths = [len([w for w in s.split() if w]) for s in _blog_split_sentences(cleaned)]
    avg_sentence_length = (sum(sentence_lengths) / len(sentence_lengths)) if sentence_lengths else 0

    structure_raw = 10
    structure_raw += _blog_clamp(heading_count * 4, 0, 16)
    structure_raw += _blog_clamp(list_count * 2, 0, 8)
    structure_raw += _blog_clamp(8 if paragraph_count >= 4 else paragraph_count * 2, 0, 8)
    structure_raw += 8 if word_count >= 700 else 4 if word_count >= 400 else 0
    structure_raw += 4 if attachment_count > 0 else 0
    structure_raw = _blog_clamp(structure_raw, 0, 35)

    readability_raw = 10
    if avg_sentence_length > 0:
        if 12 <= avg_sentence_length <= 22:
            readability_raw += 18
        elif 8 <= avg_sentence_length <= 28:
            readability_raw += 12
        else:
            readability_raw += 4
    readability_raw += 10 if paragraph_count >= 4 else 6 if paragraph_count >= 2 else 2
    readability_raw += 8 if word_count >= 500 else 5 if word_count >= 250 else 1
    readability_raw = _blog_clamp(readability_raw, 0, 30)

    seo_raw = 10
    seo_raw += 10 if heading_count > 0 else 0
    seo_raw += 6 if link_count > 0 else 0
    seo_raw += 8 if word_count >= 800 else 5 if word_count >= 500 else 1
    seo_raw += 4 if attachment_count > 0 else 0
    seo_raw = _blog_clamp(seo_raw, 0, 30)

    engagement_raw = 8
    engagement_raw += 10 if cta_count > 0 else 0
    engagement_raw += 8 if list_count > 0 else 0
    engagement_raw += 4 if _blog_count_matches(cleaned, r"\?") > 2 else 0
    engagement_raw += 4 if _blog_count_matches(cleaned, r"\b(example|tip|guide|step|how to|why|benefit)\b") > 2 else 0
    engagement_raw = _blog_clamp(engagement_raw, 0, 20)

    structure = _blog_clamp(round((structure_raw / 35) * 100), 0, 100)
    readability = _blog_clamp(round((readability_raw / 30) * 100), 0, 100)
    seo = _blog_clamp(round((seo_raw / 30) * 100), 0, 100)
    engagement = _blog_clamp(round((engagement_raw / 20) * 100), 0, 100)

    # Normalize component totals (max 115) to a 0-100 scale.
    total = structure_raw + readability_raw + seo_raw + engagement_raw
    score = _blog_clamp(round((total / 115) * 100), 0, 100)
    if word_count < 150:
        score = max(18, score - 12)
    elif word_count < 350:
        score = max(28, score - 6)
    if attachment_count > 0:
        score = _blog_clamp(score + 3, 0, 100)

    strengths = []
    if heading_count > 0:
        strengths.append("Clear sectioning with headings")
    if list_count > 0:
        strengths.append("Easy-to-scan list formatting")
    if link_count > 0:
        strengths.append("Includes useful references or links")
    if cta_count > 0:
        strengths.append("Has a conversion-focused CTA")
    if attachment_count > 0:
        strengths.append("Supported by an attached PDF reference")

    suggestions = []
    if word_count < 500:
        suggestions.append(
            "Expand the article to at least 700 to 1,000 words if this is meant to rank competitively."
        )
    if heading_count < 2:
        suggestions.append(
            "Add more H2 and H3 sections so search engines and readers can scan the content faster."
        )
    if link_count == 0:
        suggestions.append(
            "Add internal or external links to strengthen topical authority and help users explore more."
        )
    if list_count == 0:
        suggestions.append(
            "Break dense sections into bullets or numbered steps to improve readability."
        )
    if cta_count == 0:
        suggestions.append(
            "Add a stronger CTA at the end to guide the reader toward the next action."
        )
    if paragraph_count < 4:
        suggestions.append(
            "Split long walls of text into shorter paragraphs for better mobile readability."
        )
    if attachment_count > 0:
        suggestions.append(
            "Use the attached PDF to reinforce claims, cite statistics, or support the article with original evidence."
        )
    if not suggestions:
        suggestions.append(
            "The blog is already in good shape. Consider a light title optimization and one extra supporting link for polish."
        )

    grade = "Needs Work"
    if score >= 90:
        grade = "Exceptional"
    elif score >= 80:
        grade = "Strong"
    elif score >= 70:
        grade = "Good"
    elif score >= 55:
        grade = "Fair"

    overview = (
        "This article is structured well enough to compete, with only a few optimization gaps left to close."
        if score >= 80
        else "This draft has a workable base, but it needs clearer structure and stronger SEO signals before it can compete well."
        if score >= 55
        else "This draft needs more structure, clearer search intent coverage, and stronger on-page SEO signals before it is competitive."
    )

    engagement_label = "high" if engagement >= 60 else "moderate"
    summary = (
        f"A {word_count}-word {grade.lower()} draft. It features {heading_count} headings and {link_count} links, aiming for a {engagement_label} level of reader engagement."
        if word_count > 0
        else "An empty or very short draft that requires more content to analyze effectively."
    )

    weak_spots = [
        s for s in suggestions
        if "Expand" in s or "H2 and H3" in s or "Split long walls" in s
    ]
    improvements = [s for s in suggestions if s not in weak_spots]

    return {
        "metrics": {
            "wordCount": word_count,
            "paragraphCount": paragraph_count,
            "headingCount": heading_count,
            "listCount": list_count,
            "linkCount": link_count,
            "ctaCount": cta_count,
            "readability": readability,
            "structure": structure,
            "seo": seo,
            "engagement": engagement,
            "score": score,
            "grade": grade,
            "attachmentCount": attachment_count,
        },
        "strengths": strengths,
        "suggestions": suggestions,
        "overview": overview,
        "summary": summary,
        "weakSpots": weak_spots,
        "improvements": improvements,
    }

@app.post("/api/auth/google", response_model=Token)
async def api_google_login(request: GoogleAuthRequest):
    try:
        client_id = os.getenv("GOOGLE_CLIENT_ID")
        if not client_id:
            raise HTTPException(status_code=500, detail="Google Client ID not configured")
            
        idinfo = id_token.verify_oauth2_token(
            request.credential, google_requests.Request(), client_id
        )

        email = idinfo.get("email")
        name = idinfo.get("name")
        
        if not email:
            raise HTTPException(status_code=400, detail="Invalid Google token payload")

        user = await users_col.find_one({"email": email})

        if not user:
            total_users = await users_col.count_documents({})
            assigned_role = "admin" if total_users == 0 else "user"

            hashed_password = get_password_hash(secrets.token_urlsafe(32))

            new_user = {
                "name": name,
                "email": email,
                "hashed_password": hashed_password,
                "role": assigned_role,
                "status": "active",
                # Google has already verified this email address just by
                # letting the OAuth flow complete — there is no OTP step for
                # this path to ever mark it true later, so it must be set
                # here or the dashboard's route guard (needsVerification in
                # DashboardLayout) redirects every Google user to a
                # /verify-email screen expecting a code that was never sent.
                "email_verified": True,
                "created_at": datetime.utcnow().isoformat()
            }
            result = await users_col.insert_one(new_user)
            user_id = str(result.inserted_id)
            user_role = assigned_role
            user_status = "active"
            created_at = new_user["created_at"]
            email_verified = True
            # Google sign-in skips the OTP flow entirely (Google already
            # verified the email), so this is the only signal that this is
            # a brand-new account rather than a returning user — fire the
            # same welcome email the OTP signup path sends.
            asyncio.create_task(send_welcome_email(email, name))
        else:
            user_id = str(user["_id"])
            user_role = user.get("role", "user")
            user_status = user.get("status", "active")
            created_at = user.get("created_at", datetime.utcnow().isoformat())
            name = user.get("name", name)
            email_verified = bool(user.get("email_verified"))
            if not email_verified:
                # Self-heals any account already stuck from before this fix
                # (created via this same endpoint with no email_verified
                # field at all) the next time they sign in with Google.
                await users_col.update_one({"_id": user["_id"]}, {"$set": {"email_verified": True}})
                email_verified = True

        if user_status == "banned":
            raise HTTPException(status_code=403, detail="Your account has been restricted.")

        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": email, "id": user_id}, expires_delta=access_token_expires
        )

        user_response = UserResponse(
            id=user_id,
            name=name,
            email=email,
            created_at=created_at,
            role=user_role,
            status=user_status,
            email_verified=email_verified,
        )

        return Token(access_token=access_token, token_type="bearer", user=user_response)

    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid Google token")
    except HTTPException:
        raise
    except Exception as e:
        print(f"[API] ERROR during google auth: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal server error")


# ---------------------------------------------------------------------------
# J6: scan runs. A scan run groups the legs of one measurement of a business
# (website crawl, AI tracker, competitors, plan) under ONE status. A Wonder
# Score is written to the business ONLY when the run is complete, so the
# number never changes mid-run and a half-measured score can never be saved
# (A1 formula: agents/phase5/scoring.py).
# ---------------------------------------------------------------------------
SCAN_RUN_STALE_AFTER_SECONDS = 3 * 3600
WONDER_HISTORY_CAP = 200


def _scan_run_now() -> str:
    return datetime.utcnow().isoformat() + "Z"


async def _scan_run_create(*, business_id, user_id, url, trigger: str, legs: list[str]) -> str | None:
    """Start a run. Returns None (no run, no score) for anonymous / no-business
    callers such as the free public check."""
    if scan_runs_col is None or not business_id or not user_id:
        return None
    run_id = uuid.uuid4().hex
    now = _scan_run_now()
    try:
        await scan_runs_col.insert_one({
            "run_id": run_id,
            "business_id": str(business_id),
            "user_id": str(user_id),
            "url": url,
            "trigger": trigger,
            "status": "running",
            "legs": {name: {"status": "pending"} for name in legs},
            "inputs": None,
            "wonder_score": None,
            "created_at": now,
            "updated_at": now,
            "completed_at": None,
        })
    except Exception as e:
        print(f"[ScanRun] create failed: {e}")
        return None
    return run_id


async def _scan_run_set_leg(run_id, leg: str, status: str, **fields) -> None:
    """status: pending | running | ok | failed | incomplete | skipped"""
    if scan_runs_col is None or not run_id:
        return
    set_doc = {f"legs.{leg}.status": status, "updated_at": _scan_run_now()}
    for key, value in fields.items():
        set_doc[f"legs.{leg}.{key}"] = value
    try:
        await scan_runs_col.update_one({"run_id": run_id}, {"$set": set_doc})
    except Exception as e:
        print(f"[ScanRun] leg update failed run={run_id} leg={leg}: {e}")


async def _scan_run_restart(run_id) -> None:
    """A retry re-opens a finished run: it is running again until re-finalized."""
    if scan_runs_col is None or not run_id:
        return
    try:
        await scan_runs_col.update_one(
            {"run_id": run_id},
            {"$set": {"status": "running", "updated_at": _scan_run_now(), "completed_at": None}},
        )
    except Exception as e:
        print(f"[ScanRun] restart failed run={run_id}: {e}")


async def _latest_completed_tracker_mention(business_id: str) -> tuple:
    """(mention rate, its run-to-run margin, job id) of the business's latest
    COMPLETE multi-model tracker job - the tracker input for a run that did not
    re-run the tracker itself. (None, None, None) when there is none."""
    if phase5_jobs_col is None:
        return None, None, None
    job = await phase5_jobs_col.find_one(
        {"business_id": str(business_id), "job_type": "core", "model": "multi", "status": "completed"},
        sort=[("created_at", -1)],
        projection={"results": 1, "questions": 1, "job_id": 1},
    )
    if not job:
        return None, None, None
    results, questions = job.get("results") or {}, job.get("questions") or []
    return compute_mention_rate(results, questions), compute_mention_rate_margin(results, questions), job.get("job_id")


async def _latest_completed_tracker_mention_rate(business_id: str) -> float | None:
    return (await _latest_completed_tracker_mention(business_id))[0]


async def _notify(user_id, business_id, item: dict | None, dedupe_key: str) -> None:
    """J10: store one in-app notification. Best effort - never breaks the caller - and the
    same event (dedupe_key) is only ever stored once, so a re-finalized run does not repeat itself."""
    if notifications_col is None or not item or not user_id:
        return
    try:
        if await notifications_col.find_one({"user_id": str(user_id), "dedupe_key": dedupe_key}):
            return
        await notifications_col.insert_one({
            **item,
            "user_id": str(user_id),
            "business_id": str(business_id or ""),
            "dedupe_key": dedupe_key,
            "created_at": datetime.utcnow(),
            "read_at": None,
        })
    except Exception as e:
        print(f"[Notify] could not store {dedupe_key}: {e}")


async def _competitors_of_job(job_id) -> list:
    if phase5_jobs_col is None or not job_id:
        return []
    job = await phase5_jobs_col.find_one({"job_id": job_id}, projection={"deep_competitors": 1})
    return (job or {}).get("deep_competitors") or []


async def _emit_run_notifications(run: dict, business_name: str, score, previous_score, tracker_job_id) -> None:
    run_id, user_id, business_id = run.get("run_id"), run.get("user_id"), run.get("business_id")
    await _notify(user_id, business_id, notif.scan_complete(business_name, score), f"scan_complete:{run_id}")
    await _notify(user_id, business_id, notif.score_moved(business_name, previous_score, score), f"score_moved:{run_id}")
    if tracker_job_id and phase5_jobs_col is not None:
        try:
            prev_job = await phase5_jobs_col.find_one(
                {"business_id": str(business_id), "job_type": "core", "model": "multi", "status": "completed",
                 "job_id": {"$ne": tracker_job_id}},
                sort=[("created_at", -1)], projection={"deep_competitors": 1},
            )
            names = notif.overtakers((prev_job or {}).get("deep_competitors"), await _competitors_of_job(tracker_job_id))
            await _notify(user_id, business_id, notif.competitor_overtook(business_name, names), f"overtook:{run_id}")
        except Exception as e:
            print(f"[Notify] competitor check failed run={run_id}: {e}")


async def _scan_run_finalize(run_id) -> dict | None:
    """Decide the run's final status and, only if it is COMPLETE and every input
    exists, compute + store the Wonder Score. Safe to call again after a retry:
    the history entry for a run is replaced, never duplicated."""
    if scan_runs_col is None or not run_id:
        return None
    run = await scan_runs_col.find_one({"run_id": run_id})
    if not run:
        return None

    legs = run.get("legs") or {}
    # Only the measuring legs decide the run's status. The plan (blogs) is
    # recorded but a failed draft must never block the score.
    attempted = {
        n: l for n, l in legs.items()
        if n in ("crawl", "tracker", "competitors") and (l or {}).get("status") not in ("pending", "skipped")
    }
    states = [str((l or {}).get("status")) for l in attempted.values()]
    bad = [s for s in states if s in ("failed", "incomplete")]
    if any(s == "running" for s in states):
        return run  # a leg is still in flight; its owner finalizes later
    if not attempted or (bad and len(bad) == len(states)):
        status = "failed"
    elif bad:
        status = "incomplete"
    else:
        status = "complete"

    business_id = str(run.get("business_id") or "")
    inputs = None
    score = None
    tracker_job_id = None
    previous_score = None
    business_name = ""
    if status == "complete":
        crawl = legs.get("crawl") or {}
        tracker = legs.get("tracker") or {}
        competitors = legs.get("competitors") or {}
        technical = crawl.get("score") if crawl.get("status") == "ok" else None
        if technical is None and "crawl" not in attempted and businesses_col is not None:
            try:
                biz = await businesses_col.find_one({"_id": ObjectId(business_id)}, {"latest_phase1_score": 1})
                technical = (biz or {}).get("latest_phase1_score")
            except Exception:
                technical = None
        mention_rate = tracker.get("mention_rate") if tracker.get("status") == "ok" else None
        mention_margin = tracker.get("mention_rate_margin") if tracker.get("status") == "ok" else None
        # E1: remember WHICH Search Tracker run this score was built from, so a
        # shared report can show that run's per-model data and diff it properly
        tracker_job_id = tracker.get("job_id") if tracker.get("status") == "ok" else None
        if mention_rate is None and "tracker" not in attempted:
            mention_rate, mention_margin, tracker_job_id = await _latest_completed_tracker_mention(business_id)
        market = competitors.get("market_position") if competitors.get("status") == "ok" else None
        if market is None and "tracker" not in attempted and tracker_job_id and phase5_jobs_col is not None:
            stored_job = await phase5_jobs_col.find_one({"job_id": tracker_job_id}, projection={"market_ladder": 1})
            ladder_doc = (stored_job or {}).get("market_ladder") or {}
            if ladder_doc.get("measured"):
                market = ladder_doc.get("position")
        inputs = {
            "mention_rate": mention_rate,
            "technical_score": technical,
            # A2 has not shipped: 0, shown as "not yet measured", until it does
            "market_position": market if isinstance(market, (int, float)) else 0,
            "market_measured": isinstance(market, (int, float)),
            # A5: how far the score can move between two runs just because the
            # AI models answer differently each time (+/- Wonder Score points)
            "score_margin": wonder_margin(mention_margin),
        }
        score = compute_wonder_score(mention_rate, technical, inputs["market_position"])

    now = _scan_run_now()
    await scan_runs_col.update_one(
        {"run_id": run_id},
        {"$set": {"status": status, "inputs": inputs, "wonder_score": score, "tracker_job_id": tracker_job_id,
                  "updated_at": now, "completed_at": now}},
    )

    if score is not None and businesses_col is not None:
        try:
            biz = await businesses_col.find_one({"_id": ObjectId(business_id)}, {"wonder_scores": 1})
            history = [h for h in ((biz or {}).get("wonder_scores") or []) if h.get("run_id") != run_id]
            previous_score = history[-1].get("score") if history else None
            history.append({
                "run_id": run_id,
                "score": score,
                "inputs": inputs,
                "tracker_job_id": tracker_job_id,
                "trigger": run.get("trigger"),
                "week_id": _blog_week_id(datetime.now()),
                "created_at": now,
            })
            await businesses_col.update_one(
                {"_id": ObjectId(business_id)},
                {"$set": {
                    "latest_wonder_score": score,
                    "latest_wonder_at": now,
                    "latest_wonder_run_id": run_id,
                    "latest_wonder_inputs": inputs,
                    "wonder_scores": history[-WONDER_HISTORY_CAP:],
                }},
            )
        except Exception as e:
            print(f"[ScanRun] storing wonder score failed run={run_id}: {e}")

    if status == "complete":
        try:
            if businesses_col is not None:
                b = await businesses_col.find_one({"_id": ObjectId(business_id)}, {"businessName": 1})
                business_name = (b or {}).get("businessName") or ""
        except Exception:
            business_name = ""
        await _emit_run_notifications(run, business_name, score, previous_score, tracker_job_id)

    print(f"[ScanRun] run={run_id} trigger={run.get('trigger')} -> {status}, wonder_score={score}")
    return await scan_runs_col.find_one({"run_id": run_id})


async def _scan_run_active_for(business_ids: list) -> dict:
    """business_id -> its in-flight run (if any), ignoring stale ones, so the UI
    can show "Calculating…" while keeping the last complete score visible."""
    if scan_runs_col is None or not business_ids:
        return {}
    cutoff = (datetime.utcnow() - timedelta(seconds=SCAN_RUN_STALE_AFTER_SECONDS)).isoformat() + "Z"
    out: dict = {}
    cursor = scan_runs_col.find(
        {"business_id": {"$in": [str(b) for b in business_ids]}, "status": "running", "updated_at": {"$gte": cutoff}},
        {"_id": 0, "run_id": 1, "business_id": 1, "trigger": 1, "created_at": 1},
    ).sort("created_at", -1)
    async for run in cursor:
        out.setdefault(run["business_id"], {"run_id": run["run_id"], "trigger": run.get("trigger"), "started_at": run.get("created_at")})
    return out


def _fetch_failure(scrape_result) -> dict | None:
    """L1: the scraper's verdict when it could NOT get the website ({ok: False,
    http_status, reason}), else None. A result without the verdict (older
    cached results) is trusted as before."""
    info = scrape_result.get("fetch") if isinstance(scrape_result, dict) else None
    if isinstance(info, dict) and info.get("ok") is False:
        return info
    return None


def _phase1_score_of(scrape_result):
    """The website-health score to store - None when the scan did not really
    reach the website, so an error page never becomes a score."""
    if not isinstance(scrape_result, dict) or _fetch_failure(scrape_result):
        return None
    scores = scrape_result.get("scores")
    return scores.get("total") if isinstance(scores, dict) else None


async def _record_last_crawl(business_id, scrape_result) -> None:
    """Remember how the latest website scan went on the business, so the dashboard
    can say "Score not available - website scan failed" instead of a stale number."""
    if businesses_col is None or not business_id or not isinstance(scrape_result, dict):
        return
    failure = _fetch_failure(scrape_result)
    info = failure or {"ok": True, "http_status": (scrape_result.get("fetch") or {}).get("http_status"), "reason": None}
    try:
        await businesses_col.update_one(
            {"_id": ObjectId(str(business_id))},
            {"$set": {"last_crawl": {**info, "at": _scan_run_now()}}},
        )
    except Exception as e:
        print(f"[ScanRun] could not store last_crawl for {business_id}: {e}")


async def _record_manual_crawl_run(current_user, business, scrape_result) -> None:
    """A manual Analyzer crawl is a one-leg scan run. When it completes, the
    Wonder Score is recomputed from this fresh website health plus the latest
    complete tracker run - so re-crawling the Analyzer finally moves the main
    score (it used to be saved where the headline never looked)."""
    try:
        if not business or not current_user or not isinstance(scrape_result, dict):
            return
        scores = scrape_result.get("scores") if isinstance(scrape_result.get("scores"), dict) else {}
        run_id = await _scan_run_create(
            business_id=str(business.get("_id")), user_id=current_user.get("id"),
            url=business.get("url") or "", trigger="manual_analyzer", legs=["crawl"],
        )
        failure = _fetch_failure(scrape_result)
        await _record_last_crawl(business.get("_id"), scrape_result)
        if failure:
            print(f"[ScanRun] website scan failed for {business.get('url')}: HTTP {failure.get('http_status')} - {failure.get('reason')}")
            await _scan_run_set_leg(run_id, "crawl", "failed", error=failure.get("reason"), http_status=failure.get("http_status"))
        else:
            await _scan_run_set_leg(run_id, "crawl", "ok", score=scores.get("total"))
        await _scan_run_finalize(run_id)
    except Exception as e:
        print(f"[ScanRun] manual crawl run failed: {e}")


async def _ensure_wonder_scores(docs: list) -> None:
    """Businesses measured before the scan-run system have no Wonder Score
    until their next complete run. Compute one from what is already stored
    (latest website-health score + latest COMPLETE tracker run), once, so
    existing accounts don't go blank at launch. A business missing either
    input stays without a score - never a guessed one."""
    if businesses_col is None:
        return
    for doc in docs:
        if not isinstance(doc, dict) or isinstance(doc.get("latest_wonder_score"), (int, float)):
            continue
        technical = doc.get("latest_phase1_score")
        if not isinstance(technical, (int, float)):
            continue
        mention_rate, mention_margin, _backfill_job_id = await _latest_completed_tracker_mention(str(doc.get("_id")))
        score = compute_wonder_score(mention_rate, technical, 0)
        if score is None:
            continue
        stamp = max(str(doc.get("latest_phase1_at") or ""), str(doc.get("latest_phase5_at") or "")) or _scan_run_now()
        inputs = {"mention_rate": mention_rate, "technical_score": technical, "market_position": 0, "market_measured": False,
                  "score_margin": wonder_margin(mention_margin)}
        point = {"run_id": "backfill", "score": score, "inputs": inputs, "trigger": "backfill", "tracker_job_id": _backfill_job_id,
                 "week_id": None, "created_at": stamp}
        try:
            await businesses_col.update_one(
                {"_id": doc["_id"], "latest_wonder_score": {"$exists": False}},
                {"$set": {"latest_wonder_score": score, "latest_wonder_at": stamp, "latest_wonder_run_id": "backfill",
                          "latest_wonder_inputs": inputs, "wonder_scores": [point]}},
            )
        except Exception as e:
            print(f"[WonderScore] backfill failed for {doc.get('_id')}: {e}")
            continue
        doc["latest_wonder_score"], doc["latest_wonder_at"], doc["latest_wonder_inputs"] = score, stamp, inputs

