# Generated from the former backend/main.py lines 860-1269.
async def _upsert_user_business(
    *,
    current_user: dict | None,
    url: str,
    category: str | None = None,
    location: str | None = None,
    business_name: str | None = None,
    logo_url: str | None = None,
    business_description: str | None = None,
    ai_description: str | None = None,
    services: list[str] | None = None,
    target_audience: str | None = None,
    question_generation: dict | None = None,
    competitors: list[str] | None = None,
    system_competitors: list[dict] | None = None,
    tracked_pages: list[str] | None = None,
    phase1_score: int | None = None,
    phase5_score: float | None = None,
    business_id: str | None = None,
    scrape_result: dict | None = None,
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
    if phase1_score is not None:
        set_fields["latest_phase1_score"] = int(phase1_score)
        set_fields["latest_phase1_at"] = now_iso
    if phase5_score is not None:
        set_fields["latest_phase5_score"] = float(phase5_score)
        set_fields["latest_phase5_at"] = now_iso
    if scrape_result is not None:
        set_fields["latest_scrape_result"] = scrape_result
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
    if isinstance(system_competitors, list):
        cleaned_system_competitors: list[dict] = []
        seen_domains: set[str] = set()
        for item in system_competitors:
            if not isinstance(item, dict):
                continue
            domain = _normalize_site(str(item.get("domain") or item.get("url") or ""))
            if not domain or domain in seen_domains:
                continue
            seen_domains.add(domain)
            cleaned_system_competitors.append({
                "domain": domain,
                "url": item.get("url") or f"https://{domain}/",
                "name": _clean_optional_text(item.get("name")) or domain,
                "score": int(float(item.get("score") or 0)),
                "status": _clean_optional_text(item.get("status")) or _clean_optional_text(item.get("confidence")) or "AI evidence",
                "evidence": _clean_optional_text(item.get("evidence")) or "",
                "updated_at": now_iso,
            })
            if len(cleaned_system_competitors) >= 4:
                break
        set_fields["systemCompetitors"] = cleaned_system_competitors

    query: dict
    if business_id:
        try:
            query = {"_id": ObjectId(business_id), "user_id": current_user["id"]}
        except Exception:
            query = {"user_id": current_user["id"], "normalized_domain": normalized_domain}
    else:
        query = {"user_id": current_user["id"], "normalized_domain": normalized_domain}

    update_doc = {
        "$set": set_fields,
        "$setOnInsert": {
            "created_at": now_iso,
        },
    }
    if phase1_score is not None:
        update_doc["$push"] = {
            "scores_history": {
                "timestamp": now_iso,
                "score": int(phase1_score)
            }
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
                "created_at": datetime.utcnow().isoformat()
            }
            result = await users_col.insert_one(new_user)
            user_id = str(result.inserted_id)
            user_role = assigned_role
            user_status = "active"
            created_at = new_user["created_at"]
        else:
            user_id = str(user["_id"])
            user_role = user.get("role", "user")
            user_status = user.get("status", "active")
            created_at = user.get("created_at", datetime.utcnow().isoformat())
            name = user.get("name", name)

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
            status=user_status
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
