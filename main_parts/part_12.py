# Google Analytics and Search Console integrations.
#
# This chunk is deliberately self-contained so dashboard auth stays separate from
# Google data access consent. Existing Google login uses an ID token; these
# endpoints use OAuth code flow because Analytics/Search Console need refreshable
# read-only API tokens.
from fastapi.responses import RedirectResponse
from urllib.parse import urlencode, quote
import base64
import httpx
from cryptography.fernet import Fernet


GOOGLE_INTEGRATION_SCOPES = [
    "openid",
    "email",
    "https://www.googleapis.com/auth/analytics.readonly",
    "https://www.googleapis.com/auth/webmasters.readonly",
]


class GoogleIntegrationMapRequest(BaseModel):
    business_id: str | None = None
    ga_property_id: str | None = None
    search_console_site_url: str | None = None


def _google_integration_client_id() -> str:
    return (
        os.getenv("GOOGLE_INTEGRATION_CLIENT_ID")
        or os.getenv("GOOGLE_CLIENT_ID")
        or os.getenv("NEXT_PUBLIC_GOOGLE_CLIENT_ID")
        or ""
    ).strip()


def _google_integration_client_secret() -> str:
    return (os.getenv("GOOGLE_INTEGRATION_CLIENT_SECRET") or os.getenv("GOOGLE_CLIENT_SECRET") or "").strip()


def _google_integration_redirect_uri() -> str:
    return (
        os.getenv("GOOGLE_INTEGRATION_REDIRECT_URI")
        or "http://127.0.0.1:8000/api/integrations/google/callback"
    ).strip()


def _frontend_app_url() -> str:
    return (os.getenv("FRONTEND_APP_URL") or "http://localhost:3000").rstrip("/")


def _integration_cipher() -> Fernet:
    raw_key = (os.getenv("GOOGLE_INTEGRATION_TOKEN_KEY") or "").strip()
    if raw_key:
        padded_key = raw_key + ("=" * (-len(raw_key) % 4))
        return Fernet(padded_key.encode("utf-8"))
    # Dev-friendly fallback: still encrypt tokens at rest using the app secret.
    derived = base64.urlsafe_b64encode(hashlib.sha256(SECRET_KEY.encode("utf-8")).digest())
    return Fernet(derived)


def _encrypt_google_token(value: str | None) -> str | None:
    if not value:
        return None
    return _integration_cipher().encrypt(value.encode("utf-8")).decode("utf-8")


def _decrypt_google_token(value: str | None) -> str | None:
    if not value:
        return None
    return _integration_cipher().decrypt(value.encode("utf-8")).decode("utf-8")


def _public_google_integration(doc: dict | None) -> dict:
    if not doc:
        return {
            "connected": False,
            "configured": False,
            "email": None,
            "scopes": GOOGLE_INTEGRATION_SCOPES,
            "defaultMapping": None,
            "mappings": {},
        }
    return {
        "connected": True,
        "configured": bool(doc.get("default_mapping") or doc.get("mappings")),
        "email": doc.get("google_email"),
        "scopes": doc.get("scopes") or GOOGLE_INTEGRATION_SCOPES,
        "defaultMapping": doc.get("default_mapping"),
        "mappings": doc.get("mappings") if isinstance(doc.get("mappings"), dict) else {},
        "updated_at": doc.get("updated_at"),
        "created_at": doc.get("created_at"),
    }


async def _get_google_access_token(integration: dict) -> str:
    expires_at = integration.get("expires_at")
    now = datetime.utcnow()
    if isinstance(expires_at, datetime) and expires_at > now + timedelta(minutes=2):
        token = _decrypt_google_token(integration.get("access_token"))
        if token:
            return token

    refresh_token = _decrypt_google_token(integration.get("refresh_token"))
    if not refresh_token:
        raise HTTPException(status_code=409, detail="Google access expired. Please reconnect Google.")

    client_id = _google_integration_client_id()
    client_secret = _google_integration_client_secret()
    if not client_id or not client_secret:
        raise HTTPException(status_code=503, detail="Google integration is not configured on the server.")

    async with httpx.AsyncClient(timeout=20) as client:
        token_res = await client.post(
            "https://oauth2.googleapis.com/token",
            data={
                "client_id": client_id,
                "client_secret": client_secret,
                "refresh_token": refresh_token,
                "grant_type": "refresh_token",
            },
        )
    if token_res.status_code >= 400:
        raise HTTPException(status_code=409, detail="Google token refresh failed. Please reconnect Google.")
    token_data = token_res.json()
    access_token = token_data.get("access_token")
    if not access_token:
        raise HTTPException(status_code=409, detail="Google did not return an access token.")

    expires_in = int(token_data.get("expires_in") or 3600)
    await google_integrations_col.update_one(
        {"_id": integration["_id"]},
        {
            "$set": {
                "access_token": _encrypt_google_token(access_token),
                "expires_at": now + timedelta(seconds=expires_in),
                "updated_at": now.isoformat(),
            }
        },
    )
    return access_token


def _integration_mapping_for_business(integration: dict | None, business_id: str | None) -> dict:
    if not integration:
        return {}
    mappings = integration.get("mappings") if isinstance(integration.get("mappings"), dict) else {}
    if business_id and isinstance(mappings.get(business_id), dict):
        return mappings[business_id]
    return integration.get("default_mapping") if isinstance(integration.get("default_mapping"), dict) else {}


@app.get("/api/integrations/google/status")
async def api_google_integration_status(current_user: dict = Depends(get_current_user)):
    if google_integrations_col is None:
        raise HTTPException(status_code=503, detail="integration storage unavailable")
    doc = await google_integrations_col.find_one({"user_id": current_user["id"]})
    config_ready = bool(_google_integration_client_id() and _google_integration_client_secret())
    return {
        **_public_google_integration(doc),
        "serverConfigured": config_ready,
    }


@app.get("/api/integrations/google/connect")
async def api_google_integration_connect(current_user: dict = Depends(get_current_user)):
    client_id = _google_integration_client_id()
    client_secret = _google_integration_client_secret()
    if not client_id or not client_secret:
        raise HTTPException(status_code=503, detail="Add GOOGLE_INTEGRATION_CLIENT_ID and GOOGLE_INTEGRATION_CLIENT_SECRET first.")

    state = create_access_token(
        {"id": current_user["id"], "purpose": "google_integration"},
        expires_delta=timedelta(minutes=10),
    )
    params = {
        "client_id": client_id,
        "redirect_uri": _google_integration_redirect_uri(),
        "response_type": "code",
        "scope": " ".join(GOOGLE_INTEGRATION_SCOPES),
        "access_type": "offline",
        "prompt": "consent",
        "include_granted_scopes": "true",
        "state": state,
    }
    return {"authUrl": f"https://accounts.google.com/o/oauth2/v2/auth?{urlencode(params)}"}


@app.get("/api/integrations/google/callback")
async def api_google_integration_callback(code: str | None = None, state: str | None = None, error: str | None = None):
    frontend_url = _frontend_app_url()
    if error:
        return RedirectResponse(f"{frontend_url}/profile?integration=google_error")
    if not code or not state:
        return RedirectResponse(f"{frontend_url}/profile?integration=google_missing")

    try:
        payload = jwt.decode(state, SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("purpose") != "google_integration":
            raise ValueError("invalid purpose")
        user_id = payload.get("id")
    except Exception:
        return RedirectResponse(f"{frontend_url}/profile?integration=google_state_invalid")

    client_id = _google_integration_client_id()
    client_secret = _google_integration_client_secret()
    if not client_id or not client_secret:
        return RedirectResponse(f"{frontend_url}/profile?integration=google_not_configured")

    now = datetime.utcnow()
    async with httpx.AsyncClient(timeout=25) as client:
        token_res = await client.post(
            "https://oauth2.googleapis.com/token",
            data={
                "code": code,
                "client_id": client_id,
                "client_secret": client_secret,
                "redirect_uri": _google_integration_redirect_uri(),
                "grant_type": "authorization_code",
            },
        )
        if token_res.status_code >= 400:
            return RedirectResponse(f"{frontend_url}/profile?integration=google_token_error")
        token_data = token_res.json()
        access_token = token_data.get("access_token")
        refresh_token = token_data.get("refresh_token")
        userinfo = {}
        if access_token:
            try:
                info_res = await client.get(
                    "https://www.googleapis.com/oauth2/v2/userinfo",
                    headers={"Authorization": f"Bearer {access_token}"},
                )
                if info_res.status_code < 400:
                    userinfo = info_res.json()
            except Exception:
                userinfo = {}

    if google_integrations_col is None:
        return RedirectResponse(f"{frontend_url}/profile?integration=storage_unavailable")

    existing = await google_integrations_col.find_one({"user_id": user_id})
    encrypted_refresh = _encrypt_google_token(refresh_token) if refresh_token else existing.get("refresh_token") if existing else None
    await google_integrations_col.update_one(
        {"user_id": user_id},
        {
            "$set": {
                "user_id": user_id,
                "google_email": userinfo.get("email"),
                "access_token": _encrypt_google_token(access_token),
                "refresh_token": encrypted_refresh,
                "expires_at": now + timedelta(seconds=int(token_data.get("expires_in") or 3600)),
                "scopes": GOOGLE_INTEGRATION_SCOPES,
                "updated_at": now.isoformat(),
            },
            "$setOnInsert": {"created_at": now.isoformat()},
        },
        upsert=True,
    )
    return RedirectResponse(f"{frontend_url}/profile?integration=google_connected")


@app.get("/api/integrations/google/accounts")
async def api_google_integration_accounts(current_user: dict = Depends(get_current_user)):
    if google_integrations_col is None:
        raise HTTPException(status_code=503, detail="integration storage unavailable")
    integration = await google_integrations_col.find_one({"user_id": current_user["id"]})
    if not integration:
        return {"connected": False, "analyticsProperties": [], "searchConsoleSites": []}

    access_token = await _get_google_access_token(integration)
    headers = {"Authorization": f"Bearer {access_token}"}
    analytics_properties = []
    search_console_sites = []
    errors = []

    async with httpx.AsyncClient(timeout=25) as client:
        try:
            ga_res = await client.get("https://analyticsadmin.googleapis.com/v1beta/accountSummaries", headers=headers)
            if ga_res.status_code < 400:
                for account in ga_res.json().get("accountSummaries", []):
                    for prop in account.get("propertySummaries", []):
                        analytics_properties.append({
                            "account": account.get("displayName"),
                            "property": prop.get("property"),
                            "propertyId": str(prop.get("property", "")).split("/")[-1],
                            "displayName": prop.get("displayName"),
                        })
            else:
                errors.append("Could not load Google Analytics properties.")
        except Exception:
            errors.append("Could not load Google Analytics properties.")

        try:
            sc_res = await client.get("https://www.googleapis.com/webmasters/v3/sites", headers=headers)
            if sc_res.status_code < 400:
                for item in sc_res.json().get("siteEntry", []):
                    search_console_sites.append({
                        "siteUrl": item.get("siteUrl"),
                        "permissionLevel": item.get("permissionLevel"),
                    })
            else:
                errors.append("Could not load Search Console sites.")
        except Exception:
            errors.append("Could not load Search Console sites.")

    return {
        "connected": True,
        "analyticsProperties": analytics_properties,
        "searchConsoleSites": search_console_sites,
        "errors": errors,
    }


@app.post("/api/integrations/google/map")
async def api_google_integration_map(request: GoogleIntegrationMapRequest, current_user: dict = Depends(get_current_user)):
    if google_integrations_col is None:
        raise HTTPException(status_code=503, detail="integration storage unavailable")
    integration = await google_integrations_col.find_one({"user_id": current_user["id"]})
    if not integration:
        raise HTTPException(status_code=404, detail="Connect Google first.")

    mapping = {
        "gaPropertyId": (request.ga_property_id or "").strip() or None,
        "searchConsoleSiteUrl": (request.search_console_site_url or "").strip() or None,
        "updated_at": datetime.utcnow().isoformat(),
    }
    update = {"$set": {"default_mapping": mapping, "updated_at": datetime.utcnow().isoformat()}}
    if request.business_id:
        update["$set"][f"mappings.{request.business_id}"] = mapping
    await google_integrations_col.update_one({"user_id": current_user["id"]}, update)
    updated = await google_integrations_col.find_one({"user_id": current_user["id"]})
    return _public_google_integration(updated)


@app.post("/api/integrations/google/disconnect")
async def api_google_integration_disconnect(current_user: dict = Depends(get_current_user)):
    if google_integrations_col is None:
        raise HTTPException(status_code=503, detail="integration storage unavailable")
    await google_integrations_col.delete_one({"user_id": current_user["id"]})
    return {"connected": False, "message": "Google integration disconnected."}


@app.get("/api/analytics/summary")
async def api_analytics_summary(
    business_id: str | None = None,
    days: int = 28,
    current_user: dict = Depends(get_current_user),
):
    if google_integrations_col is None:
        raise HTTPException(status_code=503, detail="integration storage unavailable")
    days = max(7, min(90, int(days or 28)))
    integration = await google_integrations_col.find_one({"user_id": current_user["id"]})
    if not integration:
        return {
            "connected": False,
            "configured": False,
            "rangeDays": days,
            "summary": None,
            "message": "Connect Google Analytics and Search Console to see live analytics.",
        }

    mapping = _integration_mapping_for_business(integration, business_id)
    ga_property_id = (mapping.get("gaPropertyId") or "").strip()
    sc_site_url = (mapping.get("searchConsoleSiteUrl") or "").strip()
    if not ga_property_id and not sc_site_url:
        return {
            "connected": True,
            "configured": False,
            "rangeDays": days,
            "summary": None,
            "message": "Choose a Google Analytics property and Search Console site for this business.",
        }

    access_token = await _get_google_access_token(integration)
    headers = {"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"}
    summary = {
        "ga": None,
        "searchConsole": None,
    }
    errors = []

    async with httpx.AsyncClient(timeout=30) as client:
        if ga_property_id:
            try:
                ga_res = await client.post(
                    f"https://analyticsdata.googleapis.com/v1beta/properties/{ga_property_id}:runReport",
                    headers=headers,
                    json={
                        "dateRanges": [{"startDate": f"{days}daysAgo", "endDate": "today"}],
                        "metrics": [
                            {"name": "activeUsers"},
                            {"name": "sessions"},
                            {"name": "screenPageViews"},
                            {"name": "averageSessionDuration"},
                        ],
                    },
                )
                if ga_res.status_code < 400:
                    metric_values = (ga_res.json().get("rows") or [{}])[0].get("metricValues", [])
                    summary["ga"] = {
                        "activeUsers": float(metric_values[0].get("value", 0)) if len(metric_values) > 0 else 0,
                        "sessions": float(metric_values[1].get("value", 0)) if len(metric_values) > 1 else 0,
                        "pageViews": float(metric_values[2].get("value", 0)) if len(metric_values) > 2 else 0,
                        "averageSessionDuration": float(metric_values[3].get("value", 0)) if len(metric_values) > 3 else 0,
                    }
                else:
                    errors.append("Google Analytics report failed.")
            except Exception:
                errors.append("Google Analytics report failed.")

        if sc_site_url:
            try:
                encoded_site_url = quote(sc_site_url, safe="")
                sc_res = await client.post(
                    f"https://www.googleapis.com/webmasters/v3/sites/{encoded_site_url}/searchAnalytics/query",
                    headers=headers,
                    json={
                        "startDate": (datetime.utcnow() - timedelta(days=days)).date().isoformat(),
                        "endDate": datetime.utcnow().date().isoformat(),
                        "dimensions": ["query"],
                        "rowLimit": 10,
                    },
                )
                if sc_res.status_code < 400:
                    rows = sc_res.json().get("rows") or []
                    summary["searchConsole"] = {
                        "clicks": sum(float(row.get("clicks") or 0) for row in rows),
                        "impressions": sum(float(row.get("impressions") or 0) for row in rows),
                        "averageCtr": round(sum(float(row.get("ctr") or 0) for row in rows) / max(1, len(rows)), 4),
                        "averagePosition": round(sum(float(row.get("position") or 0) for row in rows) / max(1, len(rows)), 2),
                        "topQueries": [
                            {
                                "query": (row.get("keys") or [""])[0],
                                "clicks": row.get("clicks") or 0,
                                "impressions": row.get("impressions") or 0,
                                "position": row.get("position") or 0,
                            }
                            for row in rows
                        ],
                    }
                else:
                    errors.append("Search Console report failed.")
            except Exception:
                errors.append("Search Console report failed.")

    payload = {
        "connected": True,
        "configured": True,
        "rangeDays": days,
        "mapping": mapping,
        "summary": summary,
        "errors": errors,
        "generatedAt": datetime.utcnow().isoformat(),
    }
    if analytics_snapshots_col is not None:
        await analytics_snapshots_col.insert_one({
            "user_id": current_user["id"],
            "business_id": business_id,
            "payload": payload,
            "created_at": datetime.utcnow(),
        })
    return payload
