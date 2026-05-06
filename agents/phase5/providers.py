import asyncio
import os
import httpx
from google import genai
from google.genai import types
from utils.gemini_utils import generate_with_fallback

from .config import (
    OPENAI_PHASE5_TIMEOUT_SEC,
    PERPLEXITY_PHASE5_TIMEOUT_SEC,
    PHASE5_MODEL_CALL_TIMEOUT_SEC,
)
from .helpers import _extract_domains_from_text, _is_rate_limited_error, _normalize_domain, _safe_json_parse


_PROVIDER_STARTED_ONCE: set[str] = set()
_PROVIDER_HEALTHY_ONCE: set[str] = set()


class Phase5RateLimitError(RuntimeError):
    pass


def _log_provider_started_once(provider: str, model: str) -> None:
    key = f"{provider}:{model}"
    if key not in _PROVIDER_STARTED_ONCE:
        print(f"[Phase5][{provider}] model={model} started")
        _PROVIDER_STARTED_ONCE.add(key)


def _log_provider_healthy_once(provider: str, model: str, elapsed_sec: float) -> None:
    key = f"{provider}:{model}"
    if key not in _PROVIDER_HEALTHY_ONCE:
        print(f"[Phase5][{provider}] model={model} healthy (first success in {elapsed_sec:.2f}s)")
        _PROVIDER_HEALTHY_ONCE.add(key)


def get_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")
    return genai.Client(api_key=api_key)


def get_openai_api_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    return api_key


def get_openai_model_name() -> str:
    model = (os.getenv("OPENAI_MODEL_PHASE5") or "").strip()
    if not model:
        raise ValueError("OPENAI_MODEL_PHASE5 environment variable not set")
    return model


def get_perplexity_api_key() -> str:
    api_key = os.getenv("PERPLEXITY_API_KEY", "").strip()
    if not api_key:
        raise ValueError("PERPLEXITY_API_KEY environment variable not set")
    return api_key


def _extract_grounding_signals(response, target_domain: str) -> tuple[list[str], list[str], bool, int | None]:
    references: list[str] = []
    source_urls: list[str] = []
    target_mentioned = False
    position: int | None = None

    try:
        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            return references, source_urls, target_mentioned, position

        metadata = getattr(candidates[0], "grounding_metadata", None)
        chunks = getattr(metadata, "grounding_chunks", []) if metadata else []

        for idx, chunk in enumerate(chunks):
            web = getattr(chunk, "web", None)
            uri = getattr(web, "uri", "") if web else ""
            if not uri or "vertexaisearch.cloud.google.com" in uri:
                continue

            source_urls.append(uri)
            d = _normalize_domain(uri)
            if d:
                references.append(d)
                if d == _normalize_domain(target_domain) or d.endswith(f".{_normalize_domain(target_domain)}"):
                    target_mentioned = True
                    if position is None:
                        position = min(10, idx + 1)
    except Exception:
        pass

    return (
        list(dict.fromkeys(references)),
        list(dict.fromkeys(source_urls)),
        target_mentioned,
        position,
    )


async def _call_gemini_with_timeout(
    client: genai.Client,
    contents: str,
    config: types.GenerateContentConfig,
    timeout_sec: int | None = None,
):
    effective_timeout = max(8, timeout_sec if isinstance(timeout_sec, int) else PHASE5_MODEL_CALL_TIMEOUT_SEC)
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(
                generate_with_fallback,
                client,
                contents=contents,
                config=config,
            ),
            timeout=effective_timeout,
        )
    except asyncio.TimeoutError:
        print(f"[Phase5][Gemini] call timed out after {effective_timeout}s")
        return None
    except Exception as e:
        if _is_rate_limited_error(e):
            raise Phase5RateLimitError("Gemini rate limit reached. Please retry shortly.") from e
        print(f"[Phase5][Gemini] call failed: {type(e).__name__}: {e}")
        return None


async def _call_gemini_with_retry(
    client: genai.Client,
    contents: str,
    config: types.GenerateContentConfig,
    retry_once: bool = False,
    timeout_sec: int | None = None,
):
    response = await _call_gemini_with_timeout(client, contents, config, timeout_sec=timeout_sec)
    if response is not None:
        return response

    if not retry_once:
        return None

    try:
        return await asyncio.wait_for(
            asyncio.to_thread(
                generate_with_fallback,
                client,
                contents=contents,
                config=config,
            ),
            timeout=max(8, timeout_sec if isinstance(timeout_sec, int) else PHASE5_MODEL_CALL_TIMEOUT_SEC),
        )
    except asyncio.TimeoutError:
        print("[Phase5][Gemini] retry timed out")
        return None
    except Exception as e:
        if _is_rate_limited_error(e):
            raise Phase5RateLimitError("Gemini rate limit reached. Please retry shortly.") from e
        print(f"[Phase5][Gemini] retry failed: {type(e).__name__}: {e}")
        return None


async def _call_openai_chat_json(prompt: str, timeout_sec: int | None = None) -> dict | None:
    api_key = get_openai_api_key()
    model = get_openai_model_name()
    effective_timeout = max(8, timeout_sec if isinstance(timeout_sec, int) else OPENAI_PHASE5_TIMEOUT_SEC)
    model_lower = model.lower()
    is_gpt5_family = model_lower.startswith("gpt-5")

    chat_payload = {
        "model": model,
        "temperature": 0.0,
        "max_tokens": 700,
        "messages": [
            {
                "role": "system",
                "content": "You are a strict JSON assistant. Return only valid JSON.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    }
    responses_payload = {
        "model": model,
        "input": prompt,
        "max_output_tokens": 700,
    }

    try:
        started = asyncio.get_running_loop().time()
        _log_provider_started_once("OpenAI", model)
        async with httpx.AsyncClient(timeout=effective_timeout) as client:
            if is_gpt5_family:
                response = await client.post(
                    "https://api.openai.com/v1/responses",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json=responses_payload,
                )
            else:
                response = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json=chat_payload,
                )
        if response.status_code == 429:
            raise Phase5RateLimitError("OpenAI rate limit reached. Please retry shortly.")
        response.raise_for_status()
        body = response.json()
        if is_gpt5_family:
            text = str(body.get("output_text") or "")
            if not text:
                output = body.get("output")
                if isinstance(output, list):
                    parts = []
                    for item in output:
                        if not isinstance(item, dict):
                            continue
                        content = item.get("content")
                        if isinstance(content, list):
                            for c in content:
                                if isinstance(c, dict):
                                    t = c.get("text")
                                    if isinstance(t, str) and t.strip():
                                        parts.append(t)
                        elif isinstance(content, str) and content.strip():
                            parts.append(content)
                    text = "\n".join(parts).strip()
        else:
            text = (
                ((body.get("choices") or [{}])[0].get("message") or {}).get("content")
                or ""
            )
        parsed = _safe_json_parse(text)
        parsed_dict = parsed if isinstance(parsed, dict) else {}
        if text:
            parsed_dict["_meta_response_text"] = text[:4000]
            text_domains = _extract_domains_from_text(text)
            if text_domains:
                parsed_dict["_meta_source_domains"] = list(dict.fromkeys(text_domains))
        elapsed = asyncio.get_running_loop().time() - started
        _log_provider_healthy_once("OpenAI", model, elapsed)
        return parsed_dict
    except Phase5RateLimitError:
        raise
    except Exception as e:
        if _is_rate_limited_error(e):
            raise Phase5RateLimitError("OpenAI rate limit reached. Please retry shortly.") from e
        try:
            elapsed = asyncio.get_running_loop().time() - started
            print(f"[Phase5][OpenAI] model={model} failed in {elapsed:.2f}s: {type(e).__name__}")
        except Exception:
            pass
        if isinstance(e, httpx.HTTPStatusError) and e.response is not None:
            try:
                print(f"[Phase5][OpenAI] error body: {e.response.text[:500]}")
            except Exception:
                pass
        print(f"[Phase5][OpenAI] call failed: {e}")
        return None


async def _call_perplexity_chat_json(prompt: str, timeout_sec: int | None = None) -> dict | None:
    api_key = get_perplexity_api_key()
    model = (os.getenv("PERPLEXITY_MODEL_PHASE5") or "sonar-pro").strip()
    preset = (os.getenv("PERPLEXITY_PRESET_PHASE5") or "fast-search").strip()
    effective_timeout = max(8, timeout_sec if isinstance(timeout_sec, int) else PERPLEXITY_PHASE5_TIMEOUT_SEC)

    payload = {
        "input": prompt,
    }
    if preset:
        payload["preset"] = preset
    elif model:
        payload["model"] = model

    try:
        started = asyncio.get_running_loop().time()
        _log_provider_started_once("Perplexity", model)
        async with httpx.AsyncClient(timeout=effective_timeout) as client:
            response = await client.post(
                "https://api.perplexity.ai/v1/responses",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
        if response.status_code == 429:
            raise Phase5RateLimitError("Perplexity rate limit reached. Please retry shortly.")
        response.raise_for_status()
        body = response.json()

        text = ""
        citation_domains: list[str] = []
        citation_urls: list[str] = []
        if isinstance(body, dict):
            text = str(body.get("output_text") or "")
            if not text:
                output = body.get("output")
                if isinstance(output, list):
                    parts = []
                    for item in output:
                        if isinstance(item, dict):
                            content = item.get("content")
                            if isinstance(content, list):
                                for c in content:
                                    if isinstance(c, dict):
                                        t = c.get("text")
                                        if isinstance(t, str) and t.strip():
                                            parts.append(t)
                            elif isinstance(content, str) and content.strip():
                                parts.append(content)
                    text = "\n".join(parts).strip()

            if not text:
                text = str(body.get("text") or "")

            citations = body.get("citations")
            if isinstance(citations, list):
                for u in citations:
                    u_str = str(u or "").strip()
                    if u_str.startswith("http://") or u_str.startswith("https://"):
                        citation_urls.append(u_str)
                    d = _normalize_domain(str(u))
                    if d:
                        citation_domains.append(d)

            search_results = body.get("search_results")
            if isinstance(search_results, list):
                for item in search_results:
                    if isinstance(item, dict):
                        url_value = str(item.get("url") or "").strip()
                        if url_value.startswith("http://") or url_value.startswith("https://"):
                            citation_urls.append(url_value)
                        d = _normalize_domain(url_value)
                        if d:
                            citation_domains.append(d)

            web_results = body.get("web_results")
            if isinstance(web_results, list):
                for item in web_results:
                    if isinstance(item, dict):
                        url_value = str(item.get("url") or "").strip()
                        if url_value.startswith("http://") or url_value.startswith("https://"):
                            citation_urls.append(url_value)
                        d = _normalize_domain(url_value)
                        if d:
                            citation_domains.append(d)

        parsed = _safe_json_parse(text)
        elapsed = asyncio.get_running_loop().time() - started
        _log_provider_healthy_once("Perplexity", model, elapsed)
        parsed_dict = parsed if isinstance(parsed, dict) else {}
        if citation_domains:
            parsed_dict["_meta_source_domains"] = list(dict.fromkeys(citation_domains))
        if citation_urls:
            parsed_dict["_meta_source_urls"] = list(dict.fromkeys(citation_urls))
        if text:
            parsed_dict["_meta_response_text"] = text[:4000]
        return parsed_dict
    except Phase5RateLimitError:
        raise
    except Exception as e:
        if _is_rate_limited_error(e):
            raise Phase5RateLimitError("Perplexity rate limit reached. Please retry shortly.") from e
        try:
            elapsed = asyncio.get_running_loop().time() - started
            print(f"[Phase5][Perplexity] model={model} failed in {elapsed:.2f}s: {type(e).__name__}")
        except Exception:
            pass
        print(f"[Phase5][Perplexity] call failed: {e}")
        return None


async def _call_openai_with_retry(prompt: str, retry_once: bool = True, timeout_sec: int | None = None) -> dict | None:
    first = await _call_openai_chat_json(prompt, timeout_sec=timeout_sec)
    if first is not None:
        return first
    if not retry_once:
        return None
    return await _call_openai_chat_json(prompt, timeout_sec=timeout_sec)


async def _call_perplexity_with_retry(prompt: str, retry_once: bool = True, timeout_sec: int | None = None) -> dict | None:
    first = await _call_perplexity_chat_json(prompt, timeout_sec=timeout_sec)
    if first is not None:
        return first
    if not retry_once:
        return None
    return await _call_perplexity_chat_json(prompt, timeout_sec=timeout_sec)
