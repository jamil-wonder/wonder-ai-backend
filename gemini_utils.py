import os
import time
from typing import List


def _is_rate_limited_error(exc: Exception) -> bool:
    msg = str(exc or "").lower()
    return (
        "429" in msg
        or "resource_exhausted" in msg
        or "rate limit" in msg
        or "too many requests" in msg
        or "quota" in msg
    )


def _is_retryable_failover_error(exc: Exception) -> bool:
    msg = str(exc or "").lower()
    return (
        _is_rate_limited_error(exc)
        or "503" in msg
        or "unavailable" in msg
        or "temporarily" in msg
        or "high demand" in msg
        or "internal" in msg
        or "deadline exceeded" in msg
        or "timeout" in msg
    )


def get_model_chain() -> List[str]:
    """Return ordered paid-model chain used for automatic quota failover."""
    defaults = [
        "gemini-2.5-pro",
        "gemini-3-flash-preview",
        "gemini-2.5-flash",
        "gemini-3.1-pro-preview",
    ]

    configured_pool = (os.getenv("GEMINI_MODEL_POOL") or "").strip()
    from_pool = [m.strip() for m in configured_pool.split(",") if m.strip()]

    primary = (os.getenv("GEMINI_MODEL_PRIMARY") or os.getenv("GEMINI_MODEL") or "").strip()
    fallback = (os.getenv("GEMINI_MODEL_FALLBACK") or "").strip()

    chain: List[str] = []
    for model in [primary, *from_pool, fallback, *defaults]:
        if model and model not in chain:
            chain.append(model)
    return chain


def generate_with_fallback(client, *, contents, config=None):
    """Use the paid model chain and switch model when rate limit/quota is reached."""
    models = get_model_chain()
    last_error = None

    for idx, model in enumerate(models):
        try:
            started = time.perf_counter()
            print(f"[Gemini] model={model} request started")
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
            elapsed = time.perf_counter() - started
            print(f"[Gemini] model={model} success in {elapsed:.2f}s")
            setattr(response, "_model_used", model)
            return response
        except Exception as exc:
            elapsed = time.perf_counter() - started if 'started' in locals() else 0.0
            print(f"[Gemini] model={model} failed in {elapsed:.2f}s error={exc.__class__.__name__}")
            last_error = exc
            if idx < len(models) - 1 and _is_retryable_failover_error(exc):
                print(f"[Gemini] Retryable failure on {model}. Switching to next paid model.")
                continue
            if idx < len(models) - 1:
                print(f"[Gemini] Model failed ({model}). Stopping failover (not a rate-limit error).")
            raise

    if last_error:
        raise last_error
    raise RuntimeError("Gemini generation failed without a captured error.")
