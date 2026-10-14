import os
import re
import time
from typing import List


def describe_gemini_error(exc: Exception) -> str:
    """What Google actually said, short and safe to store or log: the HTTP code,
    the status word (RESOURCE_EXHAUSTED, PERMISSION_DENIED, ...) and the message,
    with any API key removed."""
    code = getattr(exc, "code", None)
    status = getattr(exc, "status", None)
    message = re.sub(r"AIza[0-9A-Za-z_\-]{10,}", "<key>", str(exc or "")).replace("\n", " ").strip()
    head = " ".join(str(p) for p in (code, status) if p)
    return f"{exc.__class__.__name__}{(' ' + head) if head else ''}: {message}"[:300]


_GEMINI_STARTED_ONCE: set[str] = set()
_GEMINI_HEALTHY_ONCE: set[str] = set()


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
    """Return a single configured Gemini model (no pools, no fallbacks)."""
    model = (os.getenv("GEMINI_MODEL_PRIMARY") or os.getenv("GEMINI_MODEL") or "gemini-2.5-pro").strip()
    return [model]


def generate_with_fallback(client, *, contents, config=None):
    """Use the paid model chain and switch model when rate limit/quota is reached."""
    models = get_model_chain()
    last_error = None

    for idx, model in enumerate(models):
        try:
            started = time.perf_counter()
            if model not in _GEMINI_STARTED_ONCE:
                print(f"[Gemini] model={model} started")
                _GEMINI_STARTED_ONCE.add(model)
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
            elapsed = time.perf_counter() - started
            if model not in _GEMINI_HEALTHY_ONCE:
                print(f"[Gemini] model={model} healthy (first success in {elapsed:.2f}s)")
                _GEMINI_HEALTHY_ONCE.add(model)
            setattr(response, "_model_used", model)
            return response
        except Exception as exc:
            elapsed = time.perf_counter() - started if 'started' in locals() else 0.0
            print(f"[Gemini] model={model} failed in {elapsed:.2f}s error={describe_gemini_error(exc)}")
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
