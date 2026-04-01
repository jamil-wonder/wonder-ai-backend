import os
from typing import List


def get_model_chain() -> List[str]:
    """Return ordered unique Gemini models from env with sane defaults."""
    primary = (os.getenv("GEMINI_MODEL_PRIMARY") or os.getenv("GEMINI_MODEL") or "gemini-3.1-pro-preview").strip()
    fallback = (os.getenv("GEMINI_MODEL_FALLBACK") or "gemini-3-flash-preview").strip()

    chain: List[str] = []
    for model in (primary, fallback):
      if model and model not in chain:
          chain.append(model)

    if not chain:
        chain = ["gemini-3.1-pro-preview", "gemini-3-flash-preview"]

    return chain


def generate_with_fallback(client, *, contents, config=None):
    """Try primary Gemini model first, then fallback model on failure."""
    models = get_model_chain()
    last_error = None

    for idx, model in enumerate(models):
        try:
            return client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
        except Exception as exc:
            last_error = exc
            if idx < len(models) - 1:
                print(f"[Gemini] Primary model failed ({model}). Retrying with fallback...")

    if last_error:
        raise last_error
    raise RuntimeError("Gemini generation failed without a captured error.")
