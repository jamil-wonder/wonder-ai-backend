import os

NON_COMPETITOR_DOMAINS = {
    "google.com",
    "bing.com",
    "youtube.com",
    "facebook.com",
    "instagram.com",
    "reddit.com",
    "wikipedia.org",
    "yelp.com",
    "tripadvisor.com",
    "opentable.com",
    "booking.com",
    "expedia.com",
    "kayak.com",
    "airbnb.com",
    "hotels.com",
    "maps.google.com",
}

PHASE5_VALIDATE_COMPETITORS = False
PHASE5_FAST_MODE = False
PHASE5_ENABLE_GEMINI = str(os.getenv("PHASE5_ENABLE_GEMINI", "false")).strip().lower() == "true"
PHASE5_MODEL_CALL_TIMEOUT_SEC = int(os.getenv("PHASE5_MODEL_CALL_TIMEOUT_SEC", "90"))
MAX_RETRIES = int(os.getenv("PHASE5_RATE_LIMIT_MAX_RETRIES", "3"))
OPENAI_PHASE5_TIMEOUT_SEC = int(os.getenv("OPENAI_PHASE5_TIMEOUT_SEC", "18"))
OPENAI_PHASE5_MAX_RETRIES = int(os.getenv("PHASE5_RATE_LIMIT_MAX_RETRIES_OPENAI", "2"))
PERPLEXITY_PHASE5_TIMEOUT_SEC = int(os.getenv("PERPLEXITY_PHASE5_TIMEOUT_SEC", "22"))
PERPLEXITY_PHASE5_MAX_RETRIES = int(os.getenv("PHASE5_RATE_LIMIT_MAX_RETRIES_PERPLEXITY", "2"))

PHASE5_CONTEXT_FETCH_TIMEOUT_SEC = float(os.getenv("PHASE5_CONTEXT_FETCH_TIMEOUT_SEC", "8"))
PHASE5_CONTEXT_CACHE_TTL_SEC = int(os.getenv("PHASE5_CONTEXT_CACHE_TTL_SEC", "900"))
