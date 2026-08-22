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
PHASE5_ENABLE_GEMINI = True
PHASE5_MODEL_CALL_TIMEOUT_SEC = int(os.getenv("PHASE5_MODEL_CALL_TIMEOUT_SEC", "90"))
MAX_RETRIES = int(os.getenv("PHASE5_RATE_LIMIT_MAX_RETRIES", "3"))
OPENAI_PHASE5_TIMEOUT_SEC = int(os.getenv("OPENAI_PHASE5_TIMEOUT_SEC", "18"))
OPENAI_PHASE5_MAX_RETRIES = int(os.getenv("PHASE5_RATE_LIMIT_MAX_RETRIES_OPENAI", "2"))
PERPLEXITY_PHASE5_TIMEOUT_SEC = int(os.getenv("PERPLEXITY_PHASE5_TIMEOUT_SEC", "22"))
PERPLEXITY_PHASE5_MAX_RETRIES = int(os.getenv("PHASE5_RATE_LIMIT_MAX_RETRIES_PERPLEXITY", "2"))
# Question generation (20 structured, validated questions across 4
# categories, with regeneration prompts that grow longer on each retry) is
# a heavier job than answering one question — PERPLEXITY_PHASE5_TIMEOUT_SEC
# above is tuned for the light case and was too tight for this one, cutting
# a real in-progress generation off with no error, just a silently skipped
# attempt.
PHASE5_QUESTION_GEN_TIMEOUT_SEC = int(os.getenv("PHASE5_QUESTION_GEN_TIMEOUT_SEC", "45"))
ANTHROPIC_PHASE5_TIMEOUT_SEC = int(os.getenv("ANTHROPIC_PHASE5_TIMEOUT_SEC", "30"))
ANTHROPIC_PHASE5_MAX_RETRIES = int(os.getenv("PHASE5_RATE_LIMIT_MAX_RETRIES_ANTHROPIC", "2"))

PHASE5_CONTEXT_FETCH_TIMEOUT_SEC = float(os.getenv("PHASE5_CONTEXT_FETCH_TIMEOUT_SEC", "8"))
