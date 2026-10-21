"""L2: what a customer of THIS kind of business actually asks an AI assistant.

The question prompt used to be one venue-shaped template for every business
("booking, menu, amenities, where can I book"), which produced nonsense for a
software company or a law firm. Each category now maps to a profile: which
buyer intents to ask about, whether a place name belongs in the question at
all, and which hospitality words are not allowed outside hospitality.
"""
import re

HOSPITALITY_WORDS = r"book(?:s|ed|ing|ings)?|reserv(?:e|es|ed|ation|ations|ing)|tables?|stay(?:s|ed|ing)?|visit(?:s|ed|ing)?|check[- ]?in|menus?|tasting"
_HOSPITALITY_RE = re.compile(rf"\b(?:{HOSPITALITY_WORDS})\b", re.I)

# location_mode: "required" - every local question names the place;
#                "optional" - use the place when the business has one;
#                "none"     - a place name does not belong in the questions.
FAMILIES: dict[str, dict] = {
    "hospitality": {
        "terms": "restaurant cafe coffee bakery bistro pizzeria hotel resort hostel lodge venue bar pub tour travel agency".split(),
        "label": "hospitality / venue",
        "hospitality_words_ok": True,
        "location_mode": "required",
        "intents": ["reviews", "what it offers (menu, rooms or experiences)", "atmosphere and suitability", "an occasion or group size", "price and value", "booking and availability"],
        "example": "Best restaurant in Belgravia?",
    },
    "retail": {
        "terms": "shop store boutique retailer seller supplier online".split(),
        "label": "retail / shop",
        "hospitality_words_ok": False,
        "location_mode": "optional",
        "intents": ["product range", "price and value", "quality and brands", "delivery and returns", "comparison with alternatives", "reviews and trust"],
        "example": "Where can I buy handmade leather bags with free delivery?",
    },
    "local_service": {
        "terms": "plumber electrician roofer locksmith cleaner mover painter contractor handyman company service".split(),
        "label": "local service / trade",
        "hospitality_words_ok": False,
        "location_mode": "required",
        "intents": ["cost and quotes", "reliability and response time", "qualifications and accreditation", "emergency or same-day availability", "reviews and trust", "comparison of providers"],
        "example": "Who is the most reliable emergency plumber in Leeds?",
    },
    "personal_care": {
        "terms": "salon barber spa gym studio trainer therapist clinic wellness fitness classes".split(),
        "label": "beauty / fitness / wellness",
        "hospitality_words_ok": False,
        "location_mode": "required",
        "intents": ["treatments or classes offered", "price and packages", "qualified practitioners or trainers", "results and reviews", "appointments and availability", "suitability for a goal or level"],
        "example": "Which gym in Bristol has the best personal trainers for beginners?",
    },
    "health": {
        "terms": "dentist doctor clinic physio physiotherapist dental practice pharmacy optician vet hospital".split(),
        "label": "healthcare",
        "hospitality_words_ok": False,
        "location_mode": "required",
        "intents": ["specialisms and treatments", "qualifications and registration", "fees and insurance", "waiting times and availability", "patient reviews and trust", "comparison of clinics"],
        "example": "Which dental clinic in Dhaka offers emergency appointments?",
    },
    "professional": {
        "terms": "solicitor lawyer attorney accountant firm advisor adviser consultant agency".split(),
        "label": "legal / financial / professional services",
        "hospitality_words_ok": False,
        "location_mode": "optional",
        "intents": ["areas of expertise", "fees and pricing", "credentials and experience", "client reviews and trust", "how the process works", "comparison of firms"],
        "example": "Which accounting firm is best for small e-commerce businesses?",
    },
    "real_estate": {
        "terms": "estate agent agents property builder contractor architect developer".split(),
        "label": "real estate / construction",
        "hospitality_words_ok": False,
        "location_mode": "required",
        "intents": ["areas and property types covered", "fees and costs", "track record and experience", "reviews and trust", "comparison of agents or contractors", "timelines"],
        "example": "Which estate agent in Manchester is best for first-time buyers?",
    },
    "automotive": {
        "terms": "garage mechanic dealer dealership workshop servicing".split(),
        "label": "automotive",
        "hospitality_words_ok": False,
        "location_mode": "required",
        "intents": ["services or vehicles offered", "price and quotes", "qualifications and warranty", "turnaround time", "reviews and trust", "comparison of garages or dealers"],
        "example": "Which garage in Glasgow is best for electric car servicing?",
    },
    "education": {
        "terms": "school course courses academy college training tutor bootcamp".split(),
        "label": "education / training",
        "hospitality_words_ok": False,
        "location_mode": "optional",
        "intents": ["courses and subjects", "accreditation and outcomes", "fees and funding", "study format (online or in person)", "student reviews", "comparison of providers"],
        "example": "Which coding bootcamp is best for career changers?",
    },
    "software": {
        "terms": "software platform app tool tools saas system service".split(),
        "label": "software / online service",
        "hospitality_words_ok": False,
        "location_mode": "none",
        "intents": ["features and capabilities", "pricing and plans", "integrations", "which team size or use case it suits", "alternatives and comparison", "security, support and reviews"],
        "example": "What is the best invoicing software for freelancers?",
    },
    "nonprofit": {
        "terms": "charity charities organisation organization foundation".split(),
        "label": "nonprofit / community",
        "hospitality_words_ok": False,
        "location_mode": "optional",
        "intents": ["mission and the cause it serves", "impact and transparency", "how to donate or volunteer", "trust and accountability", "comparison with similar organisations"],
        "example": "Which charity is most effective at helping homeless people in London?",
    },
    "general": {
        "terms": "".split(),
        "label": "general business",
        "hospitality_words_ok": False,
        "location_mode": "optional",
        "intents": ["what it offers", "price and value", "reviews and trust", "who it suits", "comparison with alternatives"],
        "example": "Which company is best for this service?",
    },
}

CATEGORY_FAMILY = {
    "restaurant & food": "hospitality",
    "hotel & hospitality": "hospitality",
    "travel & tourism": "hospitality",
    "entertainment & events": "hospitality",
    "retail & shopping": "retail",
    "healthcare & medical": "health",
    "legal services": "professional",
    "financial services": "professional",
    "professional services": "professional",
    "real estate": "real_estate",
    "construction & contracting": "real_estate",
    "home services": "local_service",
    "automotive": "automotive",
    "beauty & personal care": "personal_care",
    "fitness & wellness": "personal_care",
    "education & training": "education",
    "technology & software": "software",
    "nonprofit & community": "nonprofit",
}

# Free-text categories ("Restaurant", "E-Commerce SaaS") fall back to words.
_KEYWORD_FAMILY = [
    ("software", r"software|saas|\bapp\b|platform|api|cloud|cyber|\bit services?\b|web ?(design|development)|digital agency|e-?commerce (builder|platform)"),
    ("hospitality", r"restaurant|caf[eé]|bakery|bistro|pizz|catering|dining|coffee|\bpub\b|\bbar\b|brewery|hotel|resort|hostel|lodging|b&b|bed and breakfast|travel|tour|venue|nightclub|cinema|theatre|theater"),
    ("health", r"medical|clinic|dental|dentist|hospital|doctor|physio|pharmacy|optic|veterinar|health ?care"),
    ("professional", r"law|legal|solicitor|attorney|account|tax|financ|insurance|bank|consult|marketing agency|recruit"),
    ("real_estate", r"real estate|estate agent|property|construction|contractor|builder|architect"),
    ("local_service", r"plumb|electric|roof|locksmith|clean|moving|hvac|painter|handyman|pest|landscap"),
    ("automotive", r"auto|car |garage|mechanic|tyre|tire|dealership"),
    ("personal_care", r"salon|barber|beauty|spa|nail|tattoo|gym|fitness|yoga|pilates|wellness|personal train"),
    ("education", r"school|academy|college|universit|course|training|tutor|bootcamp"),
    ("retail", r"shop|store|retail|boutique|florist|grocery|supermarket|fashion|jewel"),
    ("nonprofit", r"charity|nonprofit|non-profit|ngo|foundation|community"),
]


def family_for(category, sub_type="") -> str:
    cat = re.sub(r"\s+", " ", str(category or "").strip().lower())
    if cat in CATEGORY_FAMILY:
        return CATEGORY_FAMILY[cat]
    text = f"{cat} {str(sub_type or '').lower()}"
    for family, pattern in _KEYWORD_FAMILY:
        if re.search(pattern, text):
            return family
    return "general"


def question_profile(category, sub_type="") -> dict:
    family = family_for(category, sub_type)
    return {**FAMILIES[family], "family": family}


def natural_terms(profile: dict, sub_type="") -> list[str]:
    """Everyday words customers use for this kind of business ("solicitors", "dentists"),
    so a question that says that - instead of repeating our category label - still counts as on-topic."""
    out = list(profile.get("terms") or [])
    out += re.findall(r"[a-z0-9]+", str(sub_type or "").lower())
    return [t for t in dict.fromkeys(out) if len(t) >= 4]


ONLINE_LOCATION = "Online / everywhere"


def is_online_location(location) -> bool:
    """J3: a business that works everywhere / online has no place to name in its questions."""
    return str(location or "").strip().lower() in {ONLINE_LOCATION.lower(), "online", "everywhere", "worldwide"}


def uses_location(profile: dict, location) -> bool:
    """Whether the questions for this business should name a place."""
    if is_online_location(location):
        return False
    mode = profile.get("location_mode")
    if mode == "required":
        return True
    if mode == "optional":
        return bool(str(location or "").strip())
    return False


def violates_profile(text, profile: dict) -> bool:
    """A question using hospitality wording (book / table / stay / visit / menu ...)
    for a business that is not hospitality."""
    if profile.get("hospitality_words_ok"):
        return False
    return bool(_HOSPITALITY_RE.search(str(text or "")))
