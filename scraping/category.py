"""Deterministic business-category inference for onboarding prefill.

The dashboard's category picker offers a fixed list. Instead of leaving it blank
(and making the user guess what we'd call them), score the signals the site
already publishes — schema.org types first, then title/headings/descriptions —
and return one of those exact labels, with how sure we are.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

CATEGORIES = [
    "Restaurant & Food", "Hotel & Hospitality", "Retail & Shopping", "Healthcare & Medical", "Legal Services",
    "Real Estate", "Home Services", "Automotive", "Beauty & Personal Care", "Fitness & Wellness",
    "Professional Services", "Education & Training", "Technology & Software", "Financial Services",
    "Construction & Contracting", "Entertainment & Events", "Travel & Tourism", "Nonprofit & Community",
]

_SCHEMA_TYPES: Dict[str, str] = {}
for _category, _types in {
    "Restaurant & Food": "Restaurant CafeOrCoffeeShop Bakery FoodEstablishment BarOrPub Brewery FastFoodRestaurant IceCreamShop Winery Distillery",
    "Hotel & Hospitality": "Hotel LodgingBusiness Resort BedAndBreakfast Hostel Motel Campground",
    "Retail & Shopping": "Store OnlineStore GroceryStore ClothingStore ShoeStore JewelryStore FurnitureStore BookStore ElectronicsStore HardwareStore PetStore Florist DepartmentStore ConvenienceStore LiquorStore",
    "Healthcare & Medical": "MedicalBusiness Physician Dentist Hospital Pharmacy MedicalClinic Optician VeterinaryCare DiagnosticLab",
    "Legal Services": "LegalService Attorney Notary",
    "Real Estate": "RealEstateAgent",
    "Home Services": "HomeAndConstructionBusiness Plumber Electrician RoofingContractor Locksmith HousePainter MovingCompany HVACBusiness",
    "Automotive": "AutoDealer AutoRepair AutomotiveBusiness AutoBodyShop AutoPartsStore CarWash GasStation",
    "Beauty & Personal Care": "BeautySalon HairSalon DaySpa NailSalon HealthAndBeautyBusiness BarberShop TattooParlor",
    "Fitness & Wellness": "ExerciseGym SportsActivityLocation HealthClub",
    "Professional Services": "AccountingService ProfessionalService EmploymentAgency",
    "Education & Training": "EducationalOrganization School CollegeOrUniversity Preschool Course",
    "Technology & Software": "SoftwareApplication WebApplication MobileApplication",
    "Financial Services": "FinancialService BankOrCreditUnion InsuranceAgency AutomatedTeller",
    "Construction & Contracting": "GeneralContractor",
    "Entertainment & Events": "EventVenue MovieTheater NightClub EntertainmentBusiness AmusementPark Museum PerformingArtsTheater",
    "Travel & Tourism": "TravelAgency TouristInformationCenter TouristAttraction",
    "Nonprofit & Community": "NGO",
}.items():
    for _t in _types.split():
        _SCHEMA_TYPES[_t.lower()] = _category

_KEYWORDS: Dict[str, str] = {
    "Restaurant & Food": r"restaurants?|caf[eé]s?|bakery|bistro|pizzeria|pizza|catering|takeaway|dining|coffee|pub|brewery|chef|brunch|cuisine|bar and grill|menu",
    "Hotel & Hospitality": r"hotels?|inn|resort|hostel|b&b|bed and breakfast|guest ?house|accommodation|boutique hotel|rooms? (?:and|&) suites",
    "Retail & Shopping": r"shop|shopping|store|stores|grocery|groceries|supermarket|boutique|fashion|clothing|furniture|marketplace|e-?commerce|retail|online shop|add to (?:cart|bag|basket)|jewell?ery|gifts?",
    "Healthcare & Medical": r"clinic|dental|dentist|doctors?|hospital|medical|physio(?:therapy)?|pharmacy|therapy|healthcare|health care|optician|veterinary|vets?",
    "Legal Services": r"law firm|solicitors?|attorneys?|lawyers?|legal services|barristers?|conveyancing",
    "Real Estate": r"real estate|estate agents?|property|properties|lettings?|realtors?|homes for sale|mortgage broker",
    "Home Services": r"plumbers?|plumbing|electricians?|cleaning|roofing|handyman|locksmith|pest control|landscap(?:e|ing)|hvac|removals|gardening|decorators?",
    "Automotive": r"cars?|auto|garage|mot|tyres?|dealership|motors?|vehicles?|automotive|car rental|bikes?",
    "Beauty & Personal Care": r"salon|beauty|barber|spa|nails|hair|cosmetics?|makeup|skincare|aesthetics?",
    "Fitness & Wellness": r"gym|fitness|yoga|pilates|personal train(?:er|ing)|wellness|crossfit|workout",
    "Professional Services": r"consult(?:ing|ancy|ants?)|accountants?|accounting|marketing agency|agency|advisory|recruitment|architects?|design studio|bookkeeping",
    "Education & Training": r"school|academy|courses?|training|tutor(?:s|ing)?|university|college|bootcamp|learn(?:ing)?|classes",
    "Technology & Software": r"software|saas|platform|apps?|api|cloud|developers?|ai-powered|it services|automation|analytics|dashboard",
    "Financial Services": r"bank|banking|insurance|mortgages?|loans?|invest(?:ing|ment|ments)?|wealth|fintech|credit|payments?|pension",
    "Construction & Contracting": r"construction|builders?|contractors?|building services|civil engineering|renovations?|extensions?",
    "Entertainment & Events": r"events?|wedding|venue|festival|theatre|theater|cinema|music|tickets|nightclub|entertainment|concerts?",
    "Travel & Tourism": r"travel|tours?|holidays?|flights?|tourism|cruises?|trips?|excursions?|itinerar(?:y|ies)",
    "Nonprofit & Community": r"charity|non-?profit|foundation|donate|volunteer|ngo|community group",
}
_COMPILED = {c: re.compile(rf"(?<![\w-])(?:{p})(?![\w-])", re.I) for c, p in _KEYWORDS.items()}


def _collect_types(node: Any, out: List[str], depth: int = 0) -> None:
    if depth > 6 or len(out) > 60:
        return
    if isinstance(node, list):
        for item in node[:40]:
            _collect_types(item, out, depth + 1)
    elif isinstance(node, dict):
        t = node.get("@type")
        if isinstance(t, str):
            out.append(t)
        elif isinstance(t, list):
            out.extend(x for x in t if isinstance(x, str))
        for key in ("@graph", "mainEntity", "about", "provider", "publisher", "itemListElement"):
            if key in node:
                _collect_types(node[key], out, depth + 1)


def infer_category(
    *,
    schemas: Iterable[Any],
    soups: Iterable[Any],
    raw_meta: Optional[Dict[str, str]] = None,
    title: str = "",
    description: str = "",
) -> Tuple[str, str]:
    """Return (category, confidence); ("", "none") when nothing is convincing."""
    meta = raw_meta or {}
    scores: Dict[str, float] = {c: 0.0 for c in CATEGORIES}
    schema_hit = False

    types: List[str] = []
    _collect_types(list(schemas), types)
    for t in types:
        cat = _SCHEMA_TYPES.get(t.split("/")[-1].lower())
        if cat:
            scores[cat] += 8
            schema_hit = True

    headings: List[str] = []
    for soup in list(soups)[:1]:
        try:
            for tag in soup.find_all(["h1", "h2"], limit=6):
                headings.append(tag.get_text(" ", strip=True)[:160])
        except Exception:
            pass

    weighted = [
        (title, 3), (meta.get("og:title", ""), 3), (meta.get("og:site_name", ""), 2),
        (description, 2), (meta.get("og:description", ""), 2), (meta.get("keywords", ""), 2),
        (meta.get("og:type", ""), 1),
    ] + [(h, 2) for h in headings[:4]]

    for text, weight in weighted:
        if not text:
            continue
        for cat, rx in _COMPILED.items():
            hits = len({m.group(0).lower() for m in rx.finditer(str(text))})
            if hits:
                scores[cat] += weight * min(hits, 2)

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best, best_score = ranked[0]
    second = ranked[1][1] if len(ranked) > 1 else 0
    if best_score < 4 or best_score - second < 1.5:
        return "", "none"
    if schema_hit and best_score >= 8:
        return best, "high"
    return best, "medium" if best_score >= 7 else "low"
