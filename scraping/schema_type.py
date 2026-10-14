"""C3: which JSON-LD `@type`s mark a site as a local business.

A site whose markup only says `Organization` / `WebSite` / `Service` tells AI
models nothing about where the business is or what kind it is, so the Schema
score is capped at 50% until a LocalBusiness-family type is present."""
import re

# schema.org LocalBusiness and its commonly used sub-types
LOCAL_BUSINESS_TYPES = {
    "localbusiness", "foodestablishment", "restaurant", "fastfoodrestaurant", "cafeorcoffeeshop", "barorpub",
    "bakery", "brewery", "winery", "distillery", "icecreamshop", "foodestablishment",
    "lodgingbusiness", "hotel", "motel", "resort", "hostel", "bedandbreakfast", "campground", "vacationrental",
    "store", "shoppingcenter", "grocerystore", "supermarket", "florist", "bookstore", "clothingstore", "electronicsstore",
    "furniturestore", "hardwarestore", "jewelrystore", "liquorstore", "petstore", "shoestore", "sportinggoodsstore",
    "toystore", "departmentstore", "conveniencestore", "mensclothingstore", "tireshop", "gardenstore", "homegoodsstore",
    "healthandbeautybusiness", "beautysalon", "hairsalon", "nailsalon", "dayspa", "healthclub", "tattooparlor",
    "medicalbusiness", "medicalclinic", "dentist", "physician", "pharmacy", "hospital", "optician", "veterinarycare",
    "legalservice", "attorney", "notary", "accountingservice", "financialservice", "bankorcreditunion",
    "insuranceagency", "automatedteller", "realestateagent", "professionalservice", "travelagency", "employmentagency",
    "autorepair", "autodealer", "autobodyshop", "autowash", "gasstation", "motorcycledealer",
    "homeandconstructionbusiness", "electrician", "plumber", "generalcontractor", "roofingcontractor", "locksmith",
    "movingcompany", "housepainter", "hvacbusiness",
    "entertainmentbusiness", "sportsactivitylocation", "golfcourse", "childcare", "selfstorage", "drycleaningorlaundry",
    "artgallery", "casino", "movietheater", "nightclub", "amusementpark", "bowlingalley", "skiresort", "stadiumorarena",
    "internetcafe", "library", "radiostation", "televisionstation", "emergencyservice", "firestation", "policestation",
    "sportsteam", "governmentoffice", "postoffice",
}


def _type_names(node) -> list[str]:
    raw = node.get("@type") if isinstance(node, dict) else None
    items = raw if isinstance(raw, list) else [raw]
    out = []
    for t in items:
        if isinstance(t, str) and t.strip():
            # "https://schema.org/Restaurant" or "schema:Restaurant" -> "Restaurant"
            out.append(re.split(r"[/#:]", t.strip())[-1])
    return out


def local_business_types(schemas) -> list[str]:
    """The LocalBusiness-family types found in the markup (original spelling)."""
    found: list[str] = []
    for node in schemas or []:
        for name in _type_names(node):
            low = name.lower()
            if (low in LOCAL_BUSINESS_TYPES or low.endswith("store") or low.endswith("restaurant")) and name not in found:
                found.append(name)
    return found


SCHEMA_MAX_SCORE = 10          # the Schema area is worth 10 of the 100 points
_SCHEMA_PRESENT, _SCHEMA_TYPE, _SCHEMA_KEY_FIELDS = 4, 3, 3


def compute_schema_score(has_schema: bool, local_types: list, has_key_fields: bool) -> dict:
    """present 4 + LocalBusiness-family type 3 + key fields (address, phone, name) 3.
    Without a LocalBusiness-family type the whole score is capped at 50%."""
    present = _SCHEMA_PRESENT if has_schema else 0
    type_pts = _SCHEMA_TYPE if local_types else 0
    key_pts = _SCHEMA_KEY_FIELDS if has_key_fields else 0
    uncapped = present + type_pts + key_pts
    total = uncapped if local_types else min(uncapped, SCHEMA_MAX_SCORE // 2)
    return {"total": total, "present": present, "correctType": type_pts, "keyFields": key_pts,
            "uncapped": uncapped, "capped": total < uncapped}
