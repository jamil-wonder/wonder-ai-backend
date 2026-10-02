import unittest

from bs4 import BeautifulSoup

from scraping.locations import _parse_free_text, build_location_info


def soup(html: str) -> BeautifulSoup:
    return BeautifulSoup(html, "html.parser")


def info(schemas=(), html="", meta=None, extra=(), url="https://example.com", phones=()):
    return build_location_info(
        schemas=list(schemas), soups=[soup(html)] if html else [], raw_meta=meta or {},
        extra_addresses=list(extra), page_url=url, phones=list(phones),
    )


def postal(street, city, code, country="GB", region=""):
    return {
        "@type": "PostalAddress", "streetAddress": street, "addressLocality": city,
        "addressRegion": region, "postalCode": code, "addressCountry": country,
    }


class ChainTests(unittest.TestCase):
    def test_multi_location_chain_is_flagged_and_ordered_by_document(self):
        schema = {
            "@type": "Restaurant", "name": "Brand",
            "department": [
                {"@type": "Restaurant", "name": "Brand Soho", "address": postal("5 Soho St", "London", "W1D 3AA")},
                {"@type": "Restaurant", "name": "Brand Bham", "address": postal("1 Chamberlain Sq", "Birmingham", "B3 3DQ")},
                {"@type": "Restaurant", "name": "Brand Mcr", "address": postal("2 Deansgate", "Manchester", "M3 2BW")},
            ],
        }
        result = info([schema])
        self.assertTrue(result["isMultiLocation"])
        self.assertEqual([l["label"] for l in result["locations"]], ["London, UK", "Birmingham, UK", "Manchester, UK"])
        self.assertEqual(result["locationConfidence"], "high")

    def test_same_city_branches_are_one_location(self):
        schemas = [
            {"@type": "Restaurant", "address": postal("5 A St", "London", "W1D 3AA")},
            {"@type": "Restaurant", "address": postal("9 B St", "London", "E1 6AN")},
            {"@type": "Restaurant", "address": postal("3 C St", "London", "N1C 4AB")},
        ]
        result = info(schemas)
        self.assertFalse(result["isMultiLocation"])
        self.assertEqual(len(result["locations"]), 1)
        self.assertEqual(result["locations"][0]["branchCount"], 3)

    def test_same_branch_from_jsonld_and_address_tag_is_one_branch(self):
        schema = {"@type": "Restaurant", "address": postal("5 Soho Street", "London", "W1D 3AA")}
        result = info([schema], html="<address>5 Soho Street<br>London<br>W1D 3AA</address>")
        self.assertEqual(result["locations"][0]["branchCount"], 1)
        self.assertEqual(len(result["orderedAddresses"]), 1)

    def test_order_is_deterministic_across_runs(self):
        schemas = [{"@type": "Store", "address": postal("1 X", c, "AA1 1AA")} for c in ("Leeds", "York", "Hull")]
        first = [l["label"] for l in info(schemas)["locations"]]
        for _ in range(5):
            self.assertEqual([l["label"] for l in info(schemas)["locations"]], first)

    def test_event_venue_in_jsonld_is_ignored(self):
        schemas = [
            {"@type": "Restaurant", "address": postal("1 High St", "Bristol", "BS1 4DJ")},
            {"@type": "Event", "name": "Pop-up", "location": {"@type": "Place", "address": postal("9 Elsewhere Rd", "Glasgow", "G1 1AA")}},
        ]
        result = info(schemas)
        self.assertEqual([l["label"] for l in result["locations"]], ["Bristol, UK"])


class HtmlSourceTests(unittest.TestCase):
    def test_address_tag_with_br_lines_gets_city_from_postcode_line(self):
        html = "<footer><address>5 Stable Street<br>King's Cross<br>London<br>N1C 4AB</address></footer>"
        result = info(html=html)
        self.assertEqual([l["label"] for l in result["locations"]], ["London, UK"])

    def test_us_address(self):
        result = info(html="<address>123 Main St, Austin, TX 78701</address>")
        self.assertEqual(result["locations"][0]["label"], "Austin, TX")

    def test_european_postal_before_city(self):
        result = info(html="<address>14 Rue de Rivoli, 75001 Paris, France</address>")
        self.assertEqual(result["locations"][0]["label"], "Paris, France")

    def test_microdata_postal_address(self):
        html = (
            '<div itemscope itemtype="https://schema.org/PostalAddress">'
            '<span itemprop="streetAddress">10 Park Row</span>'
            '<span itemprop="addressLocality">Leeds</span>'
            '<span itemprop="postalCode">LS1 5HD</span></div>'
        )
        result = info(html=html)
        self.assertEqual(result["locations"][0]["label"], "Leeds, UK")

    def test_google_maps_link_query(self):
        html = '<a href="https://www.google.com/maps?q=Brand+Carnaby,+12+Carnaby+St,+London+W1F+9PS">Map</a>'
        result = info(html=html)
        self.assertEqual(result["locations"][0]["label"], "London, UK")

    def test_maps_lat_lng_is_rejected(self):
        result = info(html='<a href="https://maps.google.com/maps?q=51.5072,-0.1276">Map</a>')
        self.assertEqual(result["locations"], [])

    def test_meta_geo_placename(self):
        result = info(meta={"geo.placename": "Edinburgh"}, url="https://shop.co.uk")
        self.assertEqual(result["locations"][0]["label"], "Edinburgh, UK")

    def test_registered_office_is_not_a_venue(self):
        html = "<p>Registered office</p><address>1 Legal Way, London EC1A 1BB</address>"
        self.assertEqual(info(html=html)["locations"], [])

    def test_text_fallback_only_used_when_nothing_better(self):
        html = "<div>Visit us at<br>Unit 4, Foo Retail Park<br>Leeds LS1 4AB</div>"
        result = info(html=html)
        self.assertEqual(result["locations"][0]["label"], "Leeds, UK")
        self.assertEqual(result["locations"][0]["confidence"], "low")


class JunkAndEdgeTests(unittest.TestCase):
    def test_junk_strings_never_become_locations(self):
        for junk in (
            "Opening Times Mon-Fri 9am to 5pm",
            "51.5072,-0.1276",
            "© 2024 Brand Ltd. All rights reserved. W1U 2NX",
            "info@brand.com",
            "Book a table online",
            "Tel: 020 7946 0958",
            "Unit 4",
        ):
            self.assertIsNone(_parse_free_text(junk), junk)

    def test_sentence_fragments_are_not_cities(self):
        for fragment in ("1798 Established in 1798, Rules is London", "Rules is London, UK"):
            self.assertIsNone(_parse_free_text(fragment), fragment)

    def test_uk_county_is_not_picked_over_town(self):
        parsed = _parse_free_text("Lake Road, Windermere, Cumbria LA23 2EQ")
        self.assertEqual(parsed["city"], "Windermere")
        self.assertEqual(parsed["postal"], "LA23 2EQ")

    def test_weak_text_sources_require_a_postcode(self):
        self.assertEqual(info(extra=["Rules is London, UK", "Somewhere, London"])["locations"], [])
        self.assertEqual(info(html="<div>Visit us in London, England</div>")["locations"], [])
        result = info(extra=["London, WC2E 7LB"])
        self.assertEqual(result["locations"][0]["label"], "London, UK")

    def test_all_caps_city_is_title_cased(self):
        result = info(html="<address>1 HIGH STREET, LONDON, W1U 2NX</address>")
        self.assertEqual(result["locations"][0]["label"], "London, UK")

    def test_outward_code_stripped_from_locality(self):
        schema = {"@type": "Store", "address": postal("2 Road", "London W1", "")}
        self.assertEqual(info([schema])["locations"][0]["city"], "London")

    def test_tld_infers_country_when_missing(self):
        result = info(html="<address>12 High Street, Leeds</address>", url="https://shop.co.uk")
        self.assertEqual(result["locations"][0]["label"], "Leeds, UK")

    def test_nothing_found_is_none(self):
        result = info(html="<p>Hello</p>")
        self.assertEqual(result["locations"], [])
        self.assertEqual(result["locationConfidence"], "none")
        self.assertFalse(result["isMultiLocation"])

    def test_ai_extra_address_never_adds_city_beside_structured_data(self):
        schema = {"@type": "Restaurant", "address": postal("1 High St", "Bristol", "BS1 4DJ")}
        result = info([schema], extra=["10 Downing Street, London SW1A 2AA"])
        self.assertEqual([l["label"] for l in result["locations"]], ["Bristol, UK"])
        self.assertFalse(result["isMultiLocation"])

    def test_ai_extra_used_only_as_low_confidence_fallback(self):
        result = info(extra=["10 Downing Street, London SW1A 2AA"])
        self.assertEqual(result["locations"][0]["label"], "London, UK")
        self.assertEqual(result["locations"][0]["confidence"], "low")

    def test_ordered_addresses_are_best_first_and_deduped(self):
        schema = {"@type": "Restaurant", "address": postal("1 High St", "Bristol", "BS1 4DJ")}
        result = info([schema], extra=["1 High St, Bristol, BS1 4DJ", "garbage"])
        self.assertTrue(result["orderedAddresses"][0].startswith("1 High St"))
        self.assertEqual(len({a.lower() for a in result["orderedAddresses"]}), len(result["orderedAddresses"]))

    def test_never_raises_on_hostile_input(self):
        weird = [None, 5, "x", {"@type": None, "address": [None, 3, {"streetAddress": []}]}, {"address": {"addressLocality": {"a": 1}}}]
        result = info(weird, html="<address></address><a href='%%%'></a>", meta={"geo.placename": ""})
        self.assertIn("locations", result)


if __name__ == "__main__":
    unittest.main()
