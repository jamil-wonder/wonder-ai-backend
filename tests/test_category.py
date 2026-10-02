import unittest

from bs4 import BeautifulSoup

from scraping.category import CATEGORIES, infer_category
from scraping.locations import build_location_info


def soup(html):
    return BeautifulSoup(html, "html.parser")


class CategoryTests(unittest.TestCase):
    def test_grocery_title_is_retail(self):
        cat, conf = infer_category(
            schemas=[], soups=[soup("<h1>Shop</h1>")], raw_meta={},
            title="Shwapno Online Grocery Shopping in Dhaka", description="Order grocery online, save money, save time",
        )
        self.assertEqual(cat, "Retail & Shopping")
        self.assertIn(conf, {"medium", "high"})

    def test_schema_type_wins(self):
        cat, conf = infer_category(
            schemas=[{"@type": "Dentist", "name": "Bright Smile"}], soups=[soup("")], raw_meta={}, title="Bright Smile", description="",
        )
        self.assertEqual(cat, "Healthcare & Medical")
        self.assertEqual(conf, "high")

    def test_graph_types_are_found(self):
        cat, _ = infer_category(
            schemas=[{"@graph": [{"@type": "WebSite"}, {"@type": ["LocalBusiness", "Plumber"]}]}],
            soups=[soup("")], raw_meta={}, title="Acme", description="",
        )
        self.assertEqual(cat, "Home Services")

    def test_vague_site_returns_nothing_rather_than_guessing(self):
        self.assertEqual(infer_category(schemas=[], soups=[soup("")], raw_meta={}, title="Welcome", description="")[0], "")

    def test_result_is_always_a_known_option(self):
        cat, _ = infer_category(schemas=[], soups=[soup("")], raw_meta={}, title="Best Italian restaurant and cafe", description="")
        self.assertIn(cat, CATEGORIES)


class CoverageLocationTests(unittest.TestCase):
    def info(self, title, meta=None, **kw):
        return build_location_info(schemas=[], soups=[soup("<h1></h1>")], raw_meta=meta or {}, headline_texts=[title], **kw)

    def test_cities_from_title_with_country(self):
        out = self.info("Shwapno Online Grocery Shopping in Dhaka, Chattogram, Cumilla & Sylhet", {"og:country_name": "Bangladesh"})
        self.assertEqual([l["label"] for l in out["locations"]],
                         ["Dhaka, Bangladesh", "Chattogram, Bangladesh", "Cumilla, Bangladesh", "Sylhet, Bangladesh"])
        self.assertTrue(out["isMultiLocation"])
        self.assertEqual(out["locationConfidence"], "low")

    def test_single_city(self):
        out = self.info("Best Pizza in Brooklyn")
        self.assertEqual([l["city"] for l in out["locations"]], ["Brooklyn"])

    def test_non_places_are_ignored(self):
        for title in ("Get results in Seconds", "Everything in One place", "Marketing in Action", "Available in Stock today"):
            self.assertEqual(self.info(title)["locations"], [], title)

    def test_real_address_beats_title_cities(self):
        html = '<address>12 Baker Street, London W1U 3BW</address>'
        out = build_location_info(schemas=[], soups=[soup(html)], raw_meta={}, headline_texts=["Plumbers in Leeds, York & Hull"])
        self.assertEqual([l["city"] for l in out["locations"]], ["London"])


if __name__ == "__main__":
    unittest.main()
