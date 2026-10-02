import unittest

from scraping.naming import clean_text, prefer_brand_name


class CleanTextTests(unittest.TestCase):
    def test_unescapes_entities(self):
        self.assertEqual(clean_text("GAIL&#39;s"), "GAIL's")
        self.assertEqual(clean_text("Fish &amp; Chips"), "Fish & Chips")

    def test_double_escaped(self):
        self.assertEqual(clean_text("Tom&amp;#39;s Kitchen"), "Tom's Kitchen")

    def test_collapses_layout_whitespace(self):
        self.assertEqual(clean_text("\n      GAIL's Bakery |   Bread\n"), "GAIL's Bakery | Bread")

    def test_none_and_empty(self):
        self.assertEqual(clean_text(None), "")
        self.assertEqual(clean_text("   "), "")


class BrandNameTests(unittest.TestCase):
    def test_branch_name_with_city_becomes_brand(self):
        self.assertEqual(
            prefer_brand_name("Hawksmoor Lakeland Guest House Windermere", "Hawksmoor", cities=["Windermere"]),
            "Hawksmoor",
        )

    def test_branch_name_on_multi_location_site_becomes_brand(self):
        self.assertEqual(prefer_brand_name("Dishoom Carnaby", "Dishoom", multi_location=True), "Dishoom")

    def test_descriptive_suffix_kept_on_single_location_site(self):
        self.assertEqual(prefer_brand_name("Dishoom Indian Restaurants", "Dishoom"), "Dishoom Indian Restaurants")

    def test_not_a_word_boundary_is_not_a_prefix(self):
        self.assertEqual(prefer_brand_name("Dishoomery Cafe", "Dishoom", multi_location=True), "Dishoomery Cafe")

    def test_unrelated_site_name_never_replaces(self):
        self.assertEqual(prefer_brand_name("Rules Restaurant", "Something Else", multi_location=True), "Rules Restaurant")

    def test_missing_site_name_keeps_name(self):
        self.assertEqual(prefer_brand_name("Rules Restaurant", None, multi_location=True), "Rules Restaurant")

    def test_domain_label_recovers_brand_when_rest_is_a_detected_city(self):
        self.assertEqual(
            prefer_brand_name("Hawksmoor Lakeland Guest House Windermere", None, cities=["Windermere"], domain_label="hawksmoor"),
            "Hawksmoor",
        )

    def test_domain_label_ignores_spaces_in_name(self):
        self.assertEqual(
            prefer_brand_name("Cafe Rouge Bristol", None, cities=["Bristol"], domain_label="caferouge"), "Cafe Rouge"
        )

    def test_domain_label_never_uses_the_multi_location_shortcut(self):
        self.assertEqual(
            prefer_brand_name("Brand Kitchen", None, multi_location=True, domain_label="brand"), "Brand Kitchen"
        )

    def test_domain_label_requires_a_word_boundary(self):
        self.assertEqual(
            prefer_brand_name("Brandy Bar London", None, cities=["London"], domain_label="brand"), "Brandy Bar London"
        )

    def test_also_cleans_entities(self):
        self.assertEqual(prefer_brand_name("GAIL&#39;s", None), "GAIL's")


if __name__ == "__main__":
    unittest.main()
