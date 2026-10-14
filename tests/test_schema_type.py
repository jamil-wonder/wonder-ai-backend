"""C3: the Schema score needs a LocalBusiness-family @type, or it is capped at 50%."""
import unittest

from scraping.schema_type import compute_schema_score, local_business_types


class LocalBusinessTypeTests(unittest.TestCase):
    def test_restaurant_and_local_business_count(self):
        self.assertEqual(local_business_types([{"@type": "Restaurant"}]), ["Restaurant"])
        self.assertEqual(local_business_types([{"@type": "LocalBusiness"}]), ["LocalBusiness"])

    def test_organization_website_and_service_alone_do_not_count(self):
        for t in ("Organization", "WebSite", "WebPage", "Service", "Person", "Product", "BreadcrumbList"):
            self.assertEqual(local_business_types([{"@type": t}]), [], t)

    def test_type_can_be_a_list_or_a_url(self):
        self.assertEqual(local_business_types([{"@type": ["Organization", "Hotel"]}]), ["Hotel"])
        self.assertEqual(local_business_types([{"@type": "https://schema.org/Dentist"}]), ["Dentist"])
        self.assertEqual(local_business_types([{"@type": "schema:Bakery"}]), ["Bakery"])

    def test_any_kind_of_store_counts(self):
        self.assertEqual(local_business_types([{"@type": "BicycleStore"}]), ["BicycleStore"])

    def test_case_is_ignored_but_spelling_is_kept(self):
        self.assertEqual(local_business_types([{"@type": "restaurant"}]), ["restaurant"])

    def test_several_nodes_are_all_checked_without_duplicates(self):
        schemas = [{"@type": "WebSite"}, {"@type": "Restaurant"}, {"@type": "Restaurant"}, {"@type": "Hotel"}]
        self.assertEqual(local_business_types(schemas), ["Restaurant", "Hotel"])

    def test_garbage_is_ignored(self):
        self.assertEqual(local_business_types([None, "x", {"@type": None}, {"@type": 5}, {}]), [])
        self.assertEqual(local_business_types(None), [])


class SchemaScoreCapTests(unittest.TestCase):
    def test_full_markup_with_a_local_type_scores_ten(self):
        r = compute_schema_score(True, ["Restaurant"], True)
        self.assertEqual((r["total"], r["capped"]), (10, False))

    def test_without_a_local_type_the_score_is_capped_at_fifty_percent(self):
        r = compute_schema_score(True, [], True)        # would have been 7
        self.assertEqual((r["uncapped"], r["total"], r["capped"]), (7, 5, True))

    def test_a_low_score_without_a_type_is_not_reported_as_capped(self):
        r = compute_schema_score(True, [], False)       # 4 is already under the cap
        self.assertEqual((r["total"], r["capped"]), (4, False))

    def test_no_markup_at_all(self):
        self.assertEqual(compute_schema_score(False, [], False)["total"], 0)

    def test_the_type_is_what_unlocks_the_rest(self):
        self.assertGreater(compute_schema_score(True, ["Hotel"], True)["total"], compute_schema_score(True, [], True)["total"])


if __name__ == "__main__":
    unittest.main()
