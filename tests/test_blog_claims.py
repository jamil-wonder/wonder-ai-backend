"""F1: blog claims must come from the site or the profile."""
import unittest
from unittest import mock

from blog_claims import build_fact_pack, facts_prompt_block, is_specific, normalize_claims, vet_blog

BIZ = {
    "businessName": "Cornus", "category": "Restaurant", "sub_type": "French", "location": "Belgravia, London",
    "services": ["Tasting menu", "Private dining"], "businessDescription": "A modern French restaurant. Seasonal produce.",
    "latest_scrape_result": {"addresses": ["Eccleston Yards, London"], "phones": ["020 7000 0000"], "openingHours": ["Tue-Sat 12-10"]},
}


class FactPackTests(unittest.TestCase):
    def test_facts_come_only_from_profile_and_website_with_sources(self):
        facts = build_fact_pack(BIZ)
        texts = [f["text"] for f in facts]
        self.assertIn("The business is called Cornus.", texts)
        self.assertIn("Offers: Private dining.", texts)
        self.assertTrue(any("Tue-Sat 12-10" in t for t in texts))
        self.assertEqual({f["source"] for f in facts}, {"profile", "website"})
        self.assertEqual([f["id"] for f in facts], [f"F{i + 1}" for i in range(len(facts))])

    def test_nothing_is_invented_for_an_empty_profile(self):
        self.assertEqual(build_fact_pack({}), [])
        self.assertIn("no verified facts", facts_prompt_block([]))

    def test_duplicates_and_size_are_capped(self):
        facts = build_fact_pack({"businessName": "X", "services": ["a"] * 50})
        self.assertLessEqual(len(facts), 40)
        self.assertEqual(len([f for f in facts if f["text"] == "Offers: a."]), 1)


class SpecificTests(unittest.TestCase):
    def test_specific_statements(self):
        for t in ("We have been open since 1998.", "Rated 4.8 stars by diners.", "Prices start at £45.", "A Michelin-starred kitchen.",
                  "Studies show people prefer tasting menus.", "Over 10,000 happy guests.", "Award-winning service."):
            self.assertTrue(is_specific(t), t)

    def test_general_statements(self):
        for t in ("A tasting menu lets the kitchen show its range.", "Booking ahead is wise for weekends.",
                  "Review menus in advance.", "Research the cooking style of each restaurant."):
            self.assertFalse(is_specific(t), t)


class VetTests(unittest.TestCase):
    FACTS = build_fact_pack(BIZ)

    def sections(self):
        return [{"id": "s1", "heading": "About", "label": "About", "content":
                 "Cornus is a French restaurant in Belgravia. It has won three awards since 2019. Booking ahead is wise for weekends."}]

    def test_unsourced_specific_general_claim_is_dropped_from_the_text(self):
        claims = [{"text": "It has won three awards since 2019.", "source": "general"},
                  {"text": "Cornus is a French restaurant in Belgravia.", "source": "F2"}]
        out = vet_blog(self.sections(), claims, self.FACTS)
        text = out["sections"][0]["content"]
        self.assertNotIn("awards", text)
        self.assertIn("French restaurant in Belgravia", text)
        self.assertIn("Booking ahead", text)
        self.assertEqual(out["droppedClaims"], ["It has won three awards since 2019."])
        self.assertTrue(any("removed 1 sentence" in c for c in out["checkBeforePublishing"]))

    def test_sourced_claims_are_kept_and_listed_for_confirmation(self):
        claims = [{"text": "Cornus is a French restaurant in Belgravia.", "source": "F3"}]
        out = vet_blog(self.sections(), claims, self.FACTS)
        self.assertEqual(out["claims"][0]["sourceLabel"], "Your profile")
        self.assertTrue(any(c.startswith("Confirm this is still true") for c in out["checkBeforePublishing"]))

    def test_a_made_up_fact_id_becomes_general_and_is_checked(self):
        claims = [{"text": "It has won three awards since 2019.", "source": "F99"}]
        out = vet_blog(self.sections(), claims, self.FACTS)
        self.assertEqual(out["droppedClaims"], ["It has won three awards since 2019."])     # not trusted just for citing an id

    def test_absolute_wording_in_general_claims_is_softened(self):
        sections = [{"id": "s", "heading": "H", "label": "H", "content": "This is the best way to plan a dinner. Always book early."}]
        claims = [{"text": "This is the best way to plan a dinner.", "source": "general"}, {"text": "Always book early.", "source": "general"}]
        out = vet_blog(sections, claims, [])
        self.assertIn("one of the best way", out["sections"][0]["content"])
        self.assertIn("often book early", out["sections"][0]["content"])
        self.assertEqual(out["softenedCount"], 2)

    def test_a_claim_that_cannot_be_found_is_flagged_not_silently_trusted(self):
        out = vet_blog(self.sections(), [{"text": "Rated 4.9 by critics everywhere.", "source": "general"}], self.FACTS)
        # the "Rated 4.9" claim is not in the text, so it can only be flagged ...
        self.assertTrue(any("couldn't source" in c and "4.9" in c for c in out["checkBeforePublishing"]))
        # ... while the sample text's own invented year is caught by the safety net
        self.assertEqual(out["droppedClaims"], ["It has won three awards since 2019."])

    def test_the_owner_is_always_reminded_to_add_a_real_detail(self):
        out = vet_blog(self.sections(), [], self.FACTS)
        self.assertTrue(any("only you know" in c for c in out["checkBeforePublishing"]))

    def test_sections_emptied_by_dropping_are_removed(self):
        sections = [{"id": "s", "heading": "Awards", "label": "A", "content": "It has won three awards since 2019."}]
        out = vet_blog(sections, [{"text": "It has won three awards since 2019.", "source": "general"}], self.FACTS)
        self.assertEqual(out["sections"], [])

    def test_malformed_claims_are_ignored(self):
        self.assertEqual(normalize_claims("nope", self.FACTS), [])
        self.assertEqual(normalize_claims([None, {"text": ""}, {"source": "F1"}], self.FACTS), [])


class SafetyNetTests(unittest.TestCase):
    FACTS = build_fact_pack({"businessName": "Cornus", "businessDescription": "Open Tuesday to Saturday. Dinner from 17:45."})

    def test_an_invented_price_year_or_count_is_removed_even_if_the_model_did_not_list_it(self):
        sections = [{"id": "s", "heading": "H", "label": "H", "content":
                     "Cornus serves dinner. Tasting menus cost £95 per person. We opened in 2012. Over 5,000 guests visit each year. Book early."}]
        out = vet_blog(sections, [], self.FACTS)
        text = out["sections"][0]["content"]
        for gone in ("£95", "2012", "5,000"):
            self.assertNotIn(gone, text)
        self.assertIn("Cornus serves dinner", text)
        self.assertIn("Book early", text)
        self.assertEqual(len(out["droppedClaims"]), 3)

    def test_a_figure_that_is_in_the_facts_is_kept(self):
        sections = [{"id": "s", "heading": "H", "label": "H", "content": "Dinner starts at 17:45 on Tuesdays. It is a 2019 favourite."}]
        facts = build_fact_pack({"businessName": "Cornus", "businessDescription": "Dinner from 17:45. A 2019 favourite."})
        out = vet_blog(sections, [], facts)
        self.assertIn("2019", out["sections"][0]["content"])

    def test_ordinary_numbers_in_advice_are_not_touched(self):
        sections = [{"id": "s", "heading": "H", "label": "H", "content": "Here are 5 tips for choosing a venue. Ask 3 questions before booking."}]
        out = vet_blog(sections, [], self.FACTS)
        self.assertIn("5 tips", out["sections"][0]["content"])
        self.assertEqual(out["droppedClaims"], [])


class GenerateBlogTests(unittest.IsolatedAsyncioTestCase):
    async def test_generator_sends_the_facts_and_applies_the_checks(self):
        from agents import ai_agent

        seen = {}

        async def fake_model_json(**kw):
            seen["prompt"] = kw["prompt"]
            return ({
                "title": "T", "sections": [{"id": "a", "heading": "H", "content": "Cornus serves a tasting menu. It was founded in 1999. Reserve early."}],
                "claims": [{"text": "Cornus serves a tasting menu.", "source": "F5"}, {"text": "It was founded in 1999.", "source": "general"}],
            }, "claude-x", "claude")

        with mock.patch.object(ai_agent, "_blog_model_json", fake_model_json):
            out = await ai_agent.generate_seo_blog(title="T", target_words=1200, business_name="Cornus", facts=build_fact_pack(BIZ))
        self.assertIn("F1: The business is called Cornus.", seen["prompt"])
        self.assertIn("State nothing about the business that is not in the facts", seen["prompt"])
        body = out["sections"][0]["content"]
        self.assertNotIn("1999", body)
        self.assertIn("tasting menu", body)
        self.assertEqual(out["droppedClaims"], ["It was founded in 1999."])
        self.assertTrue(out["checkBeforePublishing"])


if __name__ == "__main__":
    unittest.main()
