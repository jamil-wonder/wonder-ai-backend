"""F3: blogs are written in the business's own English."""
import unittest
from unittest import mock

from blog_language import language_prompt_block, localize_spelling, normalize_locale, resolve_blog_language


def biz(**kw):
    return kw


class ResolveTests(unittest.TestCase):
    def test_explicit_profile_choice_wins(self):
        r = resolve_blog_language(biz(blogLanguage="en-US", latest_scrape_result={"language": "en-GB"}))
        self.assertEqual((r["code"], r["source"]), ("en-US", "profile"))

    def test_html_lang_region_is_used_when_there_is_no_address(self):
        for declared in ("en-GB", "en_gb", "EN-GB"):
            r = resolve_blog_language(biz(latest_scrape_result={"language": declared}))
            self.assertEqual((r["code"], r["source"]), ("en-GB", "website language"), declared)
        self.assertEqual(resolve_blog_language(biz(latest_scrape_result={"language": "en-US"}))["code"], "en-US")

    def test_the_address_beats_a_site_builders_default_en_us(self):
        # a London restaurant whose site says <html lang="en-US"> is still British
        r = resolve_blog_language(biz(latest_scrape_result={"language": "en-US", "locations": [{"city": "London", "country": "UK"}]}))
        self.assertEqual((r["code"], r["source"]), ("en-GB", "address"))
        r = resolve_blog_language(biz(latest_scrape_result={"language": "en-GB", "locations": [{"city": "Austin", "country": "US"}]}))
        self.assertEqual(r["code"], "en-US")

    def test_plain_en_falls_through_to_the_address_country(self):
        r = resolve_blog_language(biz(latest_scrape_result={"language": "en", "locations": [{"city": "Sydney", "country": "Australia"}]}))
        self.assertEqual((r["code"], r["source"]), ("en-AU", "address"))

    def test_an_unmapped_country_falls_back_to_the_site_language(self):
        r = resolve_blog_language(biz(latest_scrape_result={"language": "en-IE", "locations": [{"country": "Germany"}]}))
        self.assertEqual((r["code"], r["source"]), ("en-IE", "website language"))

    def test_address_country_maps_gb_for_the_uk(self):
        r = resolve_blog_language(biz(latest_scrape_result={"language": "unknown", "locations": [{"country": "UK"}]}))
        self.assertEqual((r["code"], r["label"]), ("en-GB", "British English"))

    def test_a_non_english_page_uses_the_address_not_the_page_language(self):
        r = resolve_blog_language(biz(latest_scrape_result={"language": "fr-FR", "locations": [{"country": "Canada"}]}))
        self.assertEqual(r["code"], "en-CA")

    def test_domain_ending_is_the_next_hint(self):
        r = resolve_blog_language(biz(url="https://www.example.co.uk/", latest_scrape_result={}))
        self.assertEqual((r["code"], r["source"]), ("en-GB", "domain"))
        self.assertEqual(resolve_blog_language(biz(url="https://shop.com.au"))["code"], "en-AU")

    def test_unknown_defaults_to_british_english(self):
        r = resolve_blog_language(biz(url="https://example.com"))
        self.assertEqual((r["code"], r["source"]), ("en-GB", "default"))
        self.assertEqual(resolve_blog_language({})["code"], "en-GB")

    def test_normalize_locale(self):
        self.assertEqual(normalize_locale("en_gb"), "en-GB")
        self.assertIsNone(normalize_locale("fr-FR"))
        self.assertIsNone(normalize_locale("en"))
        self.assertIsNone(normalize_locale("en-XX"))
        self.assertIsNone(normalize_locale(None))


def sec(text, heading="H"):
    return [{"id": "s", "heading": heading, "label": heading, "content": text}]


class SpellingTests(unittest.TestCase):
    def test_american_to_british(self):
        out, n = localize_spelling(sec("Our favorite color. We organize and optimize. The center of town; an analysis, to analyze it."), "en-GB")
        text = out[0]["content"]
        self.assertEqual(text, "Our favourite colour. We organise and optimise. The centre of town; an analysis, to analyse it.")
        self.assertEqual(n, 6)

    def test_endings_and_case_are_kept(self):
        out, _ = localize_spelling(sec("Colorful, colored and coloring. Organization and ORGANIZERS. Center."), "en-GB")
        self.assertEqual(out[0]["content"], "Colourful, coloured and colouring. Organisation and ORGANISERS. Centre.")

    def test_british_to_american(self):
        out, n = localize_spelling(sec("Our favourite colour and centre. We organise and recognise. Grey skies."), "en-US")
        self.assertEqual(out[0]["content"], "Our favorite color and center. We organize and recognize. Gray skies.")
        self.assertEqual(n, 6)

    def test_headings_are_fixed_too(self):
        out, _ = localize_spelling(sec("Body.", heading="Color guide"), "en-GB")
        self.assertEqual(out[0]["heading"], "Colour guide")

    def test_canadian_keeps_ize_but_takes_our_and_re(self):
        out, _ = localize_spelling(sec("The color center. We organize and recognise things."), "en-CA")
        self.assertEqual(out[0]["content"], "The colour centre. We organize and recognise things.")

    def test_names_in_the_middle_of_a_sentence_are_left_alone(self):
        out, n = localize_spelling(sec("We love Center Parcs and the Color Run. The center is open."), "en-GB")
        self.assertEqual(out[0]["content"], "We love Center Parcs and the Color Run. The centre is open.")
        self.assertEqual(n, 1)

    def test_words_in_the_business_name_or_facts_are_protected(self):
        out, n = localize_spelling(sec("Visit the center for color."), "en-GB", protect_text="The Center Bistro. Open daily.")
        self.assertEqual(out[0]["content"], "Visit the center for colour.")
        self.assertEqual(n, 1)

    def test_words_that_merely_contain_a_stem_are_not_touched(self):
        text = "Colorado, a centered approach? Meter reading, program, license, practice. Organs and gray area."
        out, _ = localize_spelling(sec(text), "en-GB")
        self.assertIn("Colorado", out[0]["content"])
        self.assertIn("Meter reading", out[0]["content"])
        self.assertIn("program", out[0]["content"])
        self.assertIn("grey area", out[0]["content"])

    def test_already_correct_text_is_untouched(self):
        s = sec("Colour and centre and organise.")
        out, n = localize_spelling(s, "en-GB")
        self.assertEqual((out[0]["content"], n), ("Colour and centre and organise.", 0))
        self.assertEqual(s[0]["content"], "Colour and centre and organise.")


class PromptTests(unittest.TestCase):
    def test_prompt_names_the_variety(self):
        self.assertIn("British English (en-GB)", language_prompt_block("en-GB"))
        self.assertIn("holiday not vacation", language_prompt_block("en-GB"))
        self.assertIn("American English (en-US)", language_prompt_block("en-US"))

    def test_generator_uses_it_and_fixes_slips(self):
        import asyncio
        from agents import ai_agent

        seen = {}

        async def fake(**kw):
            seen["prompt"] = kw["prompt"]
            return ({"title": "T", "sections": [{"id": "a", "heading": "Color tips", "content": "Choose a favorite color. Organize the center table."}], "claims": []}, "m", "claude")

        async def run():
            with mock.patch.object(ai_agent, "_blog_model_json", fake):
                return await ai_agent.generate_seo_blog(title="T", target_words=1200, business_name="Cornus", language="en-GB")

        out = asyncio.run(run())
        self.assertIn("British English (en-GB)", seen["prompt"])
        self.assertEqual(out["sections"][0]["heading"], "Colour tips")
        self.assertEqual(out["sections"][0]["content"], "Choose a favourite colour. Organise the centre table.")
        self.assertEqual((out["language"], out["languageFixes"]), ("en-GB", 5))


if __name__ == "__main__":
    unittest.main()
