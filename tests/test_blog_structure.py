"""F4: 800-1,200 words, paragraphs <=100 words, H2/H3, FAQ of 3-5, one internal link."""
import unittest
from unittest import mock

from blog_structure import (
    body_word_count, faq_question_count, fix_structure, internal_link_count, structure_feedback,
    structure_problems, structure_prompt_block,
)
from tests.blog_fixtures import valid_sections

LINKS = ["https://cornus.co.uk/menu", "https://cornus.co.uk/private-dining"]


def words(n):
    return " ".join(["word"] * n)


class FixTests(unittest.TestCase):
    def test_long_paragraphs_are_split_at_a_sentence(self):
        sentence = "This sentence has exactly eight words in it."
        para = " ".join([sentence] * 20)                     # 160 words
        out, fixes = fix_structure([{"heading": "H", "content": para}], None)
        parts = out[0]["content"].split("\n\n")
        self.assertGreater(len(parts), 1)
        self.assertTrue(all(len(p.split()) <= 100 for p in parts))
        self.assertTrue(all(p.endswith(".") for p in parts))
        self.assertEqual(fixes["paragraphsSplit"], len(parts) - 1)
        self.assertEqual(" ".join(parts), para)               # nothing lost

    def test_headings_bullets_and_short_paragraphs_are_untouched(self):
        content = "### A heading\n\n- one\n- two\n\nShort paragraph here."
        out, fixes = fix_structure([{"heading": "H", "content": content}], None)
        self.assertEqual(out[0]["content"], content)
        self.assertEqual(fixes["paragraphsSplit"], 0)

    def test_one_endless_sentence_is_left_whole(self):
        out, _ = fix_structure([{"heading": "H", "content": words(150) + "."}], None)
        self.assertEqual(len(out[0]["content"].split("\n\n")), 1)

    def test_faq_is_trimmed_to_five(self):
        faq = "\n\n".join(f"### Q{n}?\n\nAnswer {n}." for n in range(1, 8))
        out, fixes = fix_structure([{"heading": "Frequently asked questions", "content": faq}], None)
        self.assertEqual(faq_question_count(out), 5)
        self.assertEqual(fixes["faqsTrimmed"], 2)
        self.assertIn("### Q5?", out[0]["content"])
        self.assertNotIn("### Q6?", out[0]["content"])

    def test_only_one_internal_link_survives_and_other_links_are_unlinked(self):
        text = ("See [our menu](https://cornus.co.uk/menu) and [private dining](https://cornus.co.uk/private-dining) "
                "or [Wikipedia](https://en.wikipedia.org/wiki/Dining).")
        out, fixes = fix_structure([{"heading": "H", "content": text}], LINKS)
        self.assertEqual(out[0]["content"], "See [our menu](https://cornus.co.uk/menu) and private dining or Wikipedia.")
        self.assertEqual(fixes["linksRemoved"], 2)
        self.assertEqual(internal_link_count(out, LINKS), 1)

    def test_with_no_internal_pages_every_link_is_removed(self):
        out, _ = fix_structure([{"heading": "H", "content": "See [this](https://x.com/a)."}], None)
        self.assertEqual(out[0]["content"], "See this.")

    def test_paths_and_full_urls_both_match(self):
        s = [{"heading": "H", "content": "Read [more](https://www.cornus.co.uk/menu/)."}]
        self.assertEqual(internal_link_count(s, ["/menu"]), 1)
        self.assertEqual(internal_link_count(s, ["https://cornus.co.uk/menu"]), 1)
        self.assertEqual(internal_link_count(s, ["/contact"]), 0)


class ProblemTests(unittest.TestCase):
    def test_a_well_formed_article_has_no_problems(self):
        s = valid_sections(internal_link="https://cornus.co.uk/menu")
        self.assertEqual(structure_problems(s, LINKS), [])

    def test_length_limits(self):
        short = [{"heading": "a", "content": words(300)}] * 5
        self.assertTrue(any("must be 800-1200" in p for p in structure_problems(short, None)))
        long_ = valid_sections() + [{"heading": "x", "content": words(400)}]
        self.assertGreater(body_word_count(long_), 1200)
        self.assertTrue(any("must be 800-1200" in p for p in structure_problems(long_, None)))

    def test_too_few_faqs_and_no_link_and_too_few_sections(self):
        s = valid_sections()
        s[-1]["content"] = "### Only one?\n\nYes."
        problems = structure_problems(s, LINKS)
        self.assertTrue(any("FAQ has 1 questions" in p for p in problems))
        self.assertTrue(any("no internal link" in p for p in problems))
        self.assertTrue(any("sections" in p for p in structure_problems(s[:2], None)))

    def test_no_link_is_demanded_when_the_business_has_no_pages(self):
        self.assertEqual(structure_problems(valid_sections(), None), [])
        self.assertEqual(structure_problems(valid_sections(), []), [])


class PromptTests(unittest.TestCase):
    def test_prompt_states_every_rule(self):
        block = structure_prompt_block(LINKS)
        for needle in ("800-1200", "at most 100 words", "### Sub-heading", "3-5 questions", "exactly ONE internal link", LINKS[0]):
            self.assertIn(needle, block)
        self.assertIn("Do not include any links", structure_prompt_block([]))

    def test_feedback_lists_the_problems(self):
        self.assertIn("it is 300 words", structure_feedback(["it is 300 words - it must be 800-1200"]))


class GeneratorTests(unittest.IsolatedAsyncioTestCase):
    async def _gen(self, drafts, internal_links=None):
        from agents import ai_agent
        prompts = []

        async def fake(**kw):
            prompts.append(kw["prompt"])
            return drafts[min(len(prompts) - 1, len(drafts) - 1)]

        with mock.patch.object(ai_agent, "_blog_model_json", fake):
            out = await ai_agent.generate_seo_blog(title="T", target_words=1500, business_name="Cornus", internal_links=internal_links)
        return out, prompts

    @staticmethod
    def draft(sections):
        return ({"title": "T", "sections": sections, "claims": []}, "claude-x", "claude")

    async def test_a_good_draft_is_accepted_and_the_prompt_carries_the_rules(self):
        out, prompts = await self._gen([self.draft(valid_sections(internal_link=LINKS[0]))], LINKS)
        self.assertEqual((out["keywordAttempts"], out["structureProblems"]), (1, []))
        self.assertIn("exactly ONE internal link", prompts[0])
        self.assertIn("800-1200", prompts[0])
        self.assertNotIn("15 TIMES", prompts[0])
        self.assertTrue(800 <= out["wordCount"] <= 1200)

    async def test_a_too_short_draft_is_rewritten_with_a_note(self):
        short = self.draft([{"id": "a", "heading": "H", "content": words(200)}])
        good = self.draft(valid_sections(internal_link=LINKS[0]))
        out, prompts = await self._gen([short, good], LINKS)
        self.assertEqual(out["keywordAttempts"], 2)
        self.assertIn("STRUCTURE NOTE", prompts[1])
        self.assertIn("it is", prompts[1])
        self.assertEqual(out["structureProblems"], [])

    async def test_long_paragraphs_and_extra_links_are_fixed_without_a_rewrite(self):
        secs = valid_sections(internal_link=LINKS[0])
        secs[0]["content"] += "\n\n" + " ".join(["Another useful sentence about planning ahead."] * 18) + " Visit [elsewhere](https://other.com/x)."
        out, prompts = await self._gen([self.draft(secs)], LINKS)
        self.assertEqual(len(prompts), 1)
        self.assertGreater(out["structureFixes"]["paragraphsSplit"], 0)
        self.assertEqual(out["structureFixes"]["linksRemoved"], 1)
        self.assertTrue(all(len(p.split()) <= 100 for sec in out["sections"] for p in sec["content"].split("\n\n") if not p.startswith("#")))

    async def test_a_draft_that_never_complies_is_kept_and_the_owner_is_told(self):
        short = self.draft([{"id": "a", "heading": "H", "content": words(200)}])
        out, prompts = await self._gen([short, short, short], LINKS)
        self.assertEqual(len(prompts), 3)
        self.assertTrue(out["structureProblems"])
        self.assertTrue(out["checkBeforePublishing"][0].startswith("This draft doesn't fully meet the length and structure rules"))


if __name__ == "__main__":
    unittest.main()
