"""F2: keywords steer the topic; a keyword phrase appears word for word at most once."""
import unittest
from unittest import mock

from tests.blog_fixtures import valid_sections
from blog_keywords import clean_keyword_list, count_verbatim, countable_phrases, retry_feedback, topics_prompt_block, violations


def sec(text, heading="H"):
    return {"id": "s", "heading": heading, "label": heading, "content": text}


class CountingTests(unittest.TestCase):
    def test_case_punctuation_and_spacing_are_ignored(self):
        s = [sec("Looking for Belgravia fine dining? Belgravia   Fine-Dining is a theme. belgravia fine dining again.")]
        self.assertEqual(count_verbatim(s, ["belgravia fine dining"]), {"belgravia fine dining": 3})

    def test_headings_count_too(self):
        s = [sec("Some text.", heading="Belgravia fine dining"), sec("Belgravia fine dining is discussed.")]
        self.assertEqual(count_verbatim(s, ["Belgravia fine dining"])["Belgravia fine dining"], 2)

    def test_whole_words_only(self):
        s = [sec("Belgravia fine diningroom and Belgravia fine dinings are not the phrase.")]
        self.assertEqual(count_verbatim(s, ["belgravia fine dining"])["belgravia fine dining"], 0)

    def test_single_words_cannot_be_varied_so_they_are_not_counted(self):
        self.assertEqual(countable_phrases(["restaurant", "fine dining", "  "]), ["fine dining"])
        self.assertEqual(count_verbatim([sec("restaurant restaurant restaurant")], ["restaurant"]), {})

    def test_repeats_in_a_row_are_all_counted(self):
        self.assertEqual(count_verbatim([sec("fine dining fine dining fine dining")], ["fine dining"])["fine dining"], 3)

    def test_one_use_is_fine_two_is_a_violation(self):
        self.assertEqual(violations([sec("Try fine dining once.")], ["fine dining"]), {})
        self.assertEqual(violations([sec("Fine dining here. More fine dining there.")], ["fine dining"]), {"fine dining": 2})

    def test_keyword_list_is_cleaned(self):
        self.assertEqual(clean_keyword_list(["A b", "a  B", None, "", "c d"]), ["A b", "c d"])


class PromptTests(unittest.TestCase):
    def test_keywords_are_framed_as_topics_not_phrases(self):
        block = topics_prompt_block(["Belgravia fine dining", "private dining London"])
        self.assertIn("themes, NOT phrases to insert", block)
        self.assertIn("Never repeat a topic phrase word for word more than once", block)
        self.assertIn("Belgravia fine dining", block)

    def test_no_keywords_no_block(self):
        self.assertEqual(topics_prompt_block([]), "")

    def test_retry_note_names_the_offending_phrases(self):
        self.assertIn('"fine dining" (4 times)', retry_feedback({"fine dining": 4}))


class GeneratorLoopTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def draft(text):
        # a well-formed article (F4) whose first section carries the text under test
        return ({"title": "T", "sections": valid_sections(text), "claims": []}, "claude-x", "claude")

    async def _run(self, drafts, keywords=("belgravia fine dining",)):
        from agents import ai_agent
        prompts = []

        async def fake(**kw):
            prompts.append(kw["prompt"])
            return drafts[min(len(prompts) - 1, len(drafts) - 1)]

        with mock.patch.object(ai_agent, "_blog_model_json", fake):
            out = await ai_agent.generate_seo_blog(title="T", target_words=1200, business_name="Cornus",
                                                   primary_keyword="Belgravia fine dining", keywords=list(keywords))
        return out, prompts

    async def test_a_clean_first_draft_is_accepted_without_retry(self):
        out, prompts = await self._run([self.draft("Belgravia fine dining is a theme. Elsewhere we vary the words.")])
        self.assertEqual((out["keywordAttempts"], len(prompts), out["keywordRepeats"]), (1, 1, {}))
        self.assertIn("TOPICS TO COVER", prompts[0])
        self.assertNotIn("Primary keyword:", prompts[0])

    async def test_a_stuffed_draft_is_rejected_and_rewritten(self):
        stuffed = self.draft("Belgravia fine dining here. Belgravia fine dining there. Belgravia fine dining everywhere.")
        clean = self.draft("Belgravia fine dining here. Dinner in this part of London is a treat.")
        out, prompts = await self._run([stuffed, clean])
        self.assertEqual(out["keywordAttempts"], 2)
        self.assertIn("REWRITE NOTE", prompts[1])
        self.assertIn('"Belgravia fine dining" (3 times)', prompts[1])
        self.assertEqual(out["keywordRepeats"], {})
        self.assertNotIn("everywhere", out["sections"][0]["content"])

    async def test_still_stuffed_after_retries_keeps_the_best_and_tells_the_owner(self):
        worst = self.draft("Belgravia fine dining. Belgravia fine dining. Belgravia fine dining. Belgravia fine dining.")
        better = self.draft("Belgravia fine dining. Belgravia fine dining.")
        out, prompts = await self._run([worst, better, worst])
        self.assertEqual(out["keywordAttempts"], 3)
        self.assertEqual(out["keywordRepeats"], {"Belgravia fine dining": 2})         # the best attempt, not the last
        self.assertTrue(out["checkBeforePublishing"][0].startswith("Reword the repeated keyword phrases"))

    async def test_other_keywords_given_as_topics_are_checked_too(self):
        d = self.draft("Private dining London works well. Private dining London is popular. Belgravia fine dining once.")
        out, _ = await self._run([d, d, d], keywords=("private dining london", "belgravia fine dining"))
        self.assertIn("private dining london", out["keywordRepeats"])
        self.assertNotIn("Belgravia fine dining", out["keywordRepeats"])


if __name__ == "__main__":
    unittest.main()
