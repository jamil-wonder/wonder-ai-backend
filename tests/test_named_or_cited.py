"""B1: a mention is being named in the answer text OR cited as a source."""
import unittest

from agents.phase5 import analysis
from agents.phase5.analysis import _named_in_answer, _safe_provider_result


class NamedInAnswerTests(unittest.TestCase):
    def test_business_name_in_the_answer_counts(self):
        named, ev = _named_in_answer("For fine dining in Belgravia, Cornus is a strong choice. Others exist.", "cornusrestaurant.co.uk", "Cornus")
        self.assertTrue(named)
        self.assertIn("Cornus", ev)

    def test_domain_or_brand_word_counts(self):
        self.assertTrue(_named_in_answer("See cornusrestaurant.co.uk for the menu.", "cornusrestaurant.co.uk", None)[0])
        self.assertTrue(_named_in_answer("Dishoom is famous for its bacon naan.", "www.dishoom.com", None)[0])

    def test_whole_words_only(self):
        self.assertFalse(_named_in_answer("The cornucopia of options is large.", "cornus.co.uk", "Cornus")[0])

    def test_saying_it_is_absent_is_not_a_mention(self):
        self.assertFalse(_named_in_answer("There is no mention of Cornus in these results.", "cornus.co.uk", "Cornus")[0])
        self.assertFalse(_named_in_answer("Cornus does not appear in the top results.", "cornus.co.uk", "Cornus")[0])

    def test_one_negative_sentence_does_not_hide_a_positive_one(self):
        named, _ = _named_in_answer("Cornus is not cheap. Cornus is a Michelin-starred restaurant.", "cornus.co.uk", "Cornus")
        # first sentence is negated, second names it plainly
        self.assertTrue(named)

    def test_empty_or_nameless(self):
        self.assertEqual(_named_in_answer("", "cornus.co.uk", "Cornus"), (False, None))
        self.assertFalse(_named_in_answer("Plenty of restaurants in London.", "cornus.co.uk", "Cornus")[0])


class FlagsSurviveNormalisationTests(unittest.TestCase):
    def test_flags_and_evidence_are_kept(self):
        out = _safe_provider_result({"status": "Mentioned", "named_in_answer": True, "cited_as_source": False, "named_evidence": "Cornus is great."})
        self.assertTrue(out["named_in_answer"])
        self.assertFalse(out["cited_as_source"])
        self.assertEqual(out["named_evidence"], "Cornus is great.")

    def test_missing_flags_default_to_false(self):
        out = _safe_provider_result({"status": "Not Mentioned"})
        self.assertFalse(out["named_in_answer"] or out["cited_as_source"])
        self.assertNotIn("named_evidence", out)


if __name__ == "__main__":
    unittest.main()
