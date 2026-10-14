"""B2: an answer that came back with no captured sources is flagged, and when
more than half of a run's answers are flagged the run is incomplete."""
import unittest

from agents.phase5.analysis import _safe_provider_result
from agents.phase5.scoring import compute_run_health

PROVIDERS = ("perplexity", "chatgpt", "claude", "gemini")


def cell(missing=False, state="ok"):
    c = {"mentioned": False, "status": "Not Mentioned", "fetch_status": state}
    if missing:
        c["citations_missing"] = True
    return c


def run(per_provider_missing, n=20):
    return {str(i): {"providers": {p: cell(missing=p in per_provider_missing) for p in PROVIDERS}} for i in range(n)}


class FlagTests(unittest.TestCase):
    def test_no_sources_at_all_is_flagged_with_the_raw_text(self):
        out = _safe_provider_result({"status": "Not Mentioned", "llm_response": "Some answer text"})
        self.assertTrue(out["citations_missing"])
        self.assertEqual(out["raw_response_excerpt"], "Some answer text")

    def test_any_source_clears_the_flag(self):
        for field in ("sources", "references", "source_urls"):
            out = _safe_provider_result({"status": "Not Mentioned", field: ["example.com"]})
            self.assertNotIn("citations_missing", out, field)

    def test_a_failed_answer_is_not_a_citation_failure(self):
        out = _safe_provider_result({"fetch_status": "failed", "error": "rate limit"})
        self.assertNotIn("citations_missing", out)

    def test_excerpt_is_capped(self):
        out = _safe_provider_result({"status": "Not Mentioned", "llm_response": "x" * 5000})
        self.assertEqual(len(out["raw_response_excerpt"]), 600)


class RunHealthTests(unittest.TestCase):
    def test_one_model_without_citations_does_not_fail_the_run(self):
        h = run_health = compute_run_health(run({"claude"}))   # 25% of answers
        self.assertFalse(h["incomplete"])
        self.assertFalse(h["citations_missing_majority"])
        self.assertEqual(h["citations_missing_by_provider"], {"claude": 20})

    def test_more_than_half_flagged_makes_the_run_incomplete(self):
        h = compute_run_health(run({"claude", "gemini", "chatgpt"}))   # 75%
        self.assertTrue(h["incomplete"])
        self.assertEqual(h["incomplete_reason"], "citations_missing")
        self.assertAlmostEqual(h["citations_missing_fraction"], 0.75)

    def test_exactly_half_is_not_more_than_half(self):
        self.assertFalse(compute_run_health(run({"claude", "gemini"}))["incomplete"])

    def test_failed_answers_are_not_counted_as_missing_citations(self):
        results = {str(i): {"providers": {"perplexity": cell(), "chatgpt": cell(state="failed"),
                                          "claude": cell(state="failed"), "gemini": cell(state="failed")}} for i in range(4)}
        h = compute_run_health(results)
        self.assertEqual(h["citations_missing"], 0)
        self.assertEqual(h["incomplete_reason"], "failed_answers")

    def test_a_clean_run_has_no_reason(self):
        h = compute_run_health(run(set()))
        self.assertFalse(h["incomplete"])
        self.assertIsNone(h["incomplete_reason"])


if __name__ == "__main__":
    unittest.main()
