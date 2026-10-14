"""A1: the one Wonder Score formula and the mention rate behind it. Pure maths,
no database and no AI calls."""
import unittest

from agents.phase5.scoring import compute_mention_rate, compute_wonder_score, question_weight

PROVIDERS = ("perplexity", "chatgpt", "claude", "gemini")


def ok(mentioned):
    return {"mentioned": mentioned, "status": "Mentioned" if mentioned else "Not Mentioned", "fetch_status": "ok"}


def failed():
    return {"mentioned": False, "status": "Not Mentioned", "fetch_status": "failed", "error": "rate limit"}


def results(per_model: dict, mentioned_qids=(), n=20):
    out = {}
    for i in range(1, n + 1):
        out[str(i)] = {"id": str(i), "providers": {
            p: (failed() if per_model.get(p) == "failed" else ok(i in mentioned_qids)) for p in PROVIDERS
        }}
    return out


# the default mix: 5 branded (weight 1) + 5 non-branded + 5 local + 5 broad (weight 2) = 35
DEFAULT_MIX = [{"id": str(i), "category": ("branded" if i <= 5 else "non-branded")} for i in range(1, 21)]


class FormulaTests(unittest.TestCase):
    def test_the_pdf_worked_check(self):
        # named in 1 of 20 (branded) -> mentionRate ~ 3; technical 91; market 0 -> ~ 29
        rate = compute_mention_rate(results({}, {1}), DEFAULT_MIX)
        self.assertAlmostEqual(rate, 100 / 35, places=2)
        self.assertEqual(compute_wonder_score(rate, 91, 0), 29.0)

    def test_weights_are_60_30_10(self):
        self.assertEqual(compute_wonder_score(100, 0, 0), 60.0)
        self.assertEqual(compute_wonder_score(0, 100, 0), 30.0)
        self.assertEqual(compute_wonder_score(0, 0, 100), 10.0)
        self.assertEqual(compute_wonder_score(100, 100, 100), 100.0)

    def test_no_score_when_mention_rate_or_website_health_is_missing(self):
        self.assertIsNone(compute_wonder_score(None, 90, 0))
        self.assertIsNone(compute_wonder_score(50, None, 0))
        self.assertIsNotNone(compute_wonder_score(50, 90, None))  # market defaults to 0 until A2

    def test_rounds_half_up_on_exact_ties(self):
        # float rounding disagrees between Python and JS here; whole-number maths doesn't
        self.assertEqual(compute_wonder_score(15.75, 57, 0), 26.6)

    def test_inputs_are_clamped_to_0_100(self):
        self.assertEqual(compute_wonder_score(250, 250, 250), 100.0)
        self.assertEqual(compute_wonder_score(-5, -5, -5), 0.0)


class MentionRateTests(unittest.TestCase):
    def test_question_weights(self):
        self.assertEqual(question_weight("branded"), 1)
        self.assertEqual(question_weight("Branded"), 1)
        for c in ("non-branded", "local-seo", "broad-seo", None, ""):
            self.assertEqual(question_weight(c), 2)

    def test_appearing_only_when_asked_by_name_scores_far_lower(self):
        branded_only = compute_mention_rate(results({}, {1, 2, 3, 4, 5}), DEFAULT_MIX)       # 5 of 5 branded
        unprompted = compute_mention_rate(results({}, set(range(6, 11))), DEFAULT_MIX)       # 5 non-branded
        self.assertAlmostEqual(branded_only, 100 * 5 / 35, places=2)
        self.assertAlmostEqual(unprompted, 100 * 10 / 35, places=2)
        self.assertGreater(unprompted, branded_only)

    def test_a_failed_model_is_missing_not_zero(self):
        # Gemini is down: the rate must equal what the three working models give
        healthy = compute_mention_rate(results({}, {1, 6, 7}), DEFAULT_MIX)
        one_down = compute_mention_rate(results({"gemini": "failed"}, {1, 6, 7}), DEFAULT_MIX)
        self.assertEqual(healthy, one_down)

    def test_averages_over_models_that_answered(self):
        rows = results({}, set())
        for i in range(1, 21):          # only perplexity ever mentions: 1 of 4 models -> 25%
            rows[str(i)]["providers"]["perplexity"] = ok(True)
        self.assertEqual(compute_mention_rate(rows, DEFAULT_MIX), 25.0)

    def test_none_when_no_model_answered(self):
        self.assertIsNone(compute_mention_rate(results({p: "failed" for p in PROVIDERS}), DEFAULT_MIX))
        self.assertIsNone(compute_mention_rate({}, DEFAULT_MIX))

    def test_deterministic(self):
        # A5 regression: the same answers always give the same number
        rows = results({"chatgpt": "failed"}, {1, 8, 13})
        self.assertEqual(compute_mention_rate(rows, DEFAULT_MIX), compute_mention_rate(rows, DEFAULT_MIX))

    def test_vote_share_makes_a_coin_flip_question_count_as_a_fraction(self):
        # A5: one branded question, answered yes in 2 of 3 samples by every model
        rows = {"1": {"id": "1", "providers": {p: {**ok(True), "vote": {"mentioned": 2, "of": 3}} for p in PROVIDERS}}}
        qs = [{"id": "1", "category": "branded"}]
        self.assertAlmostEqual(compute_mention_rate(rows, qs), 66.67, places=2)

    def test_single_sample_answers_still_count_whole(self):
        rows = {"1": {"id": "1", "providers": {p: ok(True) for p in PROVIDERS}}}
        self.assertEqual(compute_mention_rate(rows, [{"id": "1", "category": "branded"}]), 100.0)

    def test_an_answer_that_only_matches_the_site_is_not_a_mention(self):
        # B1: relevance to the site's facts is not the same as being named or cited
        rows = {"1": {"id": "1", "providers": {p: {"status": "Not Mentioned", "fetch_status": "ok", "target_site": {"status": "matched"}} for p in PROVIDERS}}}
        self.assertEqual(compute_mention_rate(rows, [{"id": "1", "category": "branded"}]), 0.0)


if __name__ == "__main__":
    unittest.main()
