"""A failed Gemini answer keeps what Google really said (with the key removed)."""
import unittest
from unittest import mock

from agents.phase5 import analysis, providers
from agents.phase5.providers import Phase5RateLimitError
from utils.gemini_utils import describe_gemini_error


class FakeGoogleError(Exception):
    code = 429
    status = "RESOURCE_EXHAUSTED"


class DescribeTests(unittest.TestCase):
    def test_code_status_and_message_are_kept(self):
        text = describe_gemini_error(FakeGoogleError("429 RESOURCE_EXHAUSTED. You exceeded your current quota."))
        self.assertIn("429 RESOURCE_EXHAUSTED", text)
        self.assertIn("exceeded your current quota", text)

    def test_the_api_key_never_appears(self):
        fake_key = "AI" + "za" + "Sy" + "ABCDEFGHIJKLMNOPQRSTUV1234567"        # built in pieces: not a real key
        text = describe_gemini_error(FakeGoogleError(f"bad key={fake_key} in url"))
        self.assertNotIn("AIza", text)
        self.assertIn("<key>", text)

    def test_it_is_short_and_never_raises(self):
        self.assertLessEqual(len(describe_gemini_error(Exception("x" * 5000))), 300)
        self.assertIsInstance(describe_gemini_error(None), str)


class StoredAnswerTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_rate_limit_failure_stores_the_real_reason(self):
        async def boom(*a, **k):
            raise FakeGoogleError("429 RESOURCE_EXHAUSTED: billing quota exhausted")

        def raise_it(*a, **k):
            raise FakeGoogleError("429 RESOURCE_EXHAUSTED: billing quota exhausted")

        with mock.patch.object(providers, "generate_with_fallback", raise_it):
            with self.assertRaises(Phase5RateLimitError) as ctx:
                await providers._call_gemini_with_timeout(object(), "q", object(), timeout_sec=10)
        self.assertIn("billing quota exhausted", ctx.exception.detail)
        cell = analysis._safe_provider_result(ctx.exception)
        self.assertEqual(cell["fetch_status"], "failed")
        self.assertEqual(cell["error"], "Gemini rate limit reached. Please retry shortly.")      # the friendly line is unchanged
        self.assertIn("billing quota exhausted", cell["error_detail"])                        # ... but the reason is no longer lost

    async def test_a_normal_failure_without_detail_still_works(self):
        cell = analysis._safe_provider_result(RuntimeError("boom"))
        self.assertEqual(cell["fetch_status"], "failed")
        self.assertNotIn("error_detail", cell)


if __name__ == "__main__":
    unittest.main()
