"""J4: competitor discovery must be told the business's category, specific type
(sub_type) and area, so a pizza place isn't compared with a sushi bar."""
import unittest
from unittest import mock

from agents.phase5 import competitors as comp


class CompetitorProfilePromptTests(unittest.IsolatedAsyncioTestCase):
    async def _prompts(self, profile):
        seen = []

        async def fake_research(claude_prompt, **kw):
            seen.append(claude_prompt)
            return None, "none"

        with mock.patch.object(comp, "_web_research_with_fallback", fake_research):
            await comp.generate_deep_competitor_scores(
                url="https://example-curry.co.uk",
                questions=[{"id": "1", "text": "best curry house in Leeds?"}],
                seed_results={},
                target_profile=profile,
            )
        return "\n".join(seen)

    async def test_sub_type_category_and_area_reach_the_discovery_prompt(self):
        text = await self._prompts({"category": "Restaurant", "sub_type": "Indian", "location": "Leeds, UK"})
        self.assertIn("Specific type: Indian", text)
        self.assertIn("Category: Restaurant", text)
        self.assertIn("Area: Leeds, UK", text)
        self.assertIn("same specific type", text)

    async def test_without_a_profile_nothing_is_invented(self):
        text = await self._prompts(None)
        self.assertNotIn("Specific type:", text)

    async def test_public_suggestions_prompt_includes_the_sub_type(self):
        seen = []

        async def fake_claude(prompt, **kw):
            seen.append(prompt)
            return None

        async def fake_openai(prompt, **kw):
            seen.append(prompt)
            return None

        with mock.patch.object(comp, "_call_claude_web_search_with_retry", fake_claude), \
                mock.patch.object(comp, "_call_openai_with_retry", fake_openai):
            await comp.generate_public_competitor_suggestions(
                url="https://example-curry.co.uk", questions=[{"text": "q"}], category="Restaurant",
                sub_type="Indian", location="Leeds, UK",
            )
        self.assertTrue(seen)
        self.assertIn("Specific type: Indian", seen[0])
        self.assertIn("MUST be that same specific type", seen[0])


if __name__ == "__main__":
    unittest.main()
