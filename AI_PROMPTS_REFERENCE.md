# Wonder AI Backend - Prompt Reference Manual

> **Document Version:** 1.0  
> **Audience:** Developers & System Architects  
> **Source Directory:** `backend/agents/` & `backend/engines/`  
> **Supported Providers:** OpenAI ChatGPT, Anthropic Claude, Google Gemini, Perplexity AI

---

## Executive Summary

This document serves as the authoritative technical reference for all LLM prompt templates used across the Wonder AI backend architecture. Each entry includes:
- **Source File Path & Line Numbers**
- **Function & Variable Name**
- **Target AI Provider / Model**
- **Verbatim Prompt Text** (including exact variable placeholders like `{domain}`, `{brand_name}`, `{url}`)

---

## Table of Contents

1. [Content & Homepage Evaluation](#1-content--homepage-evaluation)
2. [Search Query & Question Generation](#2-search-query--question-generation)
3. [AI Visibility, Ranking & Prompt Audit](#3-ai-visibility-ranking--prompt-audit)
4. [Competitor Discovery & Validation](#4-competitor-discovery--validation)
5. [AI Brand Insights & Perception](#5-ai-brand-insights--perception)
6. [Blog & SEO Content Generation](#6-blog--seo-content-generation)
7. [Website Extraction & Vision](#7-website-extraction--vision)

---

<a id="1-content--homepage-evaluation"></a>
## 1. Content & Homepage Evaluation

### 1.1 Homepage Content Analysis Prompt
* **File:** `backend/agents/content_agent.py` (Lines 56–75)
* **Function:** `analyze_url_content()`
* **Variable:** `prompt`
* **Target Model:** Google Gemini (`gemini-flash`)

```python
prompt = f"""
Act as an expert SEO and content strategist evaluator. Review the following text extracted from a business homepage.
Analyze it for readability, tone/sentiment, target audience clarity, core SEO strengths/weaknesses, and score. Provide concrete, short, and highly readable feedback.
CRITICAL: All generated text (strengths, weaknesses, advice) MUST be brief, easy to understand, and visually clean. Do not generate long, complicated essays or "rubbish". Do NOT use quotes (' or ") around your output text. Do NOT use markdown bolding (**). Focus on quick, punchy insights in plain text.
Also provide a score breakdown out of 100 for sub-categories that make up the total seoScore (e.g. Structure, Keywords, Relevance, Readability).
Always return your output as strictly valid JSON according to this schema:
{{
    "seoScore": int (0-100),
    "scoreBreakdown": {{ "Structure": int, "Keywords": int, "Relevance": int, "Readability": int }},
    "readability": string (MUST be exactly ONE of these 5 options: "Very Easy", "Easy", "Moderate", "Hard", "Very Hard"),
    "sentiment": string (The Tone/Sentiment. YOU MUST INCLUDE an appropriate emoji at the beginning followed by exactly ONE descriptive word. Example: "😊 Welcoming", "👔 Professional", "🔥 Energetic"),
    "targetAudience": string (A very short, clear description of who the text targets, nicely formatted),
    "strengths": [list of short, concise plain text strings without quotes],
    "weaknesses": [list of short, concise plain text strings without quotes],
    "actionableAdvice": [list of short, concise plain text strings without quotes]
}}

Website Scraped Text:
{text}
"""
```

---

<a id="2-search-query--question-generation"></a>
## 2. Search Query & Question Generation

### 2.1 Multi-Category Search Prompt Generation
* **File:** `backend/agents/phase5/questions.py` (Lines 263–331)
* **Function:** `generate_brand_questions()`
* **Variable:** `base_prompt`
* **Target Model:** Perplexity AI (Primary) / OpenAI ChatGPT (Fallback)

```python
base_prompt = f"""
You are a senior SEO search-intent strategist for AI answer engines.

Business context:
{context_block}

Generate a candidate pool of search questions for testing AI visibility.
The system will validate and keep the best {sum(counts.values())} questions.

Required structure:
- branded_queries: exactly {candidate_counts["branded"]} candidate questions.
- non_branded_queries: exactly {candidate_counts["nonBranded"]} candidate questions.
- local_seo_queries: exactly {candidate_counts["localSeo"]} candidate questions.
- broad_seo_queries: exactly {candidate_counts["broadSeo"]} candidate questions.

Important quality rules:
- Do not write thin generic prompts such as "Best Hotel in City of London?".
- Do not write broken prompts such as "Best Hotel options for Angler?".
- Do not write general advice prompts such as "What should I look for...", "How do I compare...", "How do I check...", or "What is the best way to...".
- Every question must include a useful buyer intent such as reviews, booking, menu/service details, suitability, occasion, price/value, comparison, trust, amenities, or availability.
- Prefer natural full questions a real customer would type into Google, ChatGPT, Claude, or Perplexity.
- If previous questions are provided below, do not repeat them and do not create near-duplicates with only tiny wording changes.

Previous questions to avoid:
{avoid_block if avoid_block else "- None"}

Rules for branded_queries:
- Each question MUST include the exact business name: "{brand_name}".
- Each question must also use the real category or location context.
- Make them useful for checking brand understanding, reviews, menu/services, location, trust, booking, or suitability.
- They must be SEO-friendly questions a real customer could ask.

Rules for non_branded_queries:
- Do NOT include "{brand_name}", "{domain}", the website URL, or any obvious brand variant.
- These are category/service discovery prompts and do NOT have to include the location.
- Every question must be specific to this category: "{category}" or the listed services.
- Cover discovery, comparison, quality, reviews, booking/availability, price/value, occasion/use-case, and trust intent.
- Write prompts that could surface recommended providers, competitors, or ranked options.
- Prefer patterns like "Which [category]...", "What [category] options...", "Are there [category]...", "Where can I book...".
- Do NOT write advice/how-to prompts that only teach the user how to choose, compare, or check a provider.
- Avoid repeated wording patterns.

Rules for local_seo_queries:
- Do NOT include "{brand_name}", "{domain}", the website URL, or any obvious brand variant.
- Every question MUST include the same business location: "{location}".
- Every question must be specific to this category: "{category}".
- Use service/details from the context when available.
- Cover discovery, comparison, quality, reviews, booking/availability, price/value, occasion/use-case, and trust intent.
- Do not use placeholder words such as "business", "company", or "place".
- Do not produce generic prompts like "best restaurant near me" unless the exact location and category/service detail are included.
- Avoid repeated wording patterns.

Rules for broad_seo_queries:
- Do NOT include "{brand_name}", "{domain}", the website URL, or any obvious brand variant.
- Identify the actual real-world neighboring areas, neighborhoods, boroughs, or districts surrounding "{location}" (for example, if "{location}" is "City of London", surrounding areas could include "Shoreditch", "Southwark", "Holborn", "Clerkenwell", "Whitechapel", etc.).
- Write questions using these actual neighboring location names (e.g., "Do people in [Neighboring Area] recommend any good [Category]?" or "What are the best [Category] options around [Neighboring Area]?").
- The goal is to see if the AI engine recommends "{brand_name}" to people searching from these nearby surrounding areas.
- Do NOT use the exact string "{location}" in these questions. Instead, use the real surrounding areas.
- Every question must be specific to this category: "{category}" or the listed services.
- Avoid repeated wording patterns.

Return ONLY valid JSON:
{{
  "branded_queries": ["...", "... exactly {candidate_counts["branded"]}"],
  "non_branded_queries": ["...", "... exactly {candidate_counts["nonBranded"]}"],
  "local_seo_queries": ["...", "... exactly {candidate_counts["localSeo"]}"],
  "broad_seo_queries": ["...", "... exactly {candidate_counts["broadSeo"]}"]
}}
"""
```

---

<a id="3-ai-visibility-ranking--prompt-audit"></a>
## 3. AI Visibility, Ranking & Prompt Audit

### 3.1 Direct Competitor Validation Filter
* **File:** `backend/agents/phase5/analysis.py` (Lines 144–178)
* **Function:** `_validate_direct_competitors()`
* **Variable:** `prompt`
* **Target Model:** Google Gemini (with Search Grounding)

```python
prompt = f"""
You are a market intelligence analyst.
Determine which candidate domains are DIRECT competitors to the target business for this exact search intent.

Query: '{query_text}'
Target domain: '{target_domain}'
Candidate domains: {candidate_domains}

Direct competitor rules:
- Same primary business category/service as the target.
- Similar customer intent fit for this query.
- If query is local intent (near me/nearby/city/area), competitor should be in relevant nearby geography.
- Reject marketplaces/OTAs/directories/review or media platforms, unless they are the same primary business type as target.
- Never include the target domain.

Return JSON only:
{{
  "validated": [
    {{
      "domain": "example.com",
      "position": 1,
      "category_overlap": 0,
      "geo_overlap": 0,
      "confidence": 0,
      "reason": "short reason"
    }}
  ]
}}

Constraints:
- Max 5 items.
- position must be 1..10 or null.
- overlap/confidence must be integers 0..100.
- JSON only, no markdown.
"""
```

### 3.2 Single Question OpenAI / Claude Analysis Prompt
* **File:** `backend/agents/phase5/analysis.py` (Lines 250–291)
* **Function:** `_analyze_single_question_openai()`
* **Variable:** `prompt`
* **Target Model:** OpenAI ChatGPT / Anthropic Claude

```python
prompt = f"""
Analyze this user-intent query for brand visibility using your general web/search understanding.

Query: '{question['text']}'
Target domain: '{domain}'
Target site context to check separately:
{target_context_block}

Return strict JSON only with this schema:
{{
  "target": {{
    "mentioned": true or false,
    "position": <1-10 or null>,
    "source_domains": ["domain.com"]
  }},
  "references": ["domain1.com", "domain2.com"],
  "idea_candidates": ["competitor.com"],
  "ranked_competitors": [
    {{"domain": "competitor.com", "position": 1, "evidence": "short reason"}}
  ],
  "reasoning": "1 short sentence",
  "concise_answer": "Natural user-facing answer to the query (90-180 words), mentioning top options and practical guidance.",
  "target_site_check": {{
    "status": "matched | partial | no_match",
    "matched_facts": ["facts from target site context that relate to the query"],
    "missing_facts": ["important facts missing from target site context"],
    "summary": "short explanation of whether the target site supports this query"
  }}
}}

Rules:
- Keep lists short and realistic.
- target.mentioned must be true ONLY if the target appears in organic answer/search evidence, not because target site context was provided.
- Sources/references must be third-party domains only.
- Do not count the provided target site context as organic visibility.
- Evaluate target_site_check only from the provided target site context.
- Do not include target domain in competitors.
- position must be 1..10 or null.
- concise_answer must sound like a real assistant reply to the query (no analytics wording, no "mentioned/not mentioned" wording).
- concise_answer should be compact but useful: top picks, brief why, and one practical tip.
- JSON only.
"""
```

### 3.3 Grounded Search Probe Prompt
* **File:** `backend/agents/phase5/analysis.py` (Line 778)
* **Function:** `analyze_single_question_multi()` -> `_run_grounded_probe()`
* **Variable:** `probe_prompt`
* **Target Model:** Google Gemini (Search Grounding)

```python
probe_prompt = f"Use live Google Search for this query and list top evidence domains only: '{question['text']}'."
```

### 3.4 Target Domain Verification Prompt
* **File:** `backend/agents/phase5/analysis.py` (Lines 826–837)
* **Function:** `analyze_single_question_multi()` -> `_run_target_verification()`
* **Variable:** `verify_prompt`
* **Target Model:** Google Gemini (Search Grounding)

```python
verify_prompt = f"""
Use live Google Search for this exact query and check if the target domain appears in top results.
Query: '{question['text']}'
Target domain: '{domain}'

Return strict JSON only:
{{
  "mentioned": true or false,
  "position": <1-10 or null>,
  "sources": ["domain.com"]
}}
"""
```

### 3.5 Chat-Style Verification Prompt
* **File:** `backend/agents/phase5/analysis.py` (Lines 883–899)
* **Function:** `analyze_single_question_multi()` -> `_run_chat_style_verification()`
* **Variable:** `chat_prompt`
* **Target Model:** Google Gemini

```python
chat_prompt = f"""
Query: '{question['text']}'
Target domain: '{domain}'

Return JSON only:
{{
  "mentioned": true or false,
  "position": <1-10 or null>,
  "references": ["domain.com"],
  "reason": "short reason"
}}

Rules:
- Use plain reasoning like Gemini chat style.
- No markdown.
- If uncertain, set mentioned=false.
"""
```

### 3.6 Standard Multi-Model AI Visibility Audit Prompt
* **File:** `backend/agents/phase5/analysis.py` (Lines 1002–1043)
* **Function:** `analyze_single_question_multi()`
* **Variable:** `prompt`
* **Target Model:** Google Gemini (Search Grounding)

```python
prompt = f"""
You are an expert brand visibility evaluator. Use live Google Search to answer this query:
'{question['text']}'

Target brand domain: '{domain}' (brand token: '{domain.split('.')[0]}').
Target site context to check separately:
{target_context_block}
Evaluate top search evidence and produce strict JSON only.

JSON schema:
{{
    "target": {{
        "mentioned": true or false,
        "position": <1-10 or null>,
        "source_domains": ["<domain>"]
    }},
    "references": ["<domain1.com>", "<domain2.com>", "<domain3.com>", "... up to 20"],
    "idea_candidates": ["<potential-competitor-domain.com>", "... up to 8"],
    "ranked_competitors": [
        {{"domain": "<competitor.com>", "position": <1-10>, "evidence": "short reason"}}
    ],
    "reasoning": "1 short sentence",
    "concise_answer": "Natural user-facing answer to the query (90-180 words), mentioning top options and practical guidance.",
    "target_site_check": {{
        "status": "matched | partial | no_match",
        "matched_facts": ["facts from target site context that relate to the query"],
        "missing_facts": ["important facts missing from target site context"],
        "summary": "short explanation of whether the target site supports this query"
    }}
}}

Rules:
- 'references' must contain real web domains from observed evidence.
- target.mentioned must be true ONLY if the target appears in organic search evidence, not because target site context was provided.
- Do not include the target domain in ranked_competitors.
- Do not count the provided target site context as organic visibility.
- Evaluate target_site_check only from the provided target site context.
- Keep ranked_competitors to max 5 entries.
- Output raw JSON only. No markdown.
- concise_answer must sound like a real assistant reply to the query (no analytics wording, no "mentioned/not mentioned" wording).
- concise_answer should be compact but useful: top picks, brief why, and one practical tip.
"""
```

### 3.7 Legacy Gemini Analysis Prompt
* **File:** `backend/agents/phase5/analysis.py` (Line 1507)
* **Function:** `_legacy_fast_gemini_analysis()`
* **Variable:** `prompt`
* **Target Model:** Google Gemini (Search Grounding)

```python
prompt = f"Using Google Search, answer this question: '{q['text']}'. Provide a comprehensive answer with sources."
```

---

<a id="4-competitor-discovery--validation"></a>
## 4. Competitor Discovery & Validation

### 4.1 Strict Competitor Candidate Validation Prompt
* **File:** `backend/agents/phase5/competitors.py` (Lines 139–184)
* **Function:** `_validate_competitor_candidates()`
* **Variable:** `prompt`
* **Target Model:** Anthropic Claude (Web Search) / OpenAI (Fallback)

```python
prompt = f"""
    You are a strict competitor validation analyst.
    Use live web search and validate only TRUE direct competitors.

Target domain: {target_domain}
Target context: {json.dumps(compact_context)}
Search intents: {json.dumps(query_texts[:20])}
Candidate domains: {json.dumps(candidates[:20])}

Return JSON only:
{{
  "competitors": [
    {{
      "domain": "example.com",
      "url": "https://example.com/",
      "name": "Business name",
      "is_direct_competitor": true,
      "competitor_type": "independent_business",
      "niche_match": 0,
      "business_model_match": 0,
      "score": 0,
      "position": null,
      "evidence": "short reason"
    }}
  ]
}}

Rules:
    - Include only same-niche, same customer-intent competitors.
    - A valid competitor must have both:
        1) niche_match >= 80
        2) business_model_match >= 80
    - competitor_type must be one of:
        independent_business, marketplace, directory, social_profile, review_listing, media, other
    - Prefer independent_business.
    - Reject platforms/profiles/listings when they are not actual competing businesses.
    - Use search results to verify the actual business homepage and business name.
    - url must be the official homepage or best official business page, not a directory/profile route.
- Exclude the target domain.
- Max 5 domains.
    - is_direct_competitor must be true only for real same-niche alternatives.
    - niche_match and business_model_match must be integers 0..100.
- score must be integer 0..100.
- position must be 1..10 or null.
- JSON only.
"""
```

### 4.2 Deep Competitor Market Intelligence Prompt
* **File:** `backend/agents/phase5/competitors.py` (Lines 453–489)
* **Function:** `generate_deep_competitor_scores()`
* **Variable:** `prompt`
* **Target Model:** Anthropic Claude (Web Search) / OpenAI

```python
prompt = f"""
You are a market intelligence analyst.
Use live web search and these user-intent queries to identify direct competitors for the target.

Target domain: '{domain}'
Queries: {json.dumps(compact_questions)}
Candidate domains from first-pass visibility analysis: {json.dumps(top_candidates)}

Return JSON only:
{{
  "competitors": [
    {{
      "name": "Business name",
      "domain": "example.com",
      "url": "https://example.com/",
      "position": 1,
      "score": 0,
      "confidence": "high",
      "evidence": "short reason"
    }}
  ]
}}

Rules:
- Max 5 competitors.
- Direct competitors only (same intent/category overlap).
- Prefer independent business websites that users can choose instead of the target.
- Use the official business homepage or best official business URL.
- Do not return review/listing/profile/article URLs as competitor URLs.
- Avoid platform/profile/listing style domains when they are not actual competing businesses.
- Exclude target domain.
- name must be the competitor's public business name, not a route title.
- url must match the returned domain.
- score must be integer 0..100.
- position must be 1..10 or null.
- JSON only.
"""
```

### 4.3 SaaS Onboarding Public Competitor Suggestions Prompt
* **File:** `backend/agents/phase5/competitors.py` (Lines 662–699)
* **Function:** `generate_public_competitor_suggestions()`
* **Variable:** `prompt`
* **Target Model:** Anthropic Claude (Web Search) / OpenAI

```python
prompt = f"""
You are a precise market-research assistant for a SaaS onboarding flow.
Use live web search and return direct competitor business websites for this target.

Target:
- Domain: {domain}
- Business name: {business_name or "unknown"}
- Category: {category or "unknown"}
- Location: {location or "unknown"}
- Description: {description[:240] or "unknown"}

Search intents:
{json.dumps(compact_questions)}

Return JSON only:
{{
  "competitors": [
    {{
      "name": "Official business name",
      "domain": "competitor.com",
      "url": "https://competitor.com/",
      "score": 90,
      "evidence": "Why this is a direct competitor"
    }}
  ]
}}

Rules:
- Return exactly {desired_count} competitors if direct competitors can be verified.
- Return fewer only when fewer verified direct competitors are found.
- Direct competitors must match the category and customer intent.
- Prefer businesses in or near the specified location.
- Use the official homepage or official business website.
- Do not include review sites, directories, booking platforms, articles, social media, or the target domain.
- Do not invent domains.
- name must be the real public business name.
- score must be an integer from 70 to 95 based on competitor strength.
"""
```

---

<a id="5-ai-brand-insights--perception"></a>
## 5. AI Brand Insights & Perception

### 5.1 Perplexity Brand Perception Summary Prompt
* **File:** `backend/agents/phase5/competitors.py` (Lines 319–336)
* **Function:** `generate_brand_perception_summary()`
* **Variable:** `prompt`
* **Target Model:** Perplexity AI

```python
prompt = f"""
You are a senior brand copywriter.
Write ONE clear, human-friendly paragraph that describes what this business feels like to a normal customer.

Target domain: {domain}
Analysis data: {json.dumps(compact_items)}

Requirements:
- 95 to 140 words.
- Plain English. Easy to understand. No technical language.
- Natural narrative paragraph (not bullets).
- Explain: what kind of business it appears to be, where it seems to operate, what style/tone it presents, and who it is likely for.
- Mention trust/value signals in simple words (for example: clear menu, modern feel, professional tone, local relevance).
- Make it sound like a concise profile someone can read in 10 seconds.
- Do not fabricate exact facts not implied by data; if uncertain, use cautious wording like "appears to".
- Do not mention internal fields, rankings, percentages, or prompt metrics.
- Output plain text only.
"""
```

### 5.2 Perplexity Multi-Platform Intelligence Insights Prompt
* **File:** `backend/agents/ai_agent.py` (Lines 530–551)
* **Function:** `get_ai_insights()`
* **Variable:** `prompt`
* **Target Model:** Perplexity AI

```python
prompt = f"""
You are an AI research assistant. A user has a local business or website and wants to know what you and the internet know about it.
The business name is "{business_name}" and its website is "{url}".

Please provide a balanced, medium-detail summary of what is known about this business from major platforms like Google, Reddit, YouTube, Wikipedia, Google News, or other public sources based on your training data.

You MUST respond in valid JSON format matching exactly this structure:
{{
    "modelName": "Perplexity",
    "isKnown": true or false,
    "summary": "3-4 sentences summarizing what is known.",
    "sentiment": "Positive" | "Neutral" | "Negative" | "Mixed" | "Unknown",
    "platforms": ["Google", "Reddit", "Facebook", "Wikipedia"],
    "evidence": ["4-8 bullet points with specific external facts or mentions"]
}}

Rules:
- Keep summary and evidence similar depth to a professional analyst brief.
- Prefer externally verifiable mentions over vague statements.
- Return exactly 4-8 evidence bullets.
- JSON only.
"""
```

### 5.3 ChatGPT Multi-Platform Insights Prompt
* **File:** `backend/agents/ai_agent.py` (Lines 579–598)
* **Function:** `get_ai_insights_openai()`
* **Variable:** `prompt`
* **Target Model:** OpenAI ChatGPT

```python
prompt = f"""
You are an AI research assistant. A user has a local business or website and wants to know what is known online.
The business name is "{business_name}" and website is "{url}".

    Respond in valid JSON with this exact shape:
{{
  "modelName": "ChatGPT",
  "isKnown": true or false,
        "summary": "3-4 sentences summarizing what is known.",
  "sentiment": "Positive" | "Neutral" | "Negative" | "Mixed" | "Unknown",
  "platforms": ["Google", "Reddit", "YouTube", "Wikipedia"],
        "evidence": ["4-8 bullet points with specific external facts or mentions"]
}}

    Rules:
    - Keep summary and evidence at medium detail, matching a concise analyst brief.
    - Prefer externally verifiable mentions over generic claims.
    - Return exactly 4-8 evidence bullets.
    - JSON only.
"""
```

### 5.4 Claude Multi-Platform Insights Prompt
* **File:** `backend/agents/ai_agent.py` (Lines 622–641)
* **Function:** `get_ai_insights_claude()`
* **Variable:** `prompt`
* **Target Model:** Anthropic Claude

```python
prompt = f"""
You are an AI research assistant. A user has a local business or website and wants to know what is known online.
The business name is "{business_name}" and website is "{url}".

Respond in valid JSON with this exact shape:
{{
  "modelName": "Claude",
  "isKnown": true or false,
  "summary": "3-4 sentences summarizing what is known.",
  "sentiment": "Positive" | "Neutral" | "Negative" | "Mixed" | "Unknown",
  "platforms": ["Google", "Reddit", "YouTube", "Wikipedia"],
  "evidence": ["4-8 bullet points with specific external facts or mentions"]
}}

Rules:
- Keep summary and evidence at medium detail, matching a concise analyst brief.
- Prefer externally verifiable mentions over generic claims.
- Return exactly 4-8 evidence bullets.
- JSON only.
"""
```

---

<a id="6-blog--seo-content-generation"></a>
## 6. Blog & SEO Content Generation

### 6.1 Full SEO Article Generator Prompt
* **File:** `backend/agents/ai_agent.py` (Lines 828–863)
* **Function:** `generate_seo_blog()`
* **Variable:** `prompt`
* **Target Model:** Configurable (`chatgpt` / `claude` / `perplexity`)

```python
prompt = f"""
You are an expert SEO blog strategist and conversion copywriter.
Create a comprehensive, in-depth SEO blog article as structured JSON.

Blog brief:
- Title: {title}
- Business / Brand Name: {brand or "the business"}
- Target word count: {safe_words} words (min: {min_words}, max: {max_words})
- Primary keyword: {primary_keyword or "infer from title"}
- Audience: {audience or "qualified readers searching this topic"}
- Tone: {tone or "clear, authoritative, helpful"}
- Key features: {json.dumps(key_features or [])}
- Selling points: {json.dumps(selling_points or [])}
- Internal links to naturally mention: {json.dumps(internal_links or [])}

Return strict JSON only with this exact shape:
{{
  "title": "optimized H1 title",
  "metaTitle": "SEO title, 50-60 characters",
  "metaDescription": "SEO meta description, 140-160 characters",
  "slug": "short-kebab-case-slug",
  "excerpt": "short blog excerpt",
  "keywords": ["primary", "secondary"],
  "sections": [
    {{"id": "intro", "label": "Intro", "heading": "Introduction heading", "content": "section text"}},
    {{"id": "section-1", "label": "Section label", "heading": "H2 heading", "content": "section text"}}
  ]
}}

CRITICAL CONTENT RULES:
1. BRAND MENTIONS: You MUST naturally weave the brand name "{brand}" into the article AT LEAST 15 TIMES (and up to 30 times maximum across headings, intro, body paragraphs, case examples, recommendations, and closing call to action).
2. LENGTH: The total article body MUST be between {min_words} and {max_words} words (aiming for ~{safe_words} words). Write thorough, rich, detailed paragraphs instead of brief summaries.
3. INTEGRATION: Incorporate real business context, service offerings, customer benefits, and local/industry authority.
4. STRUCTURE: Use at least 5-6 H2 sections, bullet points, and an FAQ section at the end.
5. FORMAT: Do not include markdown code block fences. Return raw JSON only.
"""
```

### 6.2 Weekly SEO Content Plan Generator Prompt
* **File:** `backend/agents/ai_agent.py` (Lines 897–935)
* **Function:** `generate_weekly_blog_ideas()`
* **Variable:** `prompt`
* **Target Model:** Configurable (`claude` / `chatgpt` / `perplexity`)

```python
prompt = f"""
You are an SEO strategist for a business client.
Suggest the most useful weekly blog plan for this business.

Business:
- Name: {business_name}
- Category: {category or "not specified"}
- Location: {location or "not specified"}
- Services: {json.dumps(services or [])}
- Business voice: {business_voice or "not supplied yet"}
- Existing or preferred keywords: {json.dumps(existing_keywords or [])}

Return strict JSON only:
{{
  "voiceSuggestion": "short suggested brand voice",
  "keywords": ["keyword 1", "keyword 2", "keyword 3", "keyword 4", "keyword 5", "keyword 6"],
  "ideas": [
    {{
      "title": "SEO blog title",
      "primaryKeyword": "main keyword",
      "audience": "who this is for",
      "angle": "why this topic should help rankings"
    }},
    {{
      "title": "SEO blog title",
      "primaryKeyword": "main keyword",
      "audience": "who this is for",
      "angle": "why this topic should help rankings"
    }}
  ]
}}

Rules:
- Return exactly 2 ideas.
- Focus on likely ranking opportunities for the business niche and location.
- Avoid generic topics; include category, service, location, or buyer intent.
- Keywords must be natural search phrases.
- JSON only.
"""
```

### 6.3 Content Page Generator Prompt
* **File:** `backend/agents/ai_agent.py` (Lines 1014–1045)
* **Function:** `generate_content_page()`
* **Variable:** `prompt`
* **Target Model:** Configurable (`chatgpt` / `claude` / `perplexity`)

```python
prompt = f"""
You are a senior SEO content strategist for Wonderscore.
Generate a useful, factual, extractability-optimized website page for this business.

Business and scan context:
{json.dumps(safe_context, ensure_ascii=False)[:18000]}

Return strict JSON only with this exact shape:
{{
  "pageTitle": "clear H1 page title",
  "metaTitle": "SEO title, 50-65 characters",
  "metaDescription": "SEO meta description, 140-165 characters",
  "sections": [
    {{"heading": "H2 heading", "body": "plain-English section copy"}}
  ],
  "faqs": [
    {{"question": "customer search question", "answer": "accurate answer based only on known facts"}}
  ],
  "factsUsed": ["fact from profile or scan"],
  "warnings": ["fact-check warning or missing information note"]
}}

Rules:
- Write for non-technical customers and AI answer engines.
- Use the exact business location and category when available.
- Do not invent awards, prices, menus, facilities, opening hours, reviews, or claims.
- If a fact is missing, mention it in warnings instead of inventing it.
- Make the page specific enough to publish after human fact-checking.
- Include 4 to 6 sections and 3 to 5 FAQs.
- Keep section bodies concise, useful, and not salesy.
- Do not include markdown fences. JSON only.
"""
```

### 6.4 Blog Section Rewriter Prompt
* **File:** `backend/agents/ai_agent.py` (Lines 1119–1138)
* **Function:** `rewrite_blog_section()`
* **Variable:** `prompt`
* **Target Model:** Configurable (`chatgpt` / `claude` / `perplexity`)

```python
prompt = f"""
Rewrite one section of an SEO blog while preserving the article strategy.

Article title: {title}
Full blog context: {json.dumps(context)[:10000]}
Section to rewrite: {json.dumps(section)}
Extra instruction: {instruction or "Improve clarity, SEO intent coverage, and conversion value."}
Target section length: about {safe_target} words.

Return strict JSON only:
{{
  "heading": "section heading",
  "content": "rewritten section body"
}}

Rules:
- Keep the rewritten section consistent with surrounding sections.
- Do not duplicate the full blog.
- Do not include markdown fences. JSON only.
"""
```

### 6.5 Blog Quality Analysis Prompt
* **File:** `backend/agents/ai_agent.py` (Lines 749–772)
* **Function:** `get_blog_analysis_perplexity()`
* **Variable:** `prompt`
* **Target Model:** Perplexity AI

```python
prompt = f"""
You are an SEO blog analyst. Use the metrics and excerpt below to generate concise feedback.

Metrics (authoritative): {json.dumps(metrics)}

Blog excerpt:
{compact_text}

Return strict JSON only with this exact shape:
{{
  "overview": "one short paragraph",
  "summary": "one sentence summary",
  "strengths": ["short bullet"],
  "weakSpots": ["short bullet"],
  "improvements": ["short bullet"],
  "suggestions": ["short bullet"]
}}

Rules:
- Keep strings short and readable.
- Do not invent facts outside the excerpt.
- Do not include markdown or quotes around list items.
- JSON only.
"""
```

---

<a id="7-website-extraction--vision"></a>
## 7. Website Extraction & Vision

### 7.1 Phase 1 Perplexity Contact Field Extractor Prompt
* **File:** `backend/agents/ai_agent.py` (Lines 286–320)
* **Function:** `get_phase1_perplexity_extractor()`
* **Variable:** `prompt`
* **Target Model:** Perplexity AI

```python
prompt = f"""
You are a business data extractor.
Use URL + provided text excerpt + existing extracted fields to find missing business contact details.

URL: {url}
Business name: {business_name}
Current extracted data: {json.dumps(seed)}

Visible page text excerpt:
{compact_text}

Return strict JSON only:
{{
  "emails": ["contact@example.com"],
  "phones": ["+44 20 7946 0958"],
  "addresses": ["221B Baker Street, London NW1 6XE"],
  "openingHours": ["Mon-Fri: 09:00-18:00"],
  "socialLinks": {{"instagram": "https://instagram.com/example"}},
  "hasBookingPath": false,
  "confidence": {{
    "emails": 0,
    "phones": 0,
    "addresses": 0,
    "openingHours": 0,
    "socialLinks": 0,
    "bookingPath": 0
  }}
}}

Rules:
- Do not invent facts.
- Prefer values visible in the text excerpt or clearly tied to the URL/business.
- confidence values must be integers 0..100.
- JSON only.
"""
```

### 7.2 Website Quality & Technical Audit Suggestions Prompt
* **File:** `backend/agents/ai_agent.py` (Lines 365–403)
* **Function:** `get_phase1_deep_analyzer()`
* **Variable:** `prompt`
* **Target Model:** OpenAI ChatGPT

```python
prompt = f"""
You are a website quality analyst.
Based on URL, business name, current extracted fields and page text, give short practical improvement suggestions.

URL: {url}
Business name: {business_name}
Current extracted data: {json.dumps(seed)}
Visible page text excerpt:
{compact_text}

Return strict JSON only:
{{
  "suggestions": {{
    "Business name": "...",
    "Description": "...",
    "Logo": "...",
    "Language": "...",
    "Phone": "...",
    "Email": "...",
    "Address": "...",
    "Hours visible": "...",
    "Hours in schema": "...",
    "Social links": "...",
    "Booking path": "...",
    "Schema present": "...",
    "Correct type": "...",
    "Key fields": "...",
    "HTTPS": "...",
    "Mobile": "...",
    "Canonical": "...",
    "Sitemap": "...",
    "Robots": "..."
  }}
}}

Rules:
- Keep each suggestion concise (max 150 chars).
- JSON only.
"""
```

### 7.3 Website Screenshot Vision OCR Extractor Prompt
* **File:** `backend/agents/ai_agent.py` (Lines 689–705)
* **Function:** `get_vision_extraction()`
* **Variable:** `prompt`
* **Target Model:** OpenAI Vision

```python
prompt = """
You are an AI data extraction agent. Analyze this screenshot of a business website.
Extract the following information if perfectly visible and identifiable:
1. Phone numbers
2. Email addresses
3. Physical addresses
4. Opening / Operating hours

You MUST respond in valid JSON format matching exactly this structure:
{
    "phones": [],
    "emails": [],
    "addresses": [],
    "hours": []
}
If a category is completely empty, or not explicitly found in the image, return an empty array for it. Do NOT hallucinate data. Make the hours array easy to read, e.g. ["Mon-Fri: 9am-5pm", "Sat: 10am-4pm"].
"""
```

### 7.4 Phase 1 Scraping Data Enrichment Prompt
* **File:** `backend/agents/ai_agent.py` (Lines 1176–1238)
* **Function:** `get_phase1_enrichment()`
* **Variable:** `prompt`
* **Target Model:** OpenAI ChatGPT

```python
prompt = f"""
You are a web data extraction and optimization assistant.
Use only the provided website text excerpt and known extracted fields.

URL: {url}
Business name: {business_name}
Current extracted data: {json.dumps(seed)}

Visible page text excerpt:
{compact_text}

Tasks:
1) Find any missed contact and operating details from the text.
2) Provide short, actionable recommendations for each score label below.

Output strict JSON only with this exact shape:
{{
  "emails": ["contact@example.com"],
  "phones": ["+44 20 7946 0958"],
  "addresses": ["221B Baker Street, London NW1 6XE"],
  "openingHours": ["Mon-Fri: 9:00-18:00"],
  "socialLinks": {{"instagram": "https://instagram.com/example"}},
  "logoUrls": ["https://example.com/logo.png"],
  "hasBookingPath": false,
        "confidence": {{
            "emails": 0,
            "phones": 0,
            "addresses": 0,
            "openingHours": 0,
            "socialLinks": 0,
            "logoUrls": 0,
            "bookingPath": 0
        }},
  "suggestions": {{
    "Business name": "...",
    "Description": "...",
    "Logo": "...",
    "Language": "...",
    "Phone": "...",
    "Email": "...",
    "Address": "...",
    "Hours visible": "...",
    "Hours in schema": "...",
    "Social links": "...",
    "Booking path": "...",
    "Schema present": "...",
    "Correct type": "...",
    "Key fields": "...",
    "HTTPS": "...",
    "Mobile": "...",
    "Canonical": "...",
    "Sitemap": "...",
    "Robots": "..."
  }}
}}

Rules:
- Do not invent facts that are not implied by the provided text.
- Keep suggestion strings concise (max 150 chars each).
- confidence values must be integers 0..100.
- Return empty arrays/objects when unknown.
- JSON only, no markdown.
"""
```

### 7.5 Grounded Contact Retrieval Fallback Prompt
* **File:** `backend/agents/ai_agent.py` (Lines 1291–1316)
* **Function:** `get_phase1_contact_fallback()`
* **Variable:** `prompt`
* **Target Model:** Perplexity AI (Grounded Search)

```python
prompt = f"""
Extract business contact details for this website using grounded web retrieval.

Website: {url}
Business: {business_name}

Return strict JSON only:
{{
  "emails": ["contact@example.com"],
  "phones": ["+44 20 7946 0958"],
  "addresses": ["221B Baker Street, London NW1 6XE"],
  "openingHours": ["Mon-Fri: 09:00-18:00"],
  "confidence": {{
    "emails": 0,
    "phones": 0,
    "addresses": 0,
    "openingHours": 0
  }}
}}

Rules:
- Prefer details found on the target domain itself.
- Do not fabricate values.
- Confidence values are integers 0..100.
- JSON only.
"""
```

---

*Generated for Wonder AI Backend Codebase*
