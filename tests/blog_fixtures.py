"""A well-formed article (F4: ~1,000 words, short paragraphs, H2 sections, 3 FAQs)
for tests that are about something else and just need a draft that passes."""

_SENTENCES = [
    "Good planning makes any evening easier and more enjoyable for everyone at the table.",
    "Think about the occasion, the size of the group and how much time you want to spend.",
    "Ask about dietary needs early so the kitchen has time to prepare something suitable.",
    "A quiet corner or a window seat can change the mood of a celebration completely.",
    "Check the opening times before you set off, since they can change through the week.",
    "Arrive a little early so that nobody feels rushed when the first course arrives.",
    "Small touches, such as a short note about the occasion, are usually appreciated.",
    "Keep the conversation relaxed and let the meal set its own pace through the evening.",
]


def _paragraph(n_sentences: int, offset: int) -> str:
    return " ".join(_SENTENCES[(offset + i) % len(_SENTENCES)] for i in range(n_sentences))


def valid_sections(first_section_text: str = "", internal_link: str | None = None) -> list[dict]:
    """Four prose sections (3 paragraphs of ~60 words each) + an FAQ with 3 questions."""
    sections = []
    for i, name in enumerate(["Introduction", "Choosing the right place", "Making the most of the evening", "After the meal"]):
        paras = [_paragraph(5, i + j) for j in range(3)]
        if i == 0 and first_section_text:
            paras.insert(0, first_section_text)
        if i == 1 and internal_link:
            paras.append(f"You can read more on [our menu page]({internal_link}) before you decide.")
        sections.append({"id": f"s{i}", "label": name, "heading": name, "content": "\n\n".join(paras)})
    faq = "\n\n".join(f"### Question number {n}?\n\n{_paragraph(2, n)}" for n in range(1, 4))
    sections.append({"id": "faq", "label": "FAQ", "heading": "Frequently asked questions", "content": faq})
    return sections
