from pydantic import BaseModel
from typing import List, Optional


class BlogAnalyzeRequest(BaseModel):
    text: str
    attachment_count: int = 0
    model: str = "perplexity"


class BlogAnalysis(BaseModel):
    score: int
    grade: str
    overview: str
    summary: str
    wordCount: int
    paragraphCount: int
    headingCount: int
    listCount: int
    linkCount: int
    ctaCount: int
    readability: int
    structure: int
    seo: int
    engagement: int
    strengths: List[str]
    weakSpots: List[str]
    improvements: List[str]
    suggestions: List[str]


class BlogAnalysisResponse(BaseModel):
    success: bool
    result: Optional[BlogAnalysis] = None
    error: Optional[str] = None


class BlogSection(BaseModel):
    id: str
    label: str
    heading: str
    content: str


class BlogGenerateRequest(BaseModel):
    title: str
    target_words: int = 1200
    primary_keyword: Optional[str] = None
    audience: Optional[str] = None
    tone: Optional[str] = None
    key_features: List[str] = []
    selling_points: List[str] = []
    internal_links: List[str] = []
    selected_model: str = "chatgpt"


class BlogGenerateResult(BaseModel):
    title: str
    metaTitle: str
    metaDescription: str
    slug: str
    excerpt: str
    keywords: List[str]
    sections: List[BlogSection]
    wordCount: int
    modelUsed: str
    # F1: what to confirm before publishing, and where each claim came from
    checkBeforePublishing: List[str] = []
    claims: List[dict] = []
    # F6: Flesch reading ease {score, band, words, sentences}
    readability: Optional[dict] = None


class BlogGenerateResponse(BaseModel):
    success: bool
    result: Optional[BlogGenerateResult] = None
    usage: Optional[dict] = None
    error: Optional[str] = None


class BlogRewriteSectionRequest(BaseModel):
    title: str
    section: BlogSection
    full_blog_context: List[BlogSection] = []
    selected_model: str = "chatgpt"
    instruction: Optional[str] = None
    target_words: Optional[int] = None


class BlogRewriteSectionResponse(BaseModel):
    success: bool
    section: Optional[BlogSection] = None
    modelUsed: Optional[str] = None
    error: Optional[str] = None


class BlogUsageResponse(BaseModel):
    limit: int
    used: int
    remaining: int
    periodStart: str
    periodEnd: str


class BlogWeeklySetupRequest(BaseModel):
    business_id: str
    voice: Optional[str] = None
    keywords: List[str] = []
    # e.g. "en-GB"; empty = detect from the website / address
    language: Optional[str] = None


class BlogWeeklyEnsureRequest(BaseModel):
    business_id: str
    voice: Optional[str] = None
    keywords: List[str] = []
    force: bool = False


class BlogDraftEditSection(BaseModel):
    id: Optional[str] = None
    heading: Optional[str] = None
    content: str


class BlogDraftUpdateRequest(BaseModel):
    """F8: the owner's edits to a draft, and whether they have published it."""
    business_id: str
    draft_id: str
    sections: Optional[List[BlogDraftEditSection]] = None
    published: Optional[bool] = None
