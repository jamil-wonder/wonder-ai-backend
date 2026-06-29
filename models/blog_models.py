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
