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
