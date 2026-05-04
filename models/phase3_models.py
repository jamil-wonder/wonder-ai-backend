from pydantic import BaseModel, Field
from typing import List, Optional, Dict

class ContentAnalysisRequest(BaseModel):
    url: str

class ContentAnalysisResponse(BaseModel):
    success: bool
    seoScore: int
    scoreBreakdown: Dict[str, int] = Field(default_factory=dict)
    readability: str
    sentiment: str
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)
    actionableAdvice: List[str] = Field(default_factory=list)
    targetAudience: str
    wordCount: int
    error: Optional[str] = None
