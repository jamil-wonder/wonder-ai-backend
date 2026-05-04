from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class ScoreBreakdown(BaseModel):
    total: int
    businessName: Optional[int] = None
    description: Optional[int] = None
    logo: Optional[int] = None
    language: Optional[int] = None
    phone: Optional[int] = None
    email: Optional[int] = None
    address: Optional[int] = None
    hoursVisible: Optional[int] = None
    hoursStructured: Optional[int] = None
    socialLinks: Optional[int] = None
    booking: Optional[int] = None
    present: Optional[int] = None
    correctType: Optional[int] = None
    keyFields: Optional[int] = None
    ssl: Optional[int] = None
    mobile: Optional[int] = None
    canonical: Optional[int] = None
    sitemap: Optional[int] = None
    robots: Optional[int] = None

class Scores(BaseModel):
    total: int
    grade: str
    coreIdentity: ScoreBreakdown
    contact: ScoreBreakdown
    operating: ScoreBreakdown
    trust: ScoreBreakdown
    schema: ScoreBreakdown  # 'schema' is a reserved word in some contexts, using alias if needed, but pydantic v2 handles it fine usually, or we just leave it
    technical: ScoreBreakdown

    class Config:
        populate_by_name = True

class ScrapeResult(BaseModel):
    url: str
    title: str
    businessName: str
    description: str
    emails: List[str]
    phones: List[str]
    addresses: List[str]
    socialLinks: Dict[str, str]
    openingHours: List[str]
    logoUrl: Optional[str] = None
    language: str
    canonicalUrl: Optional[str] = None
    sitemapFound: bool
    robotsTxtFound: bool
    hasSSL: bool
    hasMobileMeta: bool
    hasAnalytics: bool
    pageSpeedHints: List[str]
    schemas: List[Dict[str, Any]]
    scores: Dict[str, Any]
    rawMeta: Dict[str, str]
    warnings: List[str]
    seoInfo: Optional[Dict[str, Any]] = None
    technologies: List[str] = Field(default_factory=list)
    aiSuggestions: Dict[str, str] = Field(default_factory=dict)
    aiDebug: Optional[Dict[str, Any]] = None

class ScrapeRequest(BaseModel):
    url: str

class AiModelInsight(BaseModel):
    modelName: str
    isKnown: bool
    summary: str
    sentiment: str
    platforms: List[str]
    evidence: List[str]

class AiInsightsRequest(BaseModel):
    url: str
    businessName: str

class AiInsightsResult(BaseModel):
    success: bool
    insights: List[AiModelInsight] = Field(default_factory=list)
    error: Optional[str] = None

class WishlistRequest(BaseModel):
    email: str

class TrackUrlRequest(BaseModel):
    url: str
    phase: str

# Authentication Models
class UserCreate(BaseModel):
    name: str
    email: str
    password: str

class UserResponse(BaseModel):
    id: str
    name: str
    email: str
    created_at: str
    role: str = "user"
    status: str = "active"

class UserRoleUpdateRequest(BaseModel):
    role: str

class SearchHistoryResponse(BaseModel):
    id: str
    user_id: Optional[str] = None
    user_email: Optional[str] = None
    url: str
    phase: str
    timestamp: Optional[str] = None

class Token(BaseModel):
    access_token: str
    token_type: str
    user: UserResponse

class LoginRequest(BaseModel):
    email: str
    password: str
