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
    businessId: Optional[str] = None
    businessProfile: Optional[Dict[str, Any]] = None

class ScrapeRequest(BaseModel):
    url: str
    category: Optional[str] = None
    location: Optional[str] = None
    business_id: Optional[str] = None

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


class BusinessUpsertRequest(BaseModel):
    url: str
    business_id: Optional[str] = None
    category: Optional[str] = None
    location: Optional[str] = None
    businessName: Optional[str] = None
    logoUrl: Optional[str] = None
    businessDescription: Optional[str] = None
    aiDescription: Optional[str] = None
    services: Optional[List[str]] = None
    targetAudience: Optional[str] = None
    competitors: Optional[List[str]] = None
    trackedPages: Optional[List[str]] = None


class BusinessResponse(BaseModel):
    id: str
    user_id: str
    url: str
    normalized_domain: str
    businessName: Optional[str] = None
    category: Optional[str] = None
    location: Optional[str] = None
    logoUrl: Optional[str] = None
    businessDescription: Optional[str] = None
    aiDescription: Optional[str] = None
    services: List[str] = Field(default_factory=list)
    targetAudience: Optional[str] = None
    competitors: List[str] = Field(default_factory=list)
    trackedPages: List[str] = Field(default_factory=list)
    latest_phase1_score: Optional[int] = None
    latest_phase5_score: Optional[float] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
