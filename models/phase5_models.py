from pydantic import BaseModel, Field
from typing import List, Optional

class Phase5QuestionsRequest(BaseModel):
    url: str
    businessName: Optional[str] = None
    category: Optional[str] = None
    location: Optional[str] = None
    description: Optional[str] = None
    services: List[str] = Field(default_factory=list)
    business_id: Optional[str] = None
    questionGeneration: Optional[dict[str, int]] = None
    avoidQuestions: List[str] = Field(default_factory=list)

class Phase5RewriteRequest(BaseModel):
    """J8: one question to make sound more human."""
    question: str
    businessName: Optional[str] = None
    category: Optional[str] = None
    location: Optional[str] = None
    branded: bool = False

class Phase5QuestionsResponse(BaseModel):
    questions: List[str]
    questionGroups: Optional[dict[str, List[str]]] = None

class QuestionItem(BaseModel):
    id: str
    text: str
    # branded | non-branded | local-seo | broad-seo - used for the A1 question weights
    category: Optional[str] = None

class Phase5AnalyzeRequest(BaseModel):
    url: str
    questions: List[QuestionItem]

class Phase5AnalyzeResult(BaseModel):
    status: str  # "Mentioned" | "Not Mentioned"
    position: Optional[int] = None
    sources: List[str]

class Phase5AnalyzeResponse(BaseModel):
    results: dict[str, Phase5AnalyzeResult]  # Keyed by question id

class Phase5AnalyzeSingleRequest(BaseModel):
    url: str
    question: QuestionItem

class Phase5AnalyzeSingleResponse(BaseModel):
    id: str
    status: str
    position: Optional[int] = None
    sources: List[str]
    source_urls: List[str] = []
    references: List[str] = []
    competitors: List[str] = []
    competitor_scores: List[dict] = []
    reasoning: Optional[str] = None
    llm_response: Optional[str] = None
    target_site: Optional[dict] = None


class Phase5StartJobRequest(BaseModel):
    url: str
    questions: List[QuestionItem]
    seed_results: Optional[dict] = None
    model: Optional[str] = "perplexity"
    business_id: Optional[str] = None
    # Ask all four models again instead of re-using this week's stored answers
    fresh: Optional[bool] = False


class Phase5StartJobResponse(BaseModel):
    job_id: str
    status: str
    total: int


class Phase5JobStatusResponse(BaseModel):
    job_id: str
    status: str
    total: int
    processed: int
    current_question_id: Optional[str] = None
    results: dict[str, dict]
    deep_competitors: List[dict] = []
    brand_summary: Optional[str] = None
    error: Optional[str] = None
    # When the four models were actually asked (a re-run inside the week re-uses those answers)
    answers_collected_at: Optional[str] = None
    run_health: Optional[dict] = None
