from pydantic import BaseModel
from typing import List, Optional

class Phase5QuestionsRequest(BaseModel):
    url: str

class Phase5QuestionsResponse(BaseModel):
    questions: List[str]

class QuestionItem(BaseModel):
    id: str
    text: str

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


class Phase5StartJobRequest(BaseModel):
    url: str
    questions: List[QuestionItem]
    seed_results: Optional[dict] = None
    model: Optional[str] = "perplexity"


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
