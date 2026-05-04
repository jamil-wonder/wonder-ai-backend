from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from models import ScrapeResult

class CompareRequest(BaseModel):
    primary_url: str
    competitor_urls: List[str]

class FeatureDiff(BaseModel):
    feature_name: str
    primary_has_it: bool
    competitor_has_it: bool

class CompareResult(BaseModel):
    primary_data: ScrapeResult
    competitors_data: List[ScrapeResult]
