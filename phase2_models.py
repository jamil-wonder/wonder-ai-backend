from pydantic import BaseModel, HttpUrl
from typing import List, Optional, Dict

# --- Phase 2: Conflict Engine Models ---

class ConflictSource(BaseModel):
    name: str # e.g. "Yelp", "Facebook"
    url: str

class MultiScanRequest(BaseModel):
    primary_url: str
    sources: List[ConflictSource]

class ExtractedEntityData(BaseModel):
    name: Optional[str] = None
    phone: Optional[str] = None
    address: Optional[str] = None
    hours_raw: Optional[str] = None
    has_schema: bool = False
    has_logo: bool = False

class ConflictIssue(BaseModel):
    field: str # e.g. "phone", "name", "hours"
    severity: str # "Critical", "Warning", "Info"
    primary_value: str
    source_value: str
    description: str

class SourceResult(BaseModel):
    source_name: str
    url: str
    status: str # "In-Sync", "Warning", "Critical", "Error"
    extracted_data: ExtractedEntityData
    issues: List[ConflictIssue]
    used_advanced_bypass: bool = False

class MultiScanResult(BaseModel):
    primary_url: str
    primary_data: ExtractedEntityData
    sources: List[SourceResult]
    total_issues: int
    critical_issues: int
    warning_issues: int
