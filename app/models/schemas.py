from typing import Optional, List
from pydantic import BaseModel, Field

class Citation(BaseModel):
    title: str = Field(..., description="Title of the referenced PlacidWay page")
    url: str = Field(..., description="Canonical source URL")
    snippet: Optional[str] = Field(None, description="Relevant contextual snippet")

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=3000, description="Visitor's question")
    session_id: Optional[str] = Field(default="default_session", max_length=128, description="Session identifier for multi-turn history")
    language: Optional[str] = Field(default="auto", max_length=10, description="Preferred response language")

class ChatResponse(BaseModel):
    answer: str = Field(..., description="Grounded response text")
    citations: List[Citation] = Field(default_factory=list, description="List of cited source URLs")
    is_fallback: bool = Field(default=False, description="Whether the bot triggered 'I don't know'")
    disclaimer_included: bool = Field(default=False, description="Whether a medical disclaimer was attached")
    lead_prompt_recommended: bool = Field(default=False, description="Whether the user should be prompted for a quote")
    session_id: str

class LeadCreate(BaseModel):
    full_name: str = Field(..., min_length=2, max_length=120, description="Patient's full name")
    email: str = Field(..., min_length=5, max_length=180, description="Patient's email address")
    phone: Optional[str] = Field(default="", max_length=50, description="Patient's phone number")
    session_id: Optional[str] = Field(default="default_session", max_length=128)
    treatment_interest: Optional[str] = Field(default="", max_length=200, description="Treatment or package of interest")
    notes: Optional[str] = Field(default="", max_length=1500)

class LeadResponse(BaseModel):
    success: bool
    lead_id: int
    message: str

class ContentGapOut(BaseModel):
    id: int
    session_id: str
    query: str
    rewritten_query: Optional[str]
    max_similarity_score: float
    created_at: str

class IngestStatus(BaseModel):
    success: bool
    total_pages_crawled: int
    total_chunks_indexed: int
    elapsed_seconds: float
    pages: List[dict]
    errors: List[str] = Field(default_factory=list)
