import time
import logging
from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse

from app.models.schemas import (
    ChatRequest, 
    ChatResponse, 
    LeadCreate, 
    LeadResponse, 
    ContentGapOut, 
    IngestStatus
)
from app.services.rag_engine import generate_grounded_answer, stream_grounded_answer
from app.services.lead_service import process_new_lead
from app.services.crawler import crawl_all_target_urls
from app.services.vector_store import vector_store
from app.db.database import get_content_gaps, get_leads

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")

@router.post("/chat", response_model=ChatResponse, summary="Submit a question to the grounded bot")
async def chat_endpoint(request: ChatRequest):
    """Answers patient questions strictly grounded in the 7 PlacidWay pages."""
    try:
        response = await generate_grounded_answer(
            query=request.message,
            session_id=request.session_id or "default_session"
        )
        return response
    except Exception as e:
        logger.error(f"Error processing chat: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error processing response.")

@router.post("/chat/stream", summary="Stream real-time tokens using Server-Sent Events (SSE)")
async def chat_stream_endpoint(request: ChatRequest):
    """Streams tokens as they are produced for sub-second UI responsiveness."""
    return StreamingResponse(
        stream_grounded_answer(request.message, request.session_id or "default_session"),
        media_type="text/event-stream"
    )

@router.post("/leads", response_model=LeadResponse, summary="Submit contact information for a free quote")
async def create_lead(lead: LeadCreate):
    """Captures interested patient details and assigns them to the session."""
    return process_new_lead(lead)

@router.get("/leads", summary="List captured leads (Internal/Admin)")
async def list_leads(limit: int = 50):
    return get_leads(limit)

@router.post("/ingest/refresh", response_model=IngestStatus, summary="Re-crawl the 7 URLs and refresh the knowledge base")
async def refresh_knowledge_base():
    """
    Crawls the 7 PlacidWay URLs respecting robots.txt and 1 req/sec,
    re-indexes semantic vectors, and saves fresh cache.
    """
    start_time = time.time()
    parsed_pages, all_chunks, errors = await crawl_all_target_urls()
    
    if all_chunks:
        vector_store.build_index(all_chunks)
        
    elapsed = round(time.time() - start_time, 2)
    
    return IngestStatus(
        success=len(all_chunks) > 0,
        total_pages_crawled=len(parsed_pages),
        total_chunks_indexed=len(all_chunks),
        elapsed_seconds=elapsed,
        pages=[{"title": p["title"], "url": p["url"], "type": p["page_type"]} for p in parsed_pages],
        errors=errors
    )

@router.get("/admin/gaps", summary="List logged content gaps / unanswered questions")
async def list_content_gaps(limit: int = 50):
    """Exposes questions the bot could not answer from current content for editorial review."""
    return get_content_gaps(limit)

@router.get("/status", summary="Knowledge base status and indexed counts")
async def get_status():
    return {
        "status": "online",
        "total_chunks_indexed": len(vector_store.chunks),
        "target_urls_count": 7,
        "is_indexed": len(vector_store.chunks) > 0
    }
