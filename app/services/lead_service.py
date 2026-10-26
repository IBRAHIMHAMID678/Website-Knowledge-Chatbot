import logging
from typing import Dict, Any, List
from app.db.database import save_lead, get_leads, get_content_gaps
from app.models.schemas import LeadCreate, LeadResponse

logger = logging.getLogger(__name__)

def process_new_lead(lead: LeadCreate) -> LeadResponse:
    try:
        lead_id = save_lead(
            session_id=lead.session_id or "default_session",
            full_name=lead.full_name,
            email=lead.email,
            phone=lead.phone or "",
            treatment_interest=lead.treatment_interest or "",
            notes=lead.notes or ""
        )
        logger.info(f"Captured new lead #{lead_id} for {lead.full_name} ({lead.email})")
        return LeadResponse(
            success=True,
            lead_id=lead_id,
            message="Thank you! Your quote request has been received. A PlacidWay medical coordinator will reach out shortly."
        )
    except Exception as e:
        logger.error(f"Failed to save lead: {e}")
        return LeadResponse(
            success=False,
            lead_id=-1,
            message="We could not submit your request at this moment. Please try again or email us directly."
        )
