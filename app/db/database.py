import sqlite3
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
import httpx

from app.config import settings

logger = logging.getLogger(__name__)

def is_supabase_enabled() -> bool:
    """Checks whether Supabase credentials have been configured."""
    return bool(settings.SUPABASE_URL and settings.SUPABASE_KEY)

def _supabase_headers() -> Dict[str, str]:
    """Generates standard Supabase PostgREST authentication and format headers."""
    return {
        "apikey": settings.SUPABASE_KEY,
        "Authorization": f"Bearer {settings.SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation"
    }

def get_connection() -> sqlite3.Connection:
    """Local SQLite connection for local development and offline testing."""
    try:
        settings.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    except (OSError, PermissionError):
        pass
    conn = sqlite3.connect(str(settings.DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initializes local SQLite schema and validates Supabase connectivity if enabled."""
    if is_supabase_enabled():
        logger.info(f"Supabase PostgreSQL database configured at: {settings.SUPABASE_URL}")
        try:
            with httpx.Client(timeout=5.0) as client:
                res = client.get(
                    f"{settings.SUPABASE_URL.rstrip('/')}/rest/v1/",
                    headers={"apikey": settings.SUPABASE_KEY}
                )
                if res.status_code in (200, 404):
                    logger.info("Supabase connectivity verified.")
        except Exception as e:
            logger.warning(f"Supabase connection probe warning: {e}")

    # Initialize local SQLite tables as resilient fallback
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # 1. Leads Table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS leads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            full_name TEXT NOT NULL,
            email TEXT NOT NULL,
            phone TEXT,
            treatment_interest TEXT,
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # 2. Content Gaps Table (Unanswered questions / Low confidence)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS content_gaps (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            query TEXT NOT NULL,
            rewritten_query TEXT,
            max_similarity_score REAL,
            detected_language TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # 3. Chat Messages History Table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            citations TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        conn.commit()
        conn.close()
    except (OSError, PermissionError, sqlite3.OperationalError) as e:
        logger.warning(f"Local SQLite init skipped on serverless/read-only environment: {e}")

# ==========================================
# 1. Lead Operations
# ==========================================
def save_lead(session_id: str, full_name: str, email: str, phone: str = "", treatment_interest: str = "", notes: str = "") -> int:
    """Saves lead to Supabase PostgreSQL or falls back to SQLite."""
    if is_supabase_enabled():
        try:
            url = f"{settings.SUPABASE_URL.rstrip('/')}/rest/v1/leads"
            payload = {
                "session_id": session_id,
                "full_name": full_name,
                "email": email,
                "phone": phone,
                "treatment_interest": treatment_interest,
                "notes": notes
            }
            with httpx.Client(timeout=8.0) as client:
                res = client.post(url, json=payload, headers=_supabase_headers())
                if res.status_code in (200, 201):
                    data = res.json()
                    if isinstance(data, list) and data and "id" in data[0]:
                        return int(data[0]["id"])
                    return 1
                logger.warning(f"Supabase save_lead HTTP {res.status_code}: {res.text}")
        except Exception as e:
            logger.error(f"Failed to save lead to Supabase, falling back to SQLite: {e}")

    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO leads (session_id, full_name, email, phone, treatment_interest, notes)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (session_id, full_name, email, phone, treatment_interest, notes))
        lead_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return lead_id or 1
    except Exception as e:
        logger.error(f"SQLite save_lead fallback failed: {e}")
        return 1

def get_leads(limit: int = 50) -> List[Dict[str, Any]]:
    """Retrieves captured leads from Supabase or SQLite."""
    if is_supabase_enabled():
        try:
            url = f"{settings.SUPABASE_URL.rstrip('/')}/rest/v1/leads?select=*&order=created_at.desc&limit={limit}"
            with httpx.Client(timeout=8.0) as client:
                res = client.get(url, headers=_supabase_headers())
                if res.status_code == 200:
                    return res.json()
        except Exception as e:
            logger.error(f"Failed to fetch leads from Supabase, falling back to SQLite: {e}")

    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM leads ORDER BY created_at DESC LIMIT ?", (limit,))
        rows = [dict(r) for r in cursor.fetchall()]
        conn.close()
        return rows
    except Exception as e:
        logger.error(f"SQLite get_leads fallback failed: {e}")
        return []

# ==========================================
# 2. Content Gap Logging
# ==========================================
def log_content_gap(session_id: str, query: str, rewritten_query: str = "", max_score: float = 0.0, language: str = "en"):
    """Logs ungrounded or low-confidence queries for editorial review."""
    if is_supabase_enabled():
        try:
            url = f"{settings.SUPABASE_URL.rstrip('/')}/rest/v1/content_gaps"
            payload = {
                "session_id": session_id,
                "query": query,
                "rewritten_query": rewritten_query,
                "max_similarity_score": max_score,
                "detected_language": language
            }
            with httpx.Client(timeout=8.0) as client:
                res = client.post(url, json=payload, headers=_supabase_headers())
                if res.status_code in (200, 201):
                    return
        except Exception as e:
            logger.error(f"Failed to log content gap to Supabase, falling back to SQLite: {e}")

    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO content_gaps (session_id, query, rewritten_query, max_similarity_score, detected_language)
            VALUES (?, ?, ?, ?, ?)
        """, (session_id, query, rewritten_query, max_score, language))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"SQLite log_content_gap fallback failed: {e}")

def get_content_gaps(limit: int = 50) -> List[Dict[str, Any]]:
    """Retrieves logged content gaps."""
    if is_supabase_enabled():
        try:
            url = f"{settings.SUPABASE_URL.rstrip('/')}/rest/v1/content_gaps?select=*&order=created_at.desc&limit={limit}"
            with httpx.Client(timeout=8.0) as client:
                res = client.get(url, headers=_supabase_headers())
                if res.status_code == 200:
                    return res.json()
        except Exception as e:
            logger.error(f"Failed to fetch content gaps from Supabase, falling back to SQLite: {e}")

    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM content_gaps ORDER BY created_at DESC LIMIT ?", (limit,))
        rows = [dict(r) for r in cursor.fetchall()]
        conn.close()
        return rows
    except Exception as e:
        logger.error(f"SQLite get_content_gaps fallback failed: {e}")
        return []

# ==========================================
# 3. Chat History Operations
# ==========================================
def save_message(session_id: str, role: str, content: str, citations: Optional[List[dict]] = None):
    """Saves chat message to Supabase or SQLite."""
    c_json = json.dumps(citations or [])
    if is_supabase_enabled():
        try:
            url = f"{settings.SUPABASE_URL.rstrip('/')}/rest/v1/chat_messages"
            payload = {
                "session_id": session_id,
                "role": role,
                "content": content,
                "citations": c_json
            }
            with httpx.Client(timeout=8.0) as client:
                res = client.post(url, json=payload, headers=_supabase_headers())
                if res.status_code in (200, 201):
                    return
        except Exception as e:
            logger.error(f"Failed to save message to Supabase, falling back to SQLite: {e}")

    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO chat_messages (session_id, role, content, citations)
            VALUES (?, ?, ?, ?)
        """, (session_id, role, content, c_json))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"SQLite save_message fallback failed: {e}")

def get_session_history(session_id: str, limit: int = 6) -> List[Dict[str, str]]:
    """Retrieves previous chat turns for conversation context."""
    if is_supabase_enabled():
        try:
            url = f"{settings.SUPABASE_URL.rstrip('/')}/rest/v1/chat_messages?session_id=eq.{session_id}&select=role,content,id&order=id.desc&limit={limit}"
            with httpx.Client(timeout=8.0) as client:
                res = client.get(url, headers=_supabase_headers())
                if res.status_code == 200:
                    messages = res.json()
                    messages.reverse()
                    return [{"role": m["role"], "content": m["content"]} for m in messages]
        except Exception as e:
            logger.error(f"Failed to fetch session history from Supabase, falling back to SQLite: {e}")

    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT role, content FROM chat_messages 
            WHERE session_id = ? 
            ORDER BY id DESC LIMIT ?
        """, (session_id, limit))
        rows = [dict(r) for r in cursor.fetchall()]
        conn.close()
        rows.reverse()
        return rows
    except Exception as e:
        logger.error(f"SQLite get_session_history fallback failed: {e}")
        return []
