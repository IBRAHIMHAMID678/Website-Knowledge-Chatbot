import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from app.config import settings

def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(settings.DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
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

# Lead operations
def save_lead(session_id: str, full_name: str, email: str, phone: str = "", treatment_interest: str = "", notes: str = "") -> int:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO leads (session_id, full_name, email, phone, treatment_interest, notes)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (session_id, full_name, email, phone, treatment_interest, notes))
    lead_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return lead_id

def get_leads(limit: int = 50) -> List[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM leads ORDER BY created_at DESC LIMIT ?", (limit,))
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows

# Content Gap logging
def log_content_gap(session_id: str, query: str, rewritten_query: str = "", max_score: float = 0.0, language: str = "en"):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO content_gaps (session_id, query, rewritten_query, max_similarity_score, detected_language)
        VALUES (?, ?, ?, ?, ?)
    """, (session_id, query, rewritten_query, max_score, language))
    conn.commit()
    conn.close()

def get_content_gaps(limit: int = 50) -> List[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM content_gaps ORDER BY created_at DESC LIMIT ?", (limit,))
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows

# Chat history operations
def save_message(session_id: str, role: str, content: str, citations: Optional[List[dict]] = None):
    conn = get_connection()
    cursor = conn.cursor()
    c_json = json.dumps(citations or [])
    cursor.execute("""
        INSERT INTO chat_messages (session_id, role, content, citations)
        VALUES (?, ?, ?, ?)
    """, (session_id, role, content, c_json))
    conn.commit()
    conn.close()

def get_session_history(session_id: str, limit: int = 6) -> List[Dict[str, str]]:
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
