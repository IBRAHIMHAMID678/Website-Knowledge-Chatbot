-- ====================================================================
-- PlacidWay Medical Tourism AI Knowledge Chatbot
-- Supabase PostgreSQL Database Schema
-- Run this script in the Supabase Dashboard -> SQL Editor -> New Query
-- ====================================================================

-- 1. Leads Table (Stores captured prospective patient details)
CREATE TABLE IF NOT EXISTS public.leads (
    id BIGSERIAL PRIMARY KEY,
    session_id TEXT NOT NULL,
    full_name TEXT NOT NULL,
    email TEXT NOT NULL,
    phone TEXT,
    treatment_interest TEXT,
    notes TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 2. Content Gaps Table (Logs questions bot could not answer from grounded corpus)
CREATE TABLE IF NOT EXISTS public.content_gaps (
    id BIGSERIAL PRIMARY KEY,
    session_id TEXT NOT NULL,
    query TEXT NOT NULL,
    rewritten_query TEXT,
    max_similarity_score REAL,
    detected_language TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 3. Chat Messages History Table (Persists conversational turns across serverless cold starts)
CREATE TABLE IF NOT EXISTS public.chat_messages (
    id BIGSERIAL PRIMARY KEY,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    citations TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for fast ordering and lookup
CREATE INDEX IF NOT EXISTS idx_leads_created_at ON public.leads (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_content_gaps_created_at ON public.content_gaps (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_chat_messages_session ON public.chat_messages (session_id, id DESC);

-- Enable Row Level Security (RLS) on all exposed tables
ALTER TABLE public.leads ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.content_gaps ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.chat_messages ENABLE ROW LEVEL SECURITY;

-- Explicit PostgREST Data API & Role Grants
GRANT USAGE ON SCHEMA public TO anon, authenticated, service_role;
GRANT ALL ON TABLE public.leads TO anon, authenticated, service_role;
GRANT ALL ON TABLE public.content_gaps TO anon, authenticated, service_role;
GRANT ALL ON TABLE public.chat_messages TO anon, authenticated, service_role;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO anon, authenticated, service_role;

-- Drop existing policies if re-running script to avoid conflicts
DROP POLICY IF EXISTS "Allow anon and service role insert leads" ON public.leads;
DROP POLICY IF EXISTS "Allow anon and service role select leads" ON public.leads;
DROP POLICY IF EXISTS "Allow anon and service role insert content_gaps" ON public.content_gaps;
DROP POLICY IF EXISTS "Allow anon and service role select content_gaps" ON public.content_gaps;
DROP POLICY IF EXISTS "Allow anon and service role insert chat_messages" ON public.chat_messages;
DROP POLICY IF EXISTS "Allow anon and service role select chat_messages" ON public.chat_messages;

-- RLS Policies (allows both anon and service_role keys to read and write)
CREATE POLICY "Allow anon and service role insert leads" ON public.leads
    FOR INSERT TO anon, authenticated, service_role WITH CHECK (true);

CREATE POLICY "Allow anon and service role select leads" ON public.leads
    FOR SELECT TO anon, authenticated, service_role USING (true);

CREATE POLICY "Allow anon and service role insert content_gaps" ON public.content_gaps
    FOR INSERT TO anon, authenticated, service_role WITH CHECK (true);

CREATE POLICY "Allow anon and service role select content_gaps" ON public.content_gaps
    FOR SELECT TO anon, authenticated, service_role USING (true);

CREATE POLICY "Allow anon and service role insert chat_messages" ON public.chat_messages
    FOR INSERT TO anon, authenticated, service_role WITH CHECK (true);

CREATE POLICY "Allow anon and service role select chat_messages" ON public.chat_messages
    FOR SELECT TO anon, authenticated, service_role USING (true);
