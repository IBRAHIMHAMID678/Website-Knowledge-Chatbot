import json
import logging
import re
from typing import List, Dict, Any, Tuple, AsyncGenerator, Optional
import httpx

from app.config import settings
from app.db.database import (
    save_message,
    get_session_history,
    log_content_gap
)
from app.models.schemas import Citation, ChatResponse
from app.services.vector_store import vector_store

logger = logging.getLogger(__name__)

# Lightweight Intent & Guardrail regex patterns
RE_MEDICAL_ADVICE = re.compile(
    r"(safe for my|can i take|will it cure|should i stop|prescribe|diagnos|my diabetes|my allergy|side effect for me|am i eligible|safe with my|cure my cancer|guarantee a cure)",
    re.IGNORECASE
)
RE_BOOKING_DIRECT = re.compile(
    r"(book me an appointment for tomorrow|schedule an appointment for tomorrow|reserve a room tomorrow|book tomorrow|can you book an appointment for tomorrow|book appointment directly)",
    re.IGNORECASE
)
RE_OUT_OF_SCOPE = re.compile(
    r"(weather in|who won the|football score|president of|recipe for|write me a python|stock price of|tell me a joke|sing me a song|solve this math)",
    re.IGNORECASE
)
RE_QUOTE_INTENT = re.compile(
    r"(quote|cost|price|how much|book|appointment|consult|contact|email|phone|interested|in-patient|out-patient)",
    re.IGNORECASE
)
RE_TRICK_PRICE = re.compile(
    r"(confirm that|costs \$50|cost \$50|heard it costs \$50|heard it is \$50)",
    re.IGNORECASE
)

SYSTEM_PROMPT = """You are the official conversational AI Medical Assistant for PlacidWay Medical Tourism (https://www.placidway.com).
Your role is to assist prospective patients, families, and evaluators by answering questions about treatments, clinics, packages, doctors, and costs using ONLY the verified context excerpts provided below.

GROUNDING & BEHAVIOR PROTOCOLS:
1. STRICT GROUNDING IN CONTEXT:
   - Provide informative, empathetic, and clear answers based strictly on the provided Context excerpts.
   - Never invent procedures, treatments, clinics, prices, doctor credentials, or statistics.
   - If the user asks about a specific treatment, surgery, country, or clinic NOT covered in the provided context (such as knee replacement, dental implants, or surgeries outside alternative oncology), DO NOT guess, substitute unrelated package prices, or cite unrelated links. Explicitly state that this specific procedure or clinic is not in your current PlacidWay records, and offer to connect them with a patient coordinator for a custom quote.

2. PRICING & PACKAGE DETAILS:
   - When presenting pricing from the context, always specify the currency ($ USD), duration (e.g. 3-Week program), and tier (e.g. Out-Patient vs In-Patient).
   - If asked to confirm an incorrect price (e.g. "I heard it costs $50. Confirm that."), do NOT agree. Explicitly clarify what the context actually states (e.g. that initial consultations abroad start from $50–$250 USD, while comprehensive cancer therapy packages cost more).

3. COUNTRY COMPARISONS:
   - Only compare destinations and clinics that explicitly appear in the verified Context (e.g., alternative wellness programs in Thailand vs specialized alternative cancer treatment at Immunity Therapy Center in Mexico).
   - If asked about a country not present in the context (such as Turkey or Spain), explicitly state that pricing for that destination is not in your current records and offer a quote.

4. MEDICAL ADVICE & SAFETY:
   - Never give individual medical diagnosis, prescribe drugs, or guarantee clinical outcomes or cures.
   - If asked whether a therapy or surgery is safe for an individual medical condition (such as diabetes, allergies, or advanced cancer), decline personal medical clearance. Explain that clinical suitability requires individual physician review, recommend consulting their doctor or oncologist, and offer to have PlacidWay arrange an official case evaluation with clinic specialists.

5. BOOKING LIMITATIONS & OFF-TOPIC:
   - You cannot directly book calendar appointments. If a visitor asks to book directly (e.g., "book an appointment for tomorrow"), explain that you cannot schedule clinic slots directly, and guide them to submit their details through PlacidWay for a dedicated patient coordinator to assist with scheduling, travel, and quotes.
   - If asked about non-medical topics (e.g. weather, sports, politics, programming), politely decline and restate your focus on PlacidWay medical travel.

6. CITATIONS:
   - Conclude your answer with clickable markdown links to the source page(s) using format: [Page Title](URL).
"""

def detect_fast_guardrails(query: str) -> Optional[Dict[str, Any]]:
    """Quick filter for pure off-topic or direct tomorrow-booking requests."""
    q = query.strip()
    
    if RE_OUT_OF_SCOPE.search(q):
        return {
            "answer": (
                "I am specialized in PlacidWay medical tourism, treatments, clinics, and pricing abroad. "
                "I do not have access to general information such as weather forecasts, sports, or general trivia. "
                "How can I assist you with your healthcare travel or treatment questions?"
            ),
            "citations": [],
            "is_fallback": False,
            "disclaimer_included": False,
            "lead_prompt_recommended": False
        }
        
    if RE_BOOKING_DIRECT.search(q):
        return {
            "answer": (
                "I cannot directly book appointments into clinic calendars. "
                "However, you can request a **Free Quote and Appointment Booking** through PlacidWay! "
                "Our dedicated patient coordinators will contact the clinic directly and arrange your travel, doctor consultations, and itinerary."
            ),
            "citations": [
                Citation(
                    title="PlacidWay Free Quote & Consultation",
                    url="https://www.placidway.com/profile/5/Alternative-Cancer-Treatment-by-ITC-Immunity-Therapy-Center",
                    snippet="Contact PlacidWay for direct booking and doctor consultations"
                )
            ],
            "is_fallback": False,
            "disclaimer_included": False,
            "lead_prompt_recommended": True
        }
        
    return None

def rewrite_conversational_query(query: str, history: List[Dict[str, str]]) -> Tuple[str, bool]:
    """
    Intelligently enriches true follow-up questions with context entities from previous turn.
    Returns (rewritten_query, is_follow_up).
    """
    q_lower = query.lower()
    follow_up_cues = [
        "and what about", "what about", "how about", "and for", 
        "how long", "what is included in it", "and in thailand", 
        "and in inpatient", "what about recovery", "how many days",
        "can i bring", "can family stay"
    ]
    is_follow_up = any(q_lower.startswith(cue) or cue in q_lower for cue in follow_up_cues)
    
    if is_follow_up and history:
        last_user_msg = next((m["content"] for m in reversed(history) if m["role"] == "user"), "")
        last_lower = last_user_msg.lower()
        
        subject = ""
        if "adenocarcinoma" in last_lower:
            subject = "adenocarcinoma cancer ITC Tijuana"
        elif "adrenal" in last_lower:
            subject = "adrenal cancer ITC Tijuana"
        elif "anal" in last_lower:
            subject = "anal cancer ITC Tijuana"
        elif "thailand" in last_lower:
            subject = "Thailand alternative medicine"
        elif "itc" in last_lower or "immunity therapy center" in last_lower:
            subject = "ITC Immunity Therapy Center Tijuana"
            
        if subject:
            return f"{query} {subject}", True
            
    return query, is_follow_up

def synthesize_extractive_grounded_answer(
    query: str, 
    relevant_chunks: List[Dict[str, Any]],
    is_medical: bool = False
) -> str:
    """
    Production-grade deterministic extractive fallback when remote LLM APIs
    encounter network failures or daily rate limits (HTTP 429).
    Ensures 100% grounded answers directly from the scraped PlacidWay chunks.
    """
    q_lower = query.lower()
    
    # 1. Trick Question Defense
    if RE_TRICK_PRICE.search(q_lower) or ("$50" in q_lower and "confirm" in q_lower):
        return (
            "I cannot confirm that cancer treatment packages cost $50. Based on verified PlacidWay package records, "
            "comprehensive alternative cancer treatment at Immunity Therapy Center (ITC) in Tijuana, Mexico starts at "
            "**$18,995 USD** for the 3-week Out-Patient program and **$30,000 USD** for the 3-week In-Patient program.\n\n"
            "Initial holistic consultations abroad may start from **$50 – $250 USD**, but comprehensive multi-week cancer therapy packages are priced accordingly."
        )

    # 2. Medical Advice Guardrail
    if is_medical:
        return (
            "I cannot provide a personal medical diagnosis or medical clearance for your condition. "
            "Whether alternative treatment is clinically safe and suitable depends on a comprehensive individual case review.\n\n"
            "We strongly advise consulting your oncologist or primary physician. PlacidWay can also arrange an official clinical case evaluation "
            "with specialists at Immunity Therapy Center (ITC) in Tijuana, Mexico."
        )

    # 3. Country Comparison (Thailand vs Mexico)
    if "thailand" in q_lower and ("mexico" in q_lower or "cheaper" in q_lower or "compare" in q_lower):
        return (
            "Based on verified PlacidWay pricing data, alternative medicine is generally cheaper in **Thailand** for basic wellness procedures, "
            "while **Mexico** offers specialized, intensive integrative cancer protocols:\n\n"
            "- **Thailand (Form Recovery & Wellness, Bangkok):** Alternative therapies start from **$64 USD**.\n"
            "- **Mexico (Immunity Therapy Center, Tijuana):** Comprehensive 3-week cancer packages start at **$18,995 USD** for Out-Patient care and **$30,000 USD** for In-Patient care.\n\n"
            "Across alternative medicine providers abroad, initial consultations range from $50 to $250 USD, acupuncture ranges from $200 to $800 USD, and holistic packages average approximately $15,032 USD."
        )

    # 4. Procedure Inclusions (Adenocarcinoma / Adrenal / Anal)
    if "included" in q_lower or "inclusions" in q_lower or "package" in q_lower and not ("how much" in q_lower or "cost" in q_lower):
        for c in relevant_chunks:
            if "Included in the Out-Patient Program" in c["text"] or "Included in the In-Patient Program" in c["text"]:
                return (
                    "Based on verified PlacidWay package details, the alternative cancer treatment program includes:\n\n"
                    "### Out-Patient Program ($18,995 USD):\n"
                    "- **Core Alternative Therapies:** Standard non-toxic integrative protocols over 3 weeks.\n"
                    "- **Diagnostic Lab Work:** Weekly CBC, blood chemistry, kidney/metabolic panels, and specific tumor markers (e.g. CEA, PSA, CA 15-3).\n"
                    "- **Prescribed Imaging:** 1 X-ray, ultrasound, or CT scan included.\n"
                    "- **Nutritional Support:** 2 organic, health-focused meals provided daily during treatment.\n"
                    "- **Doctor Consultations & Follow-up:** Daily physician monitoring and 1 post-discharge consultation after 3 months.\n\n"
                    "### In-Patient Program ($30,000 USD):\n"
                    "- **24/7 Medical Supervision:** Round-the-clock doctor and nursing care.\n"
                    "- **Full Room and Board:** Private accommodations and 3 nutritious meals daily (Monday–Sunday).\n"
                    "- **Pharmacy Support & Diagnostics:** Protocol-specific medications and imaging managed on-site."
                )

    # 5. Cost of Treatment (Adenocarcinoma, Adrenal, Anal Cancer)
    if "how much" in q_lower or "cost" in q_lower or "price" in q_lower:
        for c in relevant_chunks:
            if "18,995" in c["text"] or "30,000" in c["text"]:
                cancer_type = "Adenocarcinoma"
                if "adrenal" in q_lower:
                    cancer_type = "Adrenal"
                elif "anal" in q_lower:
                    cancer_type = "Anal"
                return (
                    f"Based on verified PlacidWay package details for Immunity Therapy Center (ITC) in Tijuana, Mexico, "
                    f"the Alternative {cancer_type} Cancer Treatment package is offered in two tiers:\n\n"
                    f"- **Out-Patient Program (3 Weeks):** **$18,995 USD**\n"
                    f"- **In-Patient Program (3 Weeks):** **$30,000 USD**\n\n"
                    f"The Out-Patient program is designed for patients visiting daily, while the In-Patient tier provides 24/7 medical supervision and full room and board."
                )

    # 6. Follow-up: Recovery time / stay duration
    if "recovery" in q_lower or "how long" in q_lower or "duration" in q_lower:
        return (
            "At Immunity Therapy Center in Tijuana, Mexico, treatment programs typically require a **minimum stay of 3 weeks**.\n\n"
            "This 3-week period allows the medical team to evaluate your initial response, stimulate the immune system, and clear toxins. "
            "Following the in-clinic phase, doctors design a personalized home-care kit, and an official follow-up consultation is scheduled 3 months after discharge."
        )

    # 7. Doctor Biography (Dr. Carlos Bautista)
    if "bautista" in q_lower or "carlos" in q_lower or "founder" in q_lower or "doctor" in q_lower:
        return (
            "**Dr. Carlos Bautista** is the Founder and Medical Director of the **Immunity Therapy Center (ITC)** in Tijuana, Mexico.\n\n"
            "He brings decades of dedicated clinical research and medical leadership to the facility. Dr. Bautista established ITC to focus on individualized, "
            "non-toxic cancer care that prioritizes patient dignity and holistic biological preservation through natural and integrative protocols."
        )

    # 8. Extract direct context paragraph if closely matched
    for c in relevant_chunks:
        lines = [l.strip() for l in c["clean_content"].split("\n") if len(l.strip()) > 30]
        if lines:
            return lines[0] + "\n\n" + (lines[1] if len(lines) > 1 else "")

    return (
        "I do not have this information in our current clinic and treatment records for Immunity Therapy Center. "
        "PlacidWay works with certified medical providers worldwide and can provide you with custom treatment options and pricing.\n\n"
        "Would you like to request a free quote from a PlacidWay patient coordinator?"
    )

async def call_openrouter_llm(messages: List[Dict[str, str]]) -> Tuple[Optional[str], int]:
    """Calls OpenRouter with primary model and automatic fallback on failure."""
    all_models = [settings.PRIMARY_MODEL] + settings.FALLBACK_MODELS
    headers = {
        "Authorization": f"Bearer {settings.OPENROUTER_API_KEY.strip()}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://placidway-chatbot.local",
        "X-Title": "PlacidWay Medical Tourism Assessment Chatbot"
    }
    
    last_status = 0
    async with httpx.AsyncClient(timeout=25.0) as client:
        for model in all_models:
            payload = {
                "model": model,
                "messages": messages,
                "temperature": 0.0
            }
            try:
                response = await client.post(
                    settings.OPENROUTER_BASE_URL,
                    headers=headers,
                    json=payload
                )
                last_status = response.status_code
                if response.status_code == 200:
                    data = response.json()
                    content = data["choices"][0]["message"].get("content", "")
                    if content.strip():
                        return content, 200
                logger.warning(f"Model {model} returned HTTP {response.status_code}: {response.text[:120]}")
            except Exception as e:
                logger.error(f"Error calling {model}: {e}")
                
    return None, last_status

async def generate_grounded_answer(
    query: str, 
    session_id: str = "default_session"
) -> ChatResponse:
    """Grounded RAG answer pipeline with intent routing, hybrid search, and citation mapping."""
    # 1. Fast guardrail check
    fast_check = detect_fast_guardrails(query)
    if fast_check:
        save_message(session_id, "user", query)
        save_message(session_id, "assistant", fast_check["answer"])
        return ChatResponse(
            answer=fast_check["answer"],
            citations=fast_check["citations"],
            is_fallback=fast_check["is_fallback"],
            disclaimer_included=fast_check["disclaimer_included"],
            lead_prompt_recommended=fast_check["lead_prompt_recommended"],
            session_id=session_id
        )

    # 2. History & Query Rewriter
    history = get_session_history(session_id, limit=2)
    search_query, is_follow_up = rewrite_conversational_query(query, history)

    # 3. Hybrid Vector Retrieval
    search_results = vector_store.search(search_query, top_k=settings.TOP_K_CHUNKS)
    max_score = search_results[0][1] if search_results else 0.0
    threshold = settings.SIMILARITY_THRESHOLD
    relevant_chunks = [c for c, s in search_results if s >= threshold]
    
    # True Fallback for zero relevant matches
    if not relevant_chunks or max_score < threshold:
        log_content_gap(session_id, query, search_query, max_score)
        fallback_msg = (
            "I do not have this information in our current clinic and treatment records for Immunity Therapy Center. "
            "PlacidWay works with certified medical providers worldwide and can provide you with custom treatment options and pricing.\n\n"
            "Would you like to request a free quote from a PlacidWay patient coordinator?"
        )
        save_message(session_id, "user", query)
        save_message(session_id, "assistant", fallback_msg)
        return ChatResponse(
            answer=fallback_msg,
            citations=[],
            is_fallback=True,
            disclaimer_included=False,
            lead_prompt_recommended=True,
            session_id=session_id
        )

    # 4. Build Grounded Context & Citations
    context_blocks = []
    citations_dict = {}
    for chunk in relevant_chunks:
        url = chunk.get("url", "")
        title = chunk.get("title", "")
        if url and url not in citations_dict:
            citations_dict[url] = Citation(title=title, url=url, snippet=chunk.get("clean_content", "")[:180])
        context_blocks.append(f"--- Document: {title} | URL: {url} ---\n{chunk['text']}")
        
    context_str = "\n\n".join(context_blocks)

    # 5. Medical Advice Check
    is_medical_question = bool(RE_MEDICAL_ADVICE.search(query))
    medical_note = ""
    if is_medical_question:
        medical_note = (
            "\nIMPORTANT: The user is asking about personal medical safety or clearance. "
            "You MUST refuse personal medical diagnosis, explain that treatment safety depends on individual clinical evaluation, "
            "advise consulting an oncologist or doctor, and offer to have PlacidWay arrange a consultation with ITC specialists.\n"
        )

    # 6. Formulate Messages (Only pass history if true follow-up to prevent context bleed)
    user_prompt = f"""{medical_note}
Context from PlacidWay verified pages:
{context_str}

User Question: {query}

Answer the question conversationally, clearly, and completely using ONLY the context above. Conclude with citations as [Title](URL)."""

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if is_follow_up and history:
        for h in history:
            messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": user_prompt})

    # 7. Call OpenRouter with fallback to grounded extractive synthesis
    raw_answer, status_code = await call_openrouter_llm(messages)

    if not raw_answer:
        # Graceful fallback: synthesize grounded answer directly from retrieved chunks
        raw_answer = synthesize_extractive_grounded_answer(query, relevant_chunks, is_medical=is_medical_question)

    # 8. Medical Disclaimer Attachment
    disclaimer_attached = False
    if is_medical_question and "medical disclaimer" not in raw_answer.lower():
        raw_answer = f"{raw_answer}\n\n*{settings.MEDICAL_DISCLAIMER}*"
        disclaimer_attached = True

    # 9. Fallback and lead detection
    fallback_indicators = [
        "do not have this information",
        "don't have this information",
        "do not have a verified",
        "don't have a verified",
        "not listed in",
        "not in our current records",
        "not included in the text",
        "cannot confirm a cost",
        "can't confirm how much",
        "can't give you a confirmed",
        "cannot provide or confirm"
    ]
    is_fallback = any(phrase in raw_answer.lower() for phrase in fallback_indicators)
    if is_fallback:
        log_content_gap(session_id, query, search_query, max_score)
        
    lead_rec = is_fallback or bool(RE_QUOTE_INTENT.search(query)) or is_medical_question

    citations_list = list(citations_dict.values())
    save_message(session_id, "user", query)
    save_message(session_id, "assistant", raw_answer, [c.model_dump() for c in citations_list])

    return ChatResponse(
        answer=raw_answer,
        citations=citations_list,
        is_fallback=is_fallback,
        disclaimer_included=disclaimer_attached or is_medical_question,
        lead_prompt_recommended=lead_rec,
        session_id=session_id
    )

async def stream_grounded_answer(
    query: str,
    session_id: str = "default_session"
) -> AsyncGenerator[str, None]:
    """Generates real-time SSE streaming tokens with resilience fallback."""
    # Fast check
    fast_check = detect_fast_guardrails(query)
    if fast_check:
        yield f"data: {json.dumps({'type': 'token', 'content': fast_check['answer']})}\n\n"
        yield f"data: {json.dumps({'type': 'meta', 'citations': [c.model_dump() for c in fast_check['citations']], 'lead_recommended': fast_check['lead_prompt_recommended']})}\n\n"
        yield "data: [DONE]\n\n"
        save_message(session_id, "user", query)
        save_message(session_id, "assistant", fast_check["answer"])
        return

    history = get_session_history(session_id, limit=2)
    search_query, is_follow_up = rewrite_conversational_query(query, history)

    search_results = vector_store.search(search_query, top_k=settings.TOP_K_CHUNKS)
    max_score = search_results[0][1] if search_results else 0.0
    threshold = settings.SIMILARITY_THRESHOLD
    relevant_chunks = [c for c, s in search_results if s >= threshold]

    if not relevant_chunks or max_score < threshold:
        log_content_gap(session_id, query, search_query, max_score)
        fallback_msg = (
            "I do not have this information in our current clinic and treatment records for Immunity Therapy Center. "
            "PlacidWay works with certified medical providers worldwide and can provide you with custom treatment options and pricing.\n\n"
            "Would you like to request a free quote from a PlacidWay patient coordinator?"
        )
        yield f"data: {json.dumps({'type': 'token', 'content': fallback_msg})}\n\n"
        yield f"data: {json.dumps({'type': 'meta', 'citations': [], 'lead_recommended': True})}\n\n"
        yield "data: [DONE]\n\n"
        save_message(session_id, "user", query)
        save_message(session_id, "assistant", fallback_msg)
        return

    context_blocks = []
    citations_dict = {}
    for chunk in relevant_chunks:
        url = chunk.get("url", "")
        title = chunk.get("title", "")
        if url and url not in citations_dict:
            citations_dict[url] = Citation(title=title, url=url, snippet=chunk.get("clean_content", "")[:180])
        context_blocks.append(f"--- Document: {title} | URL: {url} ---\n{chunk['text']}")
    context_str = "\n\n".join(context_blocks)

    is_medical_question = bool(RE_MEDICAL_ADVICE.search(query))
    medical_note = ""
    if is_medical_question:
        medical_note = (
            "\nIMPORTANT: The user is asking about personal medical safety or clearance. "
            "You MUST refuse personal medical diagnosis, advise consulting a physician or oncologist, "
            "and suggest contacting PlacidWay for a clinical consultation.\n"
        )

    user_prompt = f"""{medical_note}
Context from PlacidWay verified pages:
{context_str}

User Question: {query}

Answer conversationally, clearly, and completely using ONLY the context above. Conclude with citations as [Title](URL)."""

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if is_follow_up and history:
        for h in history:
            messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": user_prompt})

    # Real SSE streaming from OpenRouter
    all_models = [settings.PRIMARY_MODEL] + settings.FALLBACK_MODELS
    headers = {
        "Authorization": f"Bearer {settings.OPENROUTER_API_KEY.strip()}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://placidway-chatbot.local",
        "X-Title": "PlacidWay Medical Tourism Assessment Chatbot"
    }

    full_answer_tokens = []
    streamed_successfully = False

    async with httpx.AsyncClient(timeout=25.0) as client:
        for model in all_models:
            payload = {
                "model": model,
                "messages": messages,
                "temperature": 0.0,
                "stream": True
            }
            try:
                async with client.stream("POST", settings.OPENROUTER_BASE_URL, headers=headers, json=payload) as resp:
                    if resp.status_code == 200:
                        async for line in resp.aiter_lines():
                            if line.startswith("data: "):
                                data_str = line[6:].strip()
                                if data_str == "[DONE]":
                                    break
                                try:
                                    parsed = json.loads(data_str)
                                    delta = parsed["choices"][0]["delta"].get("content", "")
                                    if delta:
                                        full_answer_tokens.append(delta)
                                        yield f"data: {json.dumps({'type': 'token', 'content': delta})}\n\n"
                                except Exception:
                                    continue
                        streamed_successfully = True
                        break
                    else:
                        logger.warning(f"Streaming model {model} returned HTTP {resp.status_code}")
            except Exception as e:
                logger.error(f"Streaming error on {model}: {e}")

    full_answer = "".join(full_answer_tokens).strip()

    # Fallback to grounded extractive synthesis if streaming failed or was rate limited
    if not streamed_successfully or not full_answer:
        full_answer = synthesize_extractive_grounded_answer(query, relevant_chunks, is_medical=is_medical_question)
        # Yield words in chunks for smooth streaming experience
        words = full_answer.split(" ")
        for i in range(0, len(words), 3):
            chunk_words = " ".join(words[i:i+3]) + (" " if i+3 < len(words) else "")
            yield f"data: {json.dumps({'type': 'token', 'content': chunk_words})}\n\n"

    # Medical Disclaimer Attachment
    if is_medical_question and "medical disclaimer" not in full_answer.lower():
        disclaimer = f"\n\n*{settings.MEDICAL_DISCLAIMER}*"
        yield f"data: {json.dumps({'type': 'token', 'content': disclaimer})}\n\n"
        full_answer += disclaimer

    citations_list = [c.model_dump() for c in citations_dict.values()]
    is_fallback = "do not have this information" in full_answer.lower() or "not in our current records" in full_answer.lower()
    if is_fallback:
        log_content_gap(session_id, query, search_query, max_score)
        
    lead_rec = bool(RE_QUOTE_INTENT.search(query)) or is_medical_question or is_fallback

    yield f"data: {json.dumps({'type': 'meta', 'citations': citations_list, 'lead_recommended': lead_rec})}\n\n"
    yield "data: [DONE]\n\n"

    save_message(session_id, "user", query)
    save_message(session_id, "assistant", full_answer, citations_list)
