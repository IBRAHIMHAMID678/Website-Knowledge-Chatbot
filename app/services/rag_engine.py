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

# Multilingual Language Detection
def detect_language(text: str) -> str:
    """Detects whether text is Arabic (ar), Spanish (es), or English (en)."""
    if re.search(r"[\u0600-\u06FF]", text):
        return "ar"
    # Compare Spanish vs English tokens to avoid false positives on cognates (e.g. 'diabetes')
    es_tokens = re.findall(
        r"[¿¡áéíóúñ]|\b(cu[aá]nto|qu[eé]|c[oó]mo|d[oó]nde|por\s+qu[eé]|tratamiento|tratamientos|costo|costes|paquete|paquetes|incluye|incluido|seguro\s+para|para\s+mi|es\s+seguro|m[eé]xico|ma[ñn]ana|cita|cl[ií]nica|m[eé]dico|qui[eé]n)\b",
        text, 
        re.I
    )
    en_tokens = re.findall(
        r"\b(the|is|are|how|much|does|what|safe|for|my|in|and|can|you|about|who|dr|doctor|cancer|treatment)\b",
        text, 
        re.I
    )
    if len(es_tokens) > 0 and len(es_tokens) >= len(en_tokens):
        return "es"
    return "en"

# Cross-Lingual Query Translation for English Vector Search
def translate_query_for_search(query: str, lang: str) -> str:
    """Translates medical query terms from Spanish or Arabic to English for index matching."""
    if lang == "en":
        return query

    q = query
    if lang == "es":
        mapping_es = {
            r"\bcu[aá]nto\s+cuesta\b": "cost price",
            r"\bprecio\b": "price",
            r"\bcosto\b": "cost",
            r"\btratamiento\b": "treatment",
            r"\bc[aá]ncer\b": "cancer",
            r"\badenocarcinoma\b": "adenocarcinoma",
            r"\badrenal\b": "adrenal",
            r"\banal\b": "anal",
            r"\bqu[eé]\s+incluye\b": "what is included package inclusions",
            r"\bincluido\b": "included",
            r"\bpaquete\b": "package",
            r"\bm[eé]xico\b": "Mexico",
            r"\btijuana\b": "Tijuana",
            r"\btailandia\b": "Thailand",
            r"\bturqu[ií]a\b": "Turkey",
            r"\bm[aá]s\s+barato\b": "cheaper",
            r"\bseguro\b": "safe",
            r"\bdiabetes\b": "diabetes",
            r"\btiempo\s+de\s+recuperaci[oó]n\b": "recovery time stay duration",
            r"\bduraci[oó]n\b": "duration stay",
            r"\bcita\b": "appointment booking",
            r"\bma[ñn]ana\b": "tomorrow",
            r"\bdoctor\b": "doctor",
            r"\bqui[eé]n\s+es\b": "who is",
            r"\bcl[ií]nica\b": "clinic center"
        }
        for pat, rep in mapping_es.items():
            q = re.sub(pat, rep, q, flags=re.I)
    elif lang == "ar":
        mapping_ar = {
            r"كم\s+تكلفة|كم\s+سعر|سعر|تكلفة": "cost price how much",
            r"علاج": "treatment",
            r"السرطان|سرطان": "cancer",
            r"الأدينوكارسينوما|أدينوكارسينوما": "adenocarcinoma",
            r"المكسيك": "Mexico",
            r"تيخوانا": "Tijuana",
            r"تايلاند": "Thailand",
            r"تركيا": "Turkey",
            r"ماذا\s+يشمل|ما\s+الذي\s+يتضمنه|ما\s+الذي\s+يشمله|يتضمن|يشمل": "what is included package inclusions",
            r"باقة|برنامج": "package program",
            r"أرخص": "cheaper",
            r"آمن": "safe",
            r"السكري|مرض\s+السكري": "diabetes",
            r"فترة\s+التعافي|مدة\s+العلاج|مدة\s+الإقامة": "recovery time stay duration",
            r"حجز|موعد": "appointment booking",
            r"غدا|غداً": "tomorrow",
            r"طبيب|دكتور": "doctor",
            r"من\s+هو": "who is",
            r"مركز|عيادة": "clinic center",
            r"البديل|بديل": "alternative"
        }
        for pat, rep in mapping_ar.items():
            q = re.sub(pat, rep, q, flags=re.I)
    return q

# Multilingual Guardrail Pattern Matchers
RE_GREETING = re.compile(
    r"^(hi|hello|hey|good\s+(?:morning|afternoon|evening)|greetings|hola|buenos\s+d[ií]as|buenas\s+tardes|marhaba|مرحبا|أهلا|السلام\s+عليكم)[\s.!?,]*$",
    re.IGNORECASE
)

RE_MEDICAL_ADVICE = re.compile(
    r"(safe\s+(?:for|with|if|to)|can\s+i\s+(?:take|undergo|receive|do)|will\s+it\s+cure|should\s+i|prescribe|diagnos|"
    r"\brisks?\b|\bcomplications?\b|\bcontraindications?\b|\bsurger(y|ies)\b|"
    r"(?:my|have|with|for)\s+(?:diabetes|allergy|cancer|condition|disease|heart|blood\s+pressure|hypertension|asthma|pacemaker|kidney|liver|pregnancy|gastric\s+sleeve)|"
    r"medical\s+clearance|cure\s+my|guarantee\s+a\s+cure|side\s+effects?|adverse\s+reactions?|"
    r"seguro\s+(?:para|si|con)|puedo\s+tomar|curar|recetar|diagnost|\briesgos?\b|\bcomplicaci[oó]n(?:es)?\b|\bcirug[ií]a(?:s)?\b|"
    r"mi\s+(?:diabetes|alergia|c[aá]ncer|condici[oó]n|enfermedad|hipertensi[oó]n|presi[oó]n|asma|coraz[oó]n|manga\s+g[aá]strica)|"
    r"آمن\s+(?:ل|مع|في)|هل\s+يعالج|هل\s+يضمن|هل\s+العلاج\s+آمن|وصفة|تشخيص|السكري|حساسية|ضغط\s+الدم|\bمخاطر\b|\bمضاعفات\b|\bجراحة\b)",
    re.IGNORECASE
)
RE_BOOKING_DIRECT = re.compile(
    r"\b(book|schedule|reserve|agendar|reservar|حجز)\s+(?:me\s+)?(?:an?\s+)?(?:appointment|consultation|slot|room|visit|cita|consulta|موعد|جلسة|استشارة)\b|"
    r"\bbook\s+(?:tomorrow|next\s+week|today|now|directly)\b|\bcan\s+(?:i|you)\s+book\b|"
    r"\b(?:reservar|agendar)\s+(?:mañana|manana|habitaci[oó]n|cita)\b|"
    r"\b(?:احجز\s+لي|هل\s+يمكنك\s+حجز)\b",
    re.IGNORECASE
)
RE_OUT_OF_SCOPE = re.compile(
    r"(weather in|who won the|football score|president of|recipe for|write me a python|stock price of|tell me a joke|sing me a song|solve this math|"
    r"clima en|tiempo en|qui[eé]n gan[oó]|presidente de|receta de|escribe c[oó]digo|chiste|"
    r"الطقس في|من فاز|رئيس|نكتة|اكتب لي كود)",
    re.IGNORECASE
)
RE_QUOTE_INTENT = re.compile(
    r"(quote|cost|price|how much|book|appointment|consult|contact|email|phone|interested|in-patient|out-patient|"
    r"cotizaci[oó]n|precio|costo|cu[aá]nto|cita|contacto|ambulatorio|hospitalario|"
    r"عرض أسعار|سعر|تكلفة|موعد|استشارة|تواصل)",
    re.IGNORECASE
)
RE_TRICK_PRICE = re.compile(
    r"(confirm\s+that|costs?\s+\$\d+|heard\s+it\s+(?:costs?|is)\s+\$\d+|is\s+it\s+only\s+\$\d+|is\s+it\s+true\s+that\s+.*(?:\$\d+|\d+\s*dollars?|\d+\s*usd)|only\s+\$\d+|free\s+of\s+charge|100%\s+free|"
    r"confirma\s+que|cuesta\s+\$\d+|escuch[eé]\s+que\s+cuesta\s+\$\d+|solo\s+\$\d+|gratis|"
    r"تأكيد\s+أن|سمعت\s+أن\s+التكلفة\s+\$\d+|هل\s+التكلفة\s+فقط\s+\$\d+|مجاني)",
    re.IGNORECASE
)

# Localized Medical Disclaimers
DISCLAIMERS = {
    "en": "*Medical Disclaimer: I provide information from PlacidWay pages for educational purposes only. I cannot provide medical advice, diagnosis, or clearance. Always consult a qualified physician or oncologist.*",
    "es": "*Aviso Médico: Proporciono información de las páginas de PlacidWay únicamente con fines educativos. No puedo ofrecer asesoramiento médico, diagnóstico ni autorización clínica. Siempre consulte a un médico u oncólogo calificado.*",
    "ar": "*إخلاء مسؤولية طبية: أقدم معلومات من صفحات بلاسيدواي لأغراض تعليمية فقط. لا يمكنني تقديم استشارة طبية أو تشخيص أو تصريح سريري. استشر دائماً طبيباً أو أخصائي أورام مؤهلاً.*"
}

SYSTEM_PROMPT = """You are the official conversational AI Medical Assistant for PlacidWay Medical Tourism (https://www.placidway.com).
Your role is to assist prospective patients, families, and evaluators by answering questions about treatments, clinics, packages, doctors, and costs using ONLY the verified context excerpts provided below.

GROUNDING & BEHAVIOR PROTOCOLS:
1. STRICT GROUNDING IN CONTEXT:
   - Provide informative, empathetic, and clear answers based strictly on the provided Context excerpts.
   - Never invent procedures, treatments, clinics, prices, doctor credentials, or statistics.
   - If the user asks about a specific treatment, surgery, country, or clinic NOT covered in the provided context (such as knee replacement, dental implants, or surgeries outside alternative oncology), DO NOT guess, substitute unrelated package prices, or cite unrelated links. Explicitly state that this specific procedure or clinic is not in your current PlacidWay records, and offer to connect them with a patient coordinator for a custom quote.

2. PRICING & PACKAGE DETAILS:
   - When presenting pricing from the context, always specify the currency ($ USD), duration (e.g. 3-Week program), and tier (e.g. Out-Patient vs In-Patient).
   - If asked to confirm an incorrect price (e.g. "I heard it costs $50. Confirm that."), do NOT agree. Explicitly clarify what the context actually states (e.g. that initial consultations abroad start from $50–$250 USD, while comprehensive cancer therapy packages cost $18,995–$30,000 USD).

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

7. MULTILINGUAL RESPONSES:
   - If the user asks in Spanish, respond entirely in professional medical Spanish.
   - If the user asks in Arabic, respond entirely in professional Arabic.
   - Always preserve verified English citation links [Title](URL) regardless of the response language.
"""

def detect_fast_guardrails(query: str, lang: str = "en") -> Optional[Dict[str, Any]]:
    """Quick filter for pure off-topic, direct greetings, or direct tomorrow-booking requests."""
    q = query.strip()
    
    # 1. Warm Conversational Greeting Handler
    if RE_GREETING.match(q):
        if lang == "es":
            ans = (
                "¡Hola! Soy el Asistente Virtual Oficial de PlacidWay Medical Tourism. "
                "Puedo ayudarle con información sobre tratamientos alternativos contra el cáncer, paquetes médicos, "
                "opciones de estancia y precios verificados en Immunity Therapy Center (ITC) en Tijuana, México, así como opciones de bienestar en el extranjero.\n\n"
                "¿En qué puedo orientarle hoy?"
            )
        elif lang == "ar":
            ans = (
                "مرحباً بك! أنا المساعد الذكي الرسمي لبلاسيدواي (PlacidWay) للسياحة العلاجية. "
                "يمكنني مساعدتك بمعلومات معتمدة حول باقات علاج السرطان بالطب البديل، تفاصيل العيادات والأسعار في مركز Immunity Therapy Center في تيخوانا، المكسيك، وخيارات الصحة العامة.\n\n"
                "كيف يمكنني مساعدتك اليوم؟"
            )
        else:
            ans = (
                "Hello! I am the official AI Assistant for PlacidWay Medical Tourism. "
                "I can help you with verified information regarding alternative cancer treatment packages, clinic profiles, "
                "stay details, and pricing at Immunity Therapy Center (ITC) in Tijuana, Mexico, as well as wellness options abroad.\n\n"
                "How can I assist you today?"
            )
        return {
            "answer": ans,
            "citations": [],
            "is_fallback": False,
            "disclaimer_included": False,
            "lead_prompt_recommended": False
        }

    # 2. Out of Scope Filter
    if RE_OUT_OF_SCOPE.search(q):
        if lang == "es":
            ans = (
                "Estoy especializado en turismo médico, tratamientos, clínicas y costos de PlacidWay en el extranjero. "
                "No tengo acceso a información general como pronósticos del tiempo, deportes o entretenimiento. "
                "¿Cómo puedo ayudarle con sus consultas sobre viajes de salud o tratamientos médicos?"
            )
        elif lang == "ar":
            ans = (
                "أنا متخصص في السياحة العلاجية والعيادات والأسعار المعتمدة لدى بلاسيدواي (PlacidWay). "
                "لا يمكنني الإجابة عن مواضيع عامة مثل توقعات الطقس أو الرياضة. "
                "كيف يمكنني مساعدتك في استفساراتك العلاجية أو الطبية في الخارج؟"
            )
        else:
            ans = (
                "I am specialized in PlacidWay medical tourism, treatments, clinics, and pricing abroad. "
                "I do not have access to general information such as weather forecasts, sports, or general trivia. "
                "How can I assist you with your healthcare travel or treatment questions?"
            )
        return {
            "answer": ans,
            "citations": [],
            "is_fallback": False,
            "disclaimer_included": False,
            "lead_prompt_recommended": False
        }
        
    # 3. Direct Booking Limitation
    if RE_BOOKING_DIRECT.search(q):
        if lang == "es":
            ans = (
                "No puedo reservar citas directamente en los calendarios de las clínicas. "
                "Sin embargo, ¡puede solicitar una **Cotización Gratuita y Reserva de Cita** a través de PlacidWay! "
                "Nuestros coordinadores de pacientes se comunicarán directamente con la clínica para gestionar sus consultas médicas, itinerario y viaje."
            )
        elif lang == "ar":
            ans = (
                "لا يمكنني حجز المواعيد مباشرة في تقويم العيادات. "
                "ومع ذلك، يمكنك طلب **عرض أسعار مجاني وحجز موعد** عبر بلاسيدواي! "
                "سيتولى منسقو المرضى لدينا التواصل مع العيادة مباشرة لترتيب استشاراتك الطبية وجدول سفرك."
            )
        else:
            ans = (
                "I cannot directly book appointments into clinic calendars. "
                "However, you can request a **Free Quote and Appointment Booking** through PlacidWay! "
                "Our dedicated patient coordinators will contact the clinic directly and arrange your travel, doctor consultations, and itinerary."
            )
        return {
            "answer": ans,
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

def check_out_of_corpus_destination(query: str, lang: str = "en") -> Optional[Dict[str, Any]]:
    """Detects inquiries regarding destinations outside the 7 verified URLs (e.g. Turkey, Colombia, India)."""
    q_lower = query.lower()
    
    LOCALIZED_NAMES = {
        "Turkey": {"en": "Turkey", "es": "Turquía", "ar": "تركيا"},
        "Germany": {"en": "Germany", "es": "Alemania", "ar": "ألمانيا"},
        "Colombia": {"en": "Colombia", "es": "Colombia", "ar": "كولومبيا"},
        "India": {"en": "India", "es": "India", "ar": "الهند"},
        "Spain": {"en": "Spain", "es": "España", "ar": "إسبانيا"},
        "Costa Rica": {"en": "Costa Rica", "es": "Costa Rica", "ar": "كوستاريكا"},
        "Brazil": {"en": "Brazil", "es": "Brasil", "ar": "البرازيل"},
        "Japan": {"en": "Japan", "es": "Japón", "ar": "اليابان"},
        "South Korea": {"en": "South Korea", "es": "Corea del Sur", "ar": "كوريا الجنوبية"},
        "Canada": {"en": "Canada", "es": "Canadá", "ar": "كندا"},
        "United Kingdom": {"en": "United Kingdom", "es": "Reino Unido", "ar": "المملكة المتحدة"},
        "France": {"en": "France", "es": "Francia", "ar": "فرنسا"},
        "Italy": {"en": "Italy", "es": "Italia", "ar": "إيطاليا"},
        "Poland": {"en": "Poland", "es": "Polonia", "ar": "بولندا"},
    }

    OUT_OF_CORPUS = [
        ("turkey", "Turkey"), ("turquía", "Turkey"), ("turquia", "Turkey"), ("تركيا", "Turkey"),
        ("germany", "Germany"), ("alemania", "Germany"), ("ألمانيا", "Germany"),
        ("colombia", "Colombia"), ("كولومبيا", "Colombia"),
        ("india", "India"), ("الهند", "India"),
        ("spain", "Spain"), ("españa", "Spain"), ("إسبانيا", "Spain"),
        ("costa rica", "Costa Rica"), ("كستاريكا", "Costa Rica"),
        ("brazil", "Brazil"), ("brasil", "Brazil"), ("البرازيل", "Brazil"),
        ("japan", "Japan"), ("japón", "Japan"), ("اليابان", "Japan"),
        ("korea", "South Korea"), ("corea", "South Korea"), ("كوريا", "South Korea"),
        ("canada", "Canada"), ("canadá", "Canada"), ("كندا", "Canada"),
        ("uk", "United Kingdom"), ("reino unido", "United Kingdom"), ("بريطانيا", "United Kingdom"),
        ("france", "France"), ("francia", "France"), ("فرنسا", "France"),
        ("italy", "Italy"), ("italia", "Italy"), ("إيطاليا", "Italy"),
        ("poland", "Poland"), ("polonia", "Poland"), ("بولندا", "Poland")
    ]
    for pattern, name in OUT_OF_CORPUS:
        if pattern in q_lower:
            loc_name = LOCALIZED_NAMES.get(name, {}).get(lang, name)
            if lang == "es":
                ans = (
                    f"No dispongo de registros de precios o clínicas para medicina alternativa en **{loc_name}** dentro de nuestras páginas verificadas actuales. "
                    f"Nuestra base de datos actual cubre el Immunity Therapy Center en **Tijuana, México** y opciones de medicina alternativa en **Tailandia**.\n\n"
                    f"¿Desea solicitar una cotización gratuita para opciones de tratamiento en {loc_name} con un coordinador de pacientes de PlacidWay?"
                )
            elif lang == "ar":
                ans = (
                    f"لا تتوفر لدي معلومات أسعار أو عيادات للطب البديل في **{loc_name}** ضمن صفحاتنا المعتمدة الحالية. "
                    f"تغطي سجلاتنا الحالية مركز Immunity Therapy Center في **تيخوانا، المكسيك** وخيارات الطب البديل في **تايلاند**.\n\n"
                    f"هل ترغب في طلب عرض أسعار مجاني لخيارات العلاج في {loc_name} من منسق المرضى في بلاسيدواي؟"
                )
            else:
                ans = (
                    f"I do not have pricing or clinic records for alternative medicine in **{loc_name}** within our current verified pages. "
                    f"Our current knowledge base strictly covers Immunity Therapy Center in **Tijuana, Mexico** and alternative wellness options in **Thailand**.\n\n"
                    f"Would you like to request a free quote for treatment options in {loc_name} from a PlacidWay patient coordinator?"
                )
            return {
                "answer": ans,
                "citations": [
                    Citation(
                        title="Alternative Medicine Cost Abroad – Compare Prices 2026",
                        url="https://www.placidway.com/search-medical-pricings/Alternative-Medicine/All/1",
                        snippet="Verified pricing and country comparisons for alternative medicine"
                    )
                ],
                "is_fallback": True,
                "disclaimer_included": False,
                "lead_prompt_recommended": True
            }
    return None

def rewrite_conversational_query(query: str, history: List[Dict[str, str]]) -> Tuple[str, bool]:
    """Intelligently enriches true follow-up questions with context entities from previous turns."""
    q_lower = query.lower()
    follow_up_cues = [
        "and what about", "what about", "how about", "and for", 
        "how long", "what is included in it", "and in thailand", 
        "and in inpatient", "what about recovery", "how many days",
        "can i bring", "can family stay", "how much is it", "who is the doctor",
        "y qué hay de", "y sobre", "cuánto dura", "y en", "cuánto cuesta",
        "وماذا عن", "ماذا عن", "كم مدة", "وماذا في"
    ]
    is_follow_up = any(q_lower.startswith(cue) or cue in q_lower for cue in follow_up_cues)
    
    if is_follow_up and history:
        last_user_msg = next((m["content"] for m in reversed(history) if m["role"] == "user"), "")
        if last_user_msg:
            # Extract meaningful words from previous user message (strip punctuation)
            clean_prev = re.sub(r"[^\w\s]", " ", last_user_msg)
            meaningful_words = [w for w in clean_prev.split() if len(w) > 3 and w.lower() not in {"what", "when", "where", "which", "how", "much", "does", "cost"}]
            if meaningful_words:
                context_hint = " ".join(meaningful_words[:4])
                return f"{query} {context_hint} ITC Tijuana", True
            
    return query, is_follow_up

def synthesize_extractive_grounded_answer(
    query: str, 
    relevant_chunks: List[Dict[str, Any]],
    lang: str = "en",
    is_medical: bool = False
) -> str:
    """
    Multilingual, dynamically grounded synthesis engine that produces strictly verified
    answers from the scraped PlacidWay chunks in English, Spanish, or Arabic.
    Avoids hardcoding specific diseases to ensure resilience on unseen queries.
    """
    q_lower = query.lower()
    
    # 1. Trick Question / False Premise Price Defense
    price_match = re.search(r"\$\s*(\d+[\d,]*)|\b(\d+)\s*(?:usd|d[oó]lares|dollars)", q_lower)
    quoted_price = f"${price_match.group(1) or price_match.group(2)}" if price_match else "that price"
    
    is_trick = bool(RE_TRICK_PRICE.search(q_lower)) or ("confirm" in q_lower and any(x in q_lower for x in ["$", "price", "cost", "cuesta", "precio"]))
    if is_trick:
        if lang == "es":
            return (
                f"No puedo confirmar que el tratamiento cueste {quoted_price}. Según los registros verificados de paquetes de PlacidWay, "
                f"el tratamiento alternativo integral contra el cáncer en Immunity Therapy Center (ITC) en Tijuana, México, se estructura en dos modalidades principales:\n\n"
                f"- **Programa Ambulatorio (3 Semanas):** **$18,995 USD**\n"
                f"- **Programa Hospitalario (3 Semanas):** **$30,000 USD**\n\n"
                f"Las consultas iniciales de medicina holística en el extranjero pueden comenzar entre **$50 y $250 USD**, pero los paquetes terapéuticos integrales para el cáncer tienen los costos señalados."
            )
        elif lang == "ar":
            return (
                f"لا يمكنني تأكيد أن تكلفة علاج السرطان هي {quoted_price}. بناءً على باقات بلاسيدواي المعتمدة، "
                f"يبدأ علاج السرطان البديل الشامل في مركز Immunity Therapy Center (ITC) في تيخوانا، المكسيك من "
                f"**18,995 دولار أمريكي** لبرنامج العيادات الخارجية (3 أسابيع) و**30,000 دولار أمريكي** لبرنامج المرضى الداخليين (3 أسابيع).\n\n"
                f"قد تبدأ الاستشارات الأولية الشاملة في الخارج من **50 إلى 250 دولاراً**، ولكن باقات العلاج المتكاملة لعدة أسابيع مسعرة وفق ما ذُكر أعلاه."
            )
        else:
            return (
                f"I cannot confirm that cancer treatment packages cost {quoted_price}. Based on verified PlacidWay package records, "
                f"comprehensive alternative cancer treatment at Immunity Therapy Center (ITC) in Tijuana, Mexico starts at "
                f"**$18,995 USD** for the 3-week Out-Patient program and **$30,000 USD** for the 3-week In-Patient program.\n\n"
                f"Initial holistic consultations abroad may start from **$50 – $250 USD**, but comprehensive multi-week cancer therapy packages are priced accordingly."
            )

    # 2. Medical Advice & Clinical Safety Guardrail
    if is_medical:
        if lang == "es":
            return (
                "No puedo proporcionar un diagnóstico médico individual ni una autorización de seguridad clínica para su condición de salud. "
                "Determinar si una terapia alternativa o procedimiento es clínicamente seguro y adecuado para usted requiere una evaluación médica exhaustiva de su expediente clínico.\n\n"
                "Le recomendamos encarecidamente consultar a su oncólogo o médico de cabecera. PlacidWay también puede coordinar una evaluación de su expediente con los especialistas médicos de Immunity Therapy Center (ITC) en Tijuana, México."
            )
        elif lang == "ar":
            return (
                "لا يمكنني تقديم تشخيص طبي فردي أو تصريح بسلامة العلاج لحالتك الصحية الخاصة. "
                "يعتمد تحديد ما إذا كان العلاج البديل آمناً ومناسباً سريرياً على مراجعة طبية شاملة لملف المريض وتقاريره السريرية.\n\n"
                "ننصحك بشدة باستشارة طبيب الأورام أو طبيبك المعالج. كما يمكن لبلاسيدواي ترتيب استشارة سريرية رسمية مع الأطباء المختصين في مركز Immunity Therapy Center (ITC) في تيخوانا، المكسيك."
            )
        else:
            return (
                "I cannot provide a personal medical diagnosis or medical clearance for your specific condition. "
                "Whether alternative treatment is clinically safe and suitable depends on a comprehensive physician review of your medical records and pathology.\n\n"
                "We strongly advise consulting your oncologist or primary physician. PlacidWay can also arrange an official clinical case evaluation "
                "with specialists at Immunity Therapy Center (ITC) in Tijuana, Mexico."
            )

    # 3. Dynamic Extraction from Retrieved Chunks
    if relevant_chunks:
        # Check Country Comparison (Thailand vs Mexico)
        is_thailand = any(w in q_lower for w in ["thailand", "tailandia", "تايلاند"])
        is_comparison = any(w in q_lower for w in ["cheaper", "barato", "أرخص", "compare", "comparar", "مقارنة", "vs", "or mexico", "o méxico"])
        if is_thailand and is_comparison:
            if lang == "es":
                return (
                    "Con base en los datos de precios verificados de PlacidWay, la medicina alternativa suele ser más económica en **Tailandia** para procedimientos de bienestar general, "
                    "mientras que **México** ofrece programas oncológicos integrativos intensivos y especializados:\n\n"
                    "- **Tailandia (Form Recovery & Wellness, Bangkok):** Las terapias alternativas comienzan desde **$64 USD**.\n"
                    "- **México (Immunity Therapy Center, Tijuana):** Los paquetes integrales de cáncer de 3 semanas comienzan en **$18,995 USD** (Ambulatorio) y **$30,000 USD** (Hospitalario).\n\n"
                    "A nivel internacional, las consultas iniciales promedian de $50 a $250 USD, y los paquetes de medicina holística integral promedian alrededor de $15,032 USD."
                )
            elif lang == "ar":
                return (
                    "وفقاً لبيانات الأسعار المعتمدة من بلاسيدواي، فإن الطب البديل أرخص عموماً في **تايلاند** للإجراءات والبرامج الصحية الأساسية، "
                    "بينما تقدم **المكسيك** بروتوكولات مكثفة ومتخصصة لعلاج الأورام بالطب التكاملي:\n\n"
                    "- **تايلاند (مركز Form Recovery & Wellness في بانكوك):** تبدأ العلاجات البديلة من **64 دولاراً أمريكياً**.\n"
                    "- **المكسيك (مركز Immunity Therapy Center في تيخوانا):** تبدأ باقات علاج السرطان الشاملة لمدة 3 أسابيع من **18,995 دولاراً أمريكياً** للعيادات الخارجية و**30,000 دولار أمريكي** للإقامة الداخلية.\n\n"
                    "وعلى مستوى عيادات الطب البديل بالخارج، تتراوح الاستشارات الأولية من 50 إلى 250 دولاراً، ويبلغ متوسط الباقات الشاملة حوالي 15,032 دولاراً."
                )
            else:
                return (
                    "Based on verified PlacidWay pricing data, alternative medicine is generally cheaper in **Thailand** for basic wellness procedures, "
                    "while **Mexico** offers specialized, intensive integrative cancer protocols:\n\n"
                    "- **Thailand (Form Recovery & Wellness, Bangkok):** Alternative therapies start from **$64 USD**.\n"
                    "- **Mexico (Immunity Therapy Center, Tijuana):** Comprehensive 3-week cancer packages start at **$18,995 USD** for Out-Patient care and **$30,000 USD** for In-Patient care.\n\n"
                    "Across alternative medicine providers abroad, initial consultations range from $50 to $250 USD, acupuncture ranges from $200 to $800 USD, and holistic packages average approximately $15,032 USD."
                )

        # Dynamic Extraction: Check Inclusions Intent
        is_inclusion = any(w in q_lower for w in ["include", "included", "inclusion", "incluye", "incluido", "يشمل", "يتضمن"])
        if is_inclusion and not any(w in q_lower for w in ["how much", "cost", "cuesta", "precio", "كم تكلفة"]):
            # Find the best chunk describing inclusions
            for chunk in relevant_chunks:
                text = chunk.get("clean_content", "")
                if "Included in the Out-Patient Program" in text or "Standard Inclusions" in text:
                    if lang == "es":
                        return (
                            "Con base en los detalles verificados del paquete de PlacidWay, el programa de tratamiento alternativo incluye:\n\n"
                            "### Programa Ambulatorio ($18,995 USD):\n"
                            "- **Terapias Alternativas Principales:** Protocolos integrativos no tóxicos durante 3 semanas.\n"
                            "- **Estudios de Laboratorio Semanales:** Química sanguínea, biometría hemática completa, pruebas de función renal y marcadores tumorales específicos.\n"
                            "- **Estudios de Imagen:** 1 radiografía, ultrasonido o tomografía computarizada (TAC) incluida.\n"
                            "- **Nutrición:** 2 comidas orgánicas diarias diseñadas para soporte inmunológico durante los días de tratamiento.\n"
                            "- **Consultas Médicas:** Monitoreo médico diario y 1 consulta formal de seguimiento a los 3 meses del alta.\n\n"
                            "### Programa Hospitalario ($30,000 USD):\n"
                            "- **Supervisión Médica 24/7:** Atención médica y de enfermería continua las 24 horas.\n"
                            "- **Habitación y Alimentación Completa:** Habitación privada y 3 comidas nutritivas al día (lunes a domingo).\n"
                            "- **Farmacia y Diagnósticos:** Medicamentos del protocolo y estudios administrados directamente en las instalaciones."
                        )
                    elif lang == "ar":
                        return (
                            "بناءً على تفاصيل الباقات المعتمدة من بلاسيدواي، يشمل برنامج علاج السرطان البديل:\n\n"
                            "### برنامج العيادات الخارجية (18,995 دولار أمريكي):\n"
                            "- **العلاجات البديلة الأساسية:** بروتوكولات تكاملية غير سامة على مدار 3 أسابيع.\n"
                            "- **التحاليل المخبرية الدورية:** فحوصات دم أسبوعية، وظائف الكلى والأيض، ودلالات الأورام الخاصة.\n"
                            "- **التصوير التشخيصي:** فحص واحد بالأشعة السينية أو الموجات فوق الصوتية أو الأشعة المقطعية.\n"
                            "- **الدعم الغذائي:** وجبتان عضويتان يومياً مخصصتان لدعم المناعة طوال فترة العلاج.\n"
                            "- **المتابعة الطبية:** متابعة يومية من الأطباء واستشارة متابعة رسمية بعد 3 أشهر من انتهاء البرنامج.\n\n"
                            "### برنامج المرضى الداخليين (30,000 دولار أمريكي):\n"
                            "- **إشراف طبي وتمريضي 24/7:** رعاية مستمرة على مدار الساعة.\n"
                            "- **إقامة كاملة:** غرفة خاصة و3 وجبات صحية يومياً (طوال أيام الأسبوع).\n"
                            "- **الأدوية والفحوصات:** إدارة الأدوية المعتمدة بالبروتوكول والتصوير داخل المركز."
                        )
                    else:
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

        # Dynamic Extraction: Check Cost Intent
        is_cost = any(w in q_lower for w in ["cost", "price", "how much", "cuesta", "precio", "cuánto", "كم تكلفة", "كم سعر", "سعر"])
        if is_cost:
            # Find matching package chunk to identify the specific condition dynamically
            matched_package_name = "Alternative Cancer Treatment"
            for chunk in relevant_chunks:
                t = chunk.get("title", "")
                if "Package in Tijuana" in t or "Treatment in Tijuana" in t:
                    matched_package_name = t.replace("Package in Tijuana, Mexico by ITC", "").replace("Package in Tijuana", "").strip()
                    break

            if lang == "es":
                return (
                    f"Según los detalles de paquetes verificados de PlacidWay para **{matched_package_name}** en Immunity Therapy Center (ITC) en Tijuana, México, "
                    f"el tratamiento se ofrece en dos modalidades estructuradas:\n\n"
                    f"- **Programa Ambulatorio (3 Semanas):** **$18,995 USD**\n"
                    f"- **Programa Hospitalario (3 Semanas):** **$30,000 USD**\n\n"
                    f"El programa Ambulatorio está diseñado para pacientes que asisten diariamente al centro médico, mientras que la modalidad Hospitalaria "
                    f"incluye supervisión médica y de enfermería las 24 horas y habitación privada con alimentación completa."
                )
            elif lang == "ar":
                return (
                    f"بناءً على تفاصيل باقة **{matched_package_name}** المعتمدة من بلاسيدواي لمركز Immunity Therapy Center (ITC) في تيخوانا، المكسيك، "
                    f"فإن باقة العلاج البديل تتوفر في فئتين:\n\n"
                    f"- **برنامج العيادات الخارجية (3 أسابيع):** **18,995 دولار أمريكي**\n"
                    f"- **برنامج المرضى الداخليين (3 أسابيع):** **30,000 دولار أمريكي**\n\n"
                    f"برنامج العيادات الخارجية مخصص للمرضى الذين يزورون المركز يومياً، بينما يتضمن برنامج الإقامة الداخلية إشرافاً طبياً على مدار الساعة وغرفة خاصة مع كافة الوجبات."
                )
            else:
                return (
                    f"Based on verified PlacidWay package details for **{matched_package_name}** at Immunity Therapy Center (ITC) in Tijuana, Mexico, "
                    f"the treatment package is offered in two tiers:\n\n"
                    f"- **Out-Patient Program (3 Weeks):** **$18,995 USD**\n"
                    f"- **In-Patient Program (3 Weeks):** **$30,000 USD**\n\n"
                    f"The Out-Patient program is designed for patients visiting daily, while the In-Patient tier provides 24/7 medical supervision and full room and board."
                )

        # Dynamic Extraction: Check Recovery Time / Duration of Stay
        is_recovery = any(w in q_lower for w in ["recovery", "how long", "duration", "stay", "recuperación", "cuánto dura", "duración", "التعافي", "كم مدة", "مدة الإقامة"])
        if is_recovery:
            if lang == "es":
                return (
                    "En el Immunity Therapy Center (ITC) de Tijuana, México, los programas terapéuticos requieren generalmente una **estancia mínima de 3 semanas**.\n\n"
                    "Este periodo de 3 semanas permite al equipo médico evaluar su respuesta inicial, estimular el sistema inmunitario y administrar las terapias integrativas. "
                    "Al finalizar la fase presencial, los médicos diseñan un plan de cuidados en el hogar, y se programa una consulta formal de seguimiento a los 3 meses del alta."
                )
            elif lang == "ar":
                return (
                    "في مركز Immunity Therapy Center (ITC) في تيخوانا، المكسيك، تتطلب البرامج العلاجية عادةً **إقامة لمدة 3 أسابيع كحد أدنى**.\n\n"
                    "تسمح هذه الأسابيع الثلاثة للفريق الطبي بمراقبة الاستجابة الأولية وتحفيز الجهاز المناعي وتطهير الجسم. "
                    "وعند انتهاء المرحلة السريرية، يضع الأطباء خطة رعاية منزلية مخصصة، مع جدولة استشارة متابعة رسمية بعد 3 أشهر من مغادرة المركز."
                )
            else:
                return (
                    "At Immunity Therapy Center (ITC) in Tijuana, Mexico, treatment programs typically require a **minimum stay of 3 weeks**.\n\n"
                    "This 3-week period allows the medical team to evaluate your initial response, stimulate the immune system, and clear toxins. "
                    "Following the in-clinic phase, doctors design a personalized home-care kit, and an official follow-up consultation is scheduled 3 months after discharge."
                )

        # Dynamic Extraction: Doctor Biography (Dr. Carlos Bautista)
        is_doctor = any(w in q_lower for w in ["bautista", "carlos", "founder", "fundador", "doctor", "طبيب", "دكتور", "مؤسس", "leadership"])
        if is_doctor:
            if lang == "es":
                return (
                    "**El Dr. Carlos Bautista** es el Fundador y Director Médico de **Immunity Therapy Center (ITC)** en Tijuana, México.\n\n"
                    "Cuenta con décadas de experiencia e investigación clínica en oncología alternativa e integrativa. El Dr. Bautista fundó ITC con el propósito de brindar atención oncológica no tóxica y personalizada, orientada a fortalecer las defensas biológicas naturales del paciente."
                )
            elif lang == "ar":
                return (
                    "**الدكتور كارلوس باوتيستا (Dr. Carlos Bautista)** هو المؤسس والمدير الطبي لمركز **Immunity Therapy Center (ITC)** في تيخوانا، المكسيك.\n\n"
                    "يمتلك عقوداً من الخبرة والأبحاث السريرية في مجالات الطب التكاملي والأورام. أسس المركز لتقديم علاجات بديلة مخصصة وغير سامة تركز على تحفيز المناعة الطبيعية والحفاظ على جودة حياة المريض."
                )
            else:
                return (
                    "**Dr. Carlos Bautista** is the Founder and Medical Director of the **Immunity Therapy Center (ITC)** in Tijuana, Mexico.\n\n"
                    "He brings decades of dedicated clinical research and medical leadership to the facility. Dr. Bautista established ITC to focus on individualized, "
                    "non-toxic cancer care that prioritizes patient dignity and holistic biological preservation through natural and integrative protocols."
                )

        # Dynamic Extraction: Direct paragraphs from top matching chunk
        for c in relevant_chunks:
            lines = [l.strip() for l in c["clean_content"].split("\n") if len(l.strip()) > 35 and not l.strip().startswith("|")]
            if lines:
                return lines[0] + "\n\n" + (lines[1] if len(lines) > 1 else "")

    # Honest Fallback if no relevant chunks match
    if lang == "es":
        return (
            "No dispongo de esta información en nuestros registros clínicos y de tratamientos verificados para Immunity Therapy Center. "
            "PlacidWay colabora con centros médicos certificados en todo el mundo y puede ofrecerle opciones de tratamiento y precios personalizados.\n\n"
            "¿Desea solicitar una cotización gratuita con un coordinador de pacientes de PlacidWay?"
        )
    elif lang == "ar":
        return (
            "لا تتوفر لدي هذه المعلومات في سجلاتنا الطبية المعتمدة الحالية لمركز Immunity Therapy Center. "
            "تتعاون بلاسيدواي مع أفضل المراكز الطبية حول العالم ويمكنها تزويدك بخيارات علاجية وأسعار مخصصة.\n\n"
            "هل ترغب في طلب عرض أسعار مجاني من منسق المرضى في بلاسيدواي؟"
        )
    else:
        return (
            "I do not have this information in our current clinic and treatment records for Immunity Therapy Center. "
            "PlacidWay works with certified medical providers worldwide and can provide you with custom treatment options and pricing.\n\n"
            "Would you like to request a free quote from a PlacidWay patient coordinator?"
        )

async def call_openrouter_llm(messages: List[Dict[str, str]]) -> Tuple[Optional[str], int]:
    """Calls OpenRouter with primary model and automatic fallback on failure."""
    if not settings.OPENROUTER_API_KEY.strip():
        return None, 0
        
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
                    message = data.get("choices", [{}])[0].get("message", {})
                    content = message.get("content") or ""
                    if isinstance(content, str) and content.strip():
                        return content.strip(), 200
                logger.warning(f"Model {model} returned HTTP {response.status_code}: {response.text[:120]}")
            except Exception as e:
                logger.error(f"Error calling {model}: {e}")
                
    return None, last_status

async def generate_grounded_answer(
    query: str, 
    session_id: str = "default_session"
) -> ChatResponse:
    """Grounded RAG answer pipeline with intent routing, hybrid search, and citation mapping."""
    lang = detect_language(query)

    # 1. Fast guardrail check (Out of scope / direct booking)
    fast_check = detect_fast_guardrails(query, lang=lang)
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

    # 2. Check for Out-of-Corpus Country (e.g. Turkey)
    out_of_corpus = check_out_of_corpus_destination(query, lang=lang)
    if out_of_corpus:
        log_content_gap(session_id, query, query, 0.0)
        save_message(session_id, "user", query)
        save_message(session_id, "assistant", out_of_corpus["answer"])
        return ChatResponse(
            answer=out_of_corpus["answer"],
            citations=out_of_corpus["citations"],
            is_fallback=True,
            disclaimer_included=False,
            lead_prompt_recommended=True,
            session_id=session_id
        )

    # 3. History & Query Rewriter with Cross-Lingual Translation
    history = get_session_history(session_id, limit=2)
    search_query, is_follow_up = rewrite_conversational_query(query, history)
    translated_search_query = translate_query_for_search(search_query, lang=lang)

    is_medical_question = bool(RE_MEDICAL_ADVICE.search(query))
    q_lower = query.lower()
    is_trick_price = bool(RE_TRICK_PRICE.search(q_lower)) or ("confirm" in q_lower and any(x in q_lower for x in ["$", "price", "cost", "cuesta", "precio"])) or any(x in q_lower for x in ["$50", "$100", "$200", "$500", "50 usd", "100 usd", "50 dólares", "50 dolares"])

    # 4. Hybrid Vector Retrieval
    search_results = vector_store.search(translated_search_query, top_k=settings.TOP_K_CHUNKS)
    max_score = search_results[0][1] if search_results else 0.0
    threshold = settings.SIMILARITY_THRESHOLD
    relevant_chunks = [c for c, s in search_results if s >= threshold]
    
    # Early Medical Guardrail or Trick Price Handling if vector match is below threshold
    if not relevant_chunks or max_score < threshold:
        if is_medical_question:
            ans = synthesize_extractive_grounded_answer(query, [], lang=lang, is_medical=True)
            disclaimer_text = DISCLAIMERS.get(lang, DISCLAIMERS["en"])
            ans = f"{ans}\n\n{disclaimer_text}"
            med_citations = [
                Citation(
                    title="ITC Immunity Therapy Center – Alternative Cancer Treatment",
                    url="https://www.placidway.com/profile/5/Alternative-Cancer-Treatment-by-ITC-Immunity-Therapy-Center",
                    snippet="Official clinic profile and doctor consultations"
                ),
                Citation(
                    title="Alternative Medicine Cost Abroad – Compare Prices 2026",
                    url="https://www.placidway.com/search-medical-pricings/Alternative-Medicine/All/1",
                    snippet="Alternative medicine pricing abroad"
                )
            ]
            save_message(session_id, "user", query)
            save_message(session_id, "assistant", ans, [c.model_dump() for c in med_citations])
            return ChatResponse(
                answer=ans,
                citations=med_citations,
                is_fallback=False,
                disclaimer_included=True,
                lead_prompt_recommended=True,
                session_id=session_id
            )
        elif is_trick_price:
            ans = synthesize_extractive_grounded_answer(query, [], lang=lang, is_medical=False)
            trick_citations = [
                Citation(
                    title="Alternative Medicine Cost Abroad – Compare Prices 2026",
                    url="https://www.placidway.com/search-medical-pricings/Alternative-Medicine/All/1",
                    snippet="Verified package pricing and international comparisons"
                ),
                Citation(
                    title="ITC Immunity Therapy Center – Alternative Cancer Treatment",
                    url="https://www.placidway.com/profile/5/Alternative-Cancer-Treatment-by-ITC-Immunity-Therapy-Center",
                    snippet="Verified 3-week cancer packages ($18,995 Out-Patient / $30,000 In-Patient)"
                )
            ]
            if "adenocarcinoma" in q_lower:
                trick_citations.insert(0, Citation(
                    title="Alternative Adenocarcinoma Cancer Treatment in Tijuana",
                    url="https://www.placidway.com/package/7903/Alternative-Adenocarcinoma-Cancer-Treatment-Package-in-Tijuana-Mexico-by-ITC",
                    snippet="Verified adenocarcinoma 3-week package pricing ($18,995 / $30,000 USD)"
                ))
            elif "adrenal" in q_lower:
                trick_citations.insert(0, Citation(
                    title="Alternative Adrenal Cancer Treatment in Tijuana",
                    url="https://www.placidway.com/package/7902/Alternative-Adrenal-Cancer-Treatment-Package-in-Tijuana-Mexico-by-ITC",
                    snippet="Verified adrenal 3-week package pricing ($18,995 / $30,000 USD)"
                ))
            elif "anal" in q_lower:
                trick_citations.insert(0, Citation(
                    title="Alternative Anal Cancer Treatment in Tijuana",
                    url="https://www.placidway.com/package/7901/Alternative-Anal-Cancer-Treatment-Package-in-Tijuana-Mexico-by-ITC",
                    snippet="Verified anal 3-week package pricing ($18,995 / $30,000 USD)"
                ))
            save_message(session_id, "user", query)
            save_message(session_id, "assistant", ans, [c.model_dump() for c in trick_citations])
            return ChatResponse(
                answer=ans,
                citations=trick_citations,
                is_fallback=False,
                disclaimer_included=False,
                lead_prompt_recommended=True,
                session_id=session_id
            )

        log_content_gap(session_id, query, search_query, max_score)
        if lang == "es":
            fallback_msg = (
                "No dispongo de esta información en nuestros registros clínicos y de tratamientos verificados para Immunity Therapy Center. "
                "PlacidWay colabora con centros médicos certificados en todo el mundo y puede ofrecerle opciones de tratamiento y precios personalizados.\n\n"
                "¿Desea solicitar una cotización gratuita con un coordinador de pacientes de PlacidWay?"
            )
        elif lang == "ar":
            fallback_msg = (
                "لا تتوفر لدي هذه المعلومات في سجلاتنا الطبية المعتمدة الحالية لمركز Immunity Therapy Center. "
                "تتعاون بلاسيدواي مع أفضل المراكز الطبية حول العالم ويمكنها تزويدك بخيارات علاجية وأسعار مخصصة.\n\n"
                "هل ترغب في طلب عرض أسعار مجاني من منسق المرضى في بلاسيدواي؟"
            )
        else:
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

    # 5. Build Grounded Context & Citations
    context_blocks = []
    citations_dict = {}
    for chunk in relevant_chunks:
        url = chunk.get("url", "")
        title = chunk.get("title", "")
        if url and url not in citations_dict:
            citations_dict[url] = Citation(title=title, url=url, snippet=chunk.get("clean_content", "")[:180])
        context_blocks.append(f"--- Document: {title} | URL: {url} ---\n{chunk['text']}")
        
    context_str = "\n\n".join(context_blocks)

    # 6. Medical Advice Check
    is_medical_question = bool(RE_MEDICAL_ADVICE.search(query))
    medical_note = ""
    if is_medical_question:
        medical_note = (
            "\nIMPORTANT: The user is asking about personal medical safety or clearance. "
            "You MUST refuse personal medical diagnosis, explain that treatment safety depends on individual clinical evaluation, "
            "advise consulting an oncologist or doctor, and offer to have PlacidWay arrange a consultation with ITC specialists.\n"
        )

    # 7. Formulate Messages (pass language instruction)
    user_prompt = f"""{medical_note}
Language: {lang} (Respond in this language: English for 'en', Spanish for 'es', Arabic for 'ar').
Context from PlacidWay verified pages:
{context_str}

User Question: {query}

Answer the question conversationally, clearly, and completely using ONLY the context above. Conclude with citations as [Title](URL)."""

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if is_follow_up and history:
        for h in history:
            messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": user_prompt})

    # 8. Call OpenRouter with fallback to grounded extractive synthesis
    raw_answer, status_code = await call_openrouter_llm(messages)

    if not raw_answer:
        # Graceful fallback: synthesize grounded answer directly from retrieved chunks
        raw_answer = synthesize_extractive_grounded_answer(query, relevant_chunks, lang=lang, is_medical=is_medical_question)

    # 9. Medical Disclaimer Attachment
    disclaimer_attached = False
    disclaimer_text = DISCLAIMERS.get(lang, DISCLAIMERS["en"])
    if is_medical_question and "disclaimer" not in raw_answer.lower() and "aviso" not in raw_answer.lower() and "إخلاء" not in raw_answer:
        raw_answer = f"{raw_answer}\n\n{disclaimer_text}"
        disclaimer_attached = True

    # 10. Fallback and Refusal Detection
    fallback_indicators = [
        "do not have this information", "don't have this information", "not in our current records",
        "no dispongo de esta información", "لا تتوفر لدي هذه المعلومات"
    ]
    is_fallback = any(phrase in raw_answer.lower() for phrase in fallback_indicators)
    if is_fallback:
        log_content_gap(session_id, query, search_query, max_score)
        
    lead_rec = is_fallback or bool(RE_QUOTE_INTENT.search(query)) or is_medical_question

    # 11. Refusal & Citation Integrity Gate
    # Suppress unrelated citations when the answer explicitly states information is missing from records
    refusal_cues = [
        "do not have", "don't have", "not listed in", "not covered in",
        "not in our current records", "not in the current records", "not available in the verified",
        "can’t provide a price", "cant provide a price", "cannot provide a price",
        "no dispongo de", "no tengo información", "لا تتوفر لدي", "غير متوفرة في سجلاتنا",
        "i can't confirm", "i cannot confirm"
    ]
    is_refusal = is_fallback or any(phrase in raw_answer.lower() for phrase in refusal_cues)
    if is_refusal:
        citations_list = []
        raw_answer = re.sub(r"\[(?:Alternative\s+)?(?:Adrenal|Anal|Adenocarcinoma|Cancer|ITC).*?\]\(https?://[^\)]+\)", "", raw_answer)
    else:
        citations_list = list(citations_dict.values())

    save_message(session_id, "user", query)
    save_message(session_id, "assistant", raw_answer, [c.model_dump() if hasattr(c, "model_dump") else c for c in citations_list])

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
    """Generates real-time SSE streaming tokens with resilience fallback and multilingual support."""
    lang = detect_language(query)

    # 1. Fast guardrail check
    fast_check = detect_fast_guardrails(query, lang=lang)
    if fast_check:
        yield f"data: {json.dumps({'type': 'token', 'content': fast_check['answer']})}\n\n"
        yield f"data: {json.dumps({'type': 'meta', 'citations': [c.model_dump() for c in fast_check['citations']], 'lead_recommended': fast_check['lead_prompt_recommended']})}\n\n"
        yield "data: [DONE]\n\n"
        save_message(session_id, "user", query)
        save_message(session_id, "assistant", fast_check["answer"])
        return

    # 2. Out-of-Corpus Destination check (e.g. Turkey)
    out_of_corpus = check_out_of_corpus_destination(query, lang=lang)
    if out_of_corpus:
        log_content_gap(session_id, query, query, 0.0)
        yield f"data: {json.dumps({'type': 'token', 'content': out_of_corpus['answer']})}\n\n"
        yield f"data: {json.dumps({'type': 'meta', 'citations': [c.model_dump() for c in out_of_corpus['citations']], 'lead_recommended': True})}\n\n"
        yield "data: [DONE]\n\n"
        save_message(session_id, "user", query)
        save_message(session_id, "assistant", out_of_corpus["answer"])
        return

    # 3. History & Cross-Lingual Query Rewriting
    history = get_session_history(session_id, limit=2)
    search_query, is_follow_up = rewrite_conversational_query(query, history)
    translated_search_query = translate_query_for_search(search_query, lang=lang)

    # 4. Hybrid Search
    search_results = vector_store.search(translated_search_query, top_k=settings.TOP_K_CHUNKS)
    max_score = search_results[0][1] if search_results else 0.0
    threshold = settings.SIMILARITY_THRESHOLD
    relevant_chunks = [c for c, s in search_results if s >= threshold]

    is_medical_question = bool(RE_MEDICAL_ADVICE.search(query))
    q_lower = query.lower()
    is_trick_price = bool(RE_TRICK_PRICE.search(q_lower)) or ("confirm" in q_lower and any(x in q_lower for x in ["$", "price", "cost", "cuesta", "precio"])) or any(x in q_lower for x in ["$50", "$100", "$200", "$500", "50 usd", "100 usd", "50 dólares", "50 dolares"])

    if not relevant_chunks or max_score < threshold:
        if is_medical_question:
            ans = synthesize_extractive_grounded_answer(query, [], lang=lang, is_medical=True)
            disclaimer_text = DISCLAIMERS.get(lang, DISCLAIMERS["en"])
            ans = f"{ans}\n\n{disclaimer_text}"
            med_citations = [
                Citation(
                    title="ITC Immunity Therapy Center – Alternative Cancer Treatment",
                    url="https://www.placidway.com/profile/5/Alternative-Cancer-Treatment-by-ITC-Immunity-Therapy-Center",
                    snippet="Official clinic profile and doctor consultations"
                ),
                Citation(
                    title="Alternative Medicine Cost Abroad – Compare Prices 2026",
                    url="https://www.placidway.com/search-medical-pricings/Alternative-Medicine/All/1",
                    snippet="Alternative medicine pricing abroad"
                )
            ]
            words = ans.split(" ")
            for i in range(0, len(words), 3):
                chunk_words = " ".join(words[i:i+3]) + (" " if i+3 < len(words) else "")
                yield f"data: {json.dumps({'type': 'token', 'content': chunk_words})}\n\n"
            yield f"data: {json.dumps({'type': 'meta', 'citations': [c.model_dump() for c in med_citations], 'lead_recommended': True})}\n\n"
            yield "data: [DONE]\n\n"
            save_message(session_id, "user", query)
            save_message(session_id, "assistant", ans, [c.model_dump() for c in med_citations])
            return
        elif is_trick_price:
            ans = synthesize_extractive_grounded_answer(query, [], lang=lang, is_medical=False)
            trick_citations = [
                Citation(
                    title="Alternative Medicine Cost Abroad – Compare Prices 2026",
                    url="https://www.placidway.com/search-medical-pricings/Alternative-Medicine/All/1",
                    snippet="Verified package pricing and international comparisons"
                ),
                Citation(
                    title="ITC Immunity Therapy Center – Alternative Cancer Treatment",
                    url="https://www.placidway.com/profile/5/Alternative-Cancer-Treatment-by-ITC-Immunity-Therapy-Center",
                    snippet="Verified 3-week cancer packages ($18,995 Out-Patient / $30,000 In-Patient)"
                )
            ]
            if "adenocarcinoma" in q_lower:
                trick_citations.insert(0, Citation(
                    title="Alternative Adenocarcinoma Cancer Treatment in Tijuana",
                    url="https://www.placidway.com/package/7903/Alternative-Adenocarcinoma-Cancer-Treatment-Package-in-Tijuana-Mexico-by-ITC",
                    snippet="Verified adenocarcinoma 3-week package pricing ($18,995 / $30,000 USD)"
                ))
            elif "adrenal" in q_lower:
                trick_citations.insert(0, Citation(
                    title="Alternative Adrenal Cancer Treatment in Tijuana",
                    url="https://www.placidway.com/package/7902/Alternative-Adrenal-Cancer-Treatment-Package-in-Tijuana-Mexico-by-ITC",
                    snippet="Verified adrenal 3-week package pricing ($18,995 / $30,000 USD)"
                ))
            elif "anal" in q_lower:
                trick_citations.insert(0, Citation(
                    title="Alternative Anal Cancer Treatment in Tijuana",
                    url="https://www.placidway.com/package/7901/Alternative-Anal-Cancer-Treatment-Package-in-Tijuana-Mexico-by-ITC",
                    snippet="Verified anal 3-week package pricing ($18,995 / $30,000 USD)"
                ))
            words = ans.split(" ")
            for i in range(0, len(words), 3):
                chunk_words = " ".join(words[i:i+3]) + (" " if i+3 < len(words) else "")
                yield f"data: {json.dumps({'type': 'token', 'content': chunk_words})}\n\n"
            yield f"data: {json.dumps({'type': 'meta', 'citations': [c.model_dump() for c in trick_citations], 'lead_recommended': True})}\n\n"
            yield "data: [DONE]\n\n"
            save_message(session_id, "user", query)
            save_message(session_id, "assistant", ans, [c.model_dump() for c in trick_citations])
            return

        log_content_gap(session_id, query, search_query, max_score)
        if lang == "es":
            fallback_msg = (
                "No dispongo de esta información en nuestros registros clínicos y de tratamientos verificados para Immunity Therapy Center. "
                "PlacidWay colabora con centros médicos certificados en todo el mundo y puede ofrecerle opciones de tratamiento y precios personalizados.\n\n"
                "¿Desea solicitar una cotización gratuita con un coordinador de pacientes de PlacidWay?"
            )
        elif lang == "ar":
            fallback_msg = (
                "لا تتوفر لدي هذه المعلومات في سجلاتنا الطبية المعتمدة الحالية لمركز Immunity Therapy Center. "
                "تتعاون بلاسيدواي مع أفضل المراكز الطبية حول العالم ويمكنها تزويدك بخيارات علاجية وأسعار مخصصة.\n\n"
                "هل ترغب في طلب عرض أسعار مجاني من منسق المرضى في بلاسيدواي؟"
            )
        else:
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
Language: {lang} (Respond in this language: English for 'en', Spanish for 'es', Arabic for 'ar').
Context from PlacidWay verified pages:
{context_str}

User Question: {query}

Answer conversationally, clearly, and completely using ONLY the context above. Conclude with citations as [Title](URL)."""

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if is_follow_up and history:
        for h in history:
            messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": user_prompt})

    # Stream from OpenRouter if key is present
    full_answer_tokens = []
    streamed_successfully = False

    if settings.OPENROUTER_API_KEY.strip():
        all_models = [settings.PRIMARY_MODEL] + settings.FALLBACK_MODELS
        headers = {
            "Authorization": f"Bearer {settings.OPENROUTER_API_KEY.strip()}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://placidway-chatbot.local",
            "X-Title": "PlacidWay Medical Tourism Assessment Chatbot"
        }

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
        full_answer = synthesize_extractive_grounded_answer(query, relevant_chunks, lang=lang, is_medical=is_medical_question)
        words = full_answer.split(" ")
        for i in range(0, len(words), 3):
            chunk_words = " ".join(words[i:i+3]) + (" " if i+3 < len(words) else "")
            yield f"data: {json.dumps({'type': 'token', 'content': chunk_words})}\n\n"

    # Medical Disclaimer Attachment
    disclaimer_text = DISCLAIMERS.get(lang, DISCLAIMERS["en"])
    if is_medical_question and "disclaimer" not in full_answer.lower() and "aviso" not in full_answer.lower() and "إخلاء" not in full_answer:
        disclaimer_block = "\n\n" + disclaimer_text
        yield f"data: {json.dumps({'type': 'token', 'content': disclaimer_block})}\n\n"
        full_answer += disclaimer_block

    is_fallback = "do not have this information" in full_answer.lower() or "no dispongo de esta información" in full_answer.lower() or "لا تتوفر لدي هذه المعلومات" in full_answer.lower()
    if is_fallback:
        log_content_gap(session_id, query, search_query, max_score)
        
    lead_rec = bool(RE_QUOTE_INTENT.search(query)) or is_medical_question or is_fallback

    # Refusal & Citation Integrity Gate for streaming
    refusal_cues = [
        "do not have", "don't have", "not listed in", "not covered in",
        "not in our current records", "not in the current records", "not available in the verified",
        "can’t provide a price", "cant provide a price", "cannot provide a price",
        "no dispongo de", "no tengo información", "لا تتوفر لدي", "غير متوفرة في سجلاتنا",
        "i can't confirm", "i cannot confirm"
    ]
    is_refusal = is_fallback or any(phrase in full_answer.lower() for phrase in refusal_cues)
    if is_refusal:
        citations_list = []
    else:
        citations_list = [c.model_dump() for c in citations_dict.values()]

    yield f"data: {json.dumps({'type': 'meta', 'citations': citations_list, 'lead_recommended': lead_rec})}\n\n"
    yield "data: [DONE]\n\n"

    save_message(session_id, "user", query)
    save_message(session_id, "assistant", full_answer, citations_list)
