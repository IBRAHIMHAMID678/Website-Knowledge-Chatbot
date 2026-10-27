import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    PROJECT_NAME: str = "PlacidWay AI Knowledge Chatbot"
    VERSION: str = "1.0.0"
    ENVIRONMENT: str = "production"
    
    # Groq Cloud API Configuration (Ultra-fast LLM Inference)
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_BASE_URL: str = "https://api.groq.com/openai/v1/chat/completions"
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
    GROQ_FALLBACK_MODELS: list[str] = [
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b"
    ]

    # OpenRouter API Configuration
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1/chat/completions"
    PRIMARY_MODEL: str = os.getenv("PRIMARY_MODEL", "nvidia/nemotron-3.5-lightning:free")
    FALLBACK_MODELS: list[str] = [
        "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
        "dots-studio/dots-3-note-preview:free"
    ]
    
    # Security Configuration
    ADMIN_API_KEY: str = os.getenv("ADMIN_API_KEY", "placidway_admin_key_2026")

    # Supabase Cloud Database Configuration
    SUPABASE_URL: str = os.getenv("SUPABASE_URL", "")
    SUPABASE_KEY: str = os.getenv("SUPABASE_KEY", os.getenv("SUPABASE_SERVICE_ROLE_KEY", os.getenv("SUPABASE_ANON_KEY", "")))
    
    # Data Storage Paths
    IS_VERCEL: bool = bool(os.getenv("VERCEL"))
    DATA_DIR: Path = BASE_DIR / "data"
    SCRAPED_PAGES_DIR: Path = Path("/tmp/scraped_pages") if IS_VERCEL else DATA_DIR / "scraped_pages"
    DB_PATH: Path = Path("/tmp/placidway_bot.db") if IS_VERCEL else DATA_DIR / "placidway_bot.db"
    VECTOR_INDEX_PATH: Path = DATA_DIR / "vector_index.json"
    
    # Ingestion & Crawler Targets (Strictly these 7 URLs)
    TARGET_URLS: list[str] = [
        "https://www.placidway.com/profile/5/Alternative-Cancer-Treatment-by-ITC-Immunity-Therapy-Center",
        "https://www.placidway.com/package/7903/Alternative-Adenocarcinoma-Cancer-Treatment-Package-in-Tijuana-Mexico-by-ITC",
        "https://www.placidway.com/package/7902/Alternative-Adrenal-Cancer-Treatment-Package-in-Tijuana-Mexico-by-ITC",
        "https://www.placidway.com/package/7901/Alternative-Anal-Cancer-Treatment-Package-in-Tijuana-Mexico-by-ITC",
        "https://www.placidway.com/profile-videos/5/Alternative-Cancer-Treatment-by-ITC-Immunity-Therapy-Center",
        "https://www.placidway.com/price-list/5/Alternative-Cancer-Treatment-by-ITC-Immunity-Therapy-Center",
        "https://www.placidway.com/search-medical-pricings/Alternative-Medicine/All/1"
    ]
    
    CRAWLER_USER_AGENT: str = "PlacidWayAssessmentBot/1.0 (+https://placidway.com/bot; medical-assessment)"
    CRAWL_DELAY_SECONDS: float = 1.0
    
    # Grounding & Retrieval Thresholds
    SIMILARITY_THRESHOLD: float = 0.08
    TOP_K_CHUNKS: int = 5
    
    # Disclaimer text
    MEDICAL_DISCLAIMER: str = (
        "Medical Disclaimer: I provide information from PlacidWay pages for educational purposes only. "
        "I cannot provide medical advice, diagnosis, or clearance. Always consult a qualified physician or oncologist."
    )
    
    # Fallback response for ungrounded / out-of-scope questions
    FALLBACK_RESPONSE: str = (
        "I do not have this information in our current clinic and treatment records. "
        "Please request a free quote or contact PlacidWay directly, and our medical coordinators will gladly assist you."
    )

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        extra="allow"
    )

settings = Settings()

# Ensure required directories exist safely on serverless filesystems
try:
    settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
except (OSError, PermissionError):
    pass

try:
    settings.SCRAPED_PAGES_DIR.mkdir(parents=True, exist_ok=True)
except (OSError, PermissionError):
    pass
