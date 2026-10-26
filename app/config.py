import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    PROJECT_NAME: str = "PlacidWay AI Knowledge Chatbot"
    VERSION: str = "1.0.0"
    ENVIRONMENT: str = "production"
    
    # OpenRouter API Configuration
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1/chat/completions"
    PRIMARY_MODEL: str = os.getenv("PRIMARY_MODEL", "recursive/flint-v1.0:free")
    FALLBACK_MODELS: list[str] = [
        "cohere/north-mini-code:free",
        "nvidia/nemotron-3.5-lightning:free"
    ]
    
    # Data Storage Paths
    DATA_DIR: Path = BASE_DIR / "data"
    SCRAPED_PAGES_DIR: Path = DATA_DIR / "scraped_pages"
    DB_PATH: Path = DATA_DIR / "placidway_bot.db"
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
    SIMILARITY_THRESHOLD: float = 0.20
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

# Ensure required directories exist
settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
settings.SCRAPED_PAGES_DIR.mkdir(parents=True, exist_ok=True)
