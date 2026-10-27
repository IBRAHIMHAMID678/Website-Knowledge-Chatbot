import logging
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse

from app.config import settings, BASE_DIR
from app.db.database import init_db
from app.services.vector_store import vector_store
from app.services.crawler import crawl_all_target_urls
from app.api.routes import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup sequence
    logger.info("Initializing PlacidWay Knowledge Chatbot system...")
    init_db()
    
    # Check if vector index exists on disk
    loaded = vector_store.load_index()
    if loaded:
        logger.info(f"Loaded existing index with {len(vector_store.chunks)} chunks.")
    else:
        logger.info("No existing vector index found. Launching initial ingestion of the 7 target URLs...")
        parsed_pages, all_chunks, errors = await crawl_all_target_urls()
        if all_chunks:
            vector_store.build_index(all_chunks)
            logger.info(f"Initial ingestion complete: {len(all_chunks)} chunks indexed.")
        if errors:
            logger.warning(f"Ingestion had {len(errors)} warnings: {errors}")
            
    yield
    # Shutdown sequence
    logger.info("PlacidWay Knowledge Chatbot shutting down gracefully.")

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Production-grade AI Knowledge Chatbot grounded strictly in PlacidWay medical pages.",
    lifespan=lifespan
)

# Cross-Origin Resource Sharing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router)

# Health probe for 7-day cloud deployment monitoring
@app.get("/healthz", tags=["Monitoring"])
async def health_check():
    return {
        "status": "healthy",
        "service": settings.PROJECT_NAME,
        "chunks_indexed": len(vector_store.chunks),
        "primary_model": settings.PRIMARY_MODEL
    }

# Locate static directory across local and serverless Vercel environments
def resolve_static_dir() -> Path:
    candidates = [
        BASE_DIR / "static",
        Path(__file__).resolve().parent.parent / "static",
        Path.cwd() / "static",
        Path("/var/task/static")
    ]
    for c in candidates:
        if c.exists() and (c / "index.html").exists():
            return c
    return BASE_DIR / "static"

static_dir = resolve_static_dir()
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

@app.get("/", include_in_schema=False)
async def serve_index():
    index_file = static_dir / "index.html"
    if index_file.exists():
        try:
            return HTMLResponse(content=index_file.read_text(encoding="utf-8"))
        except Exception:
            return FileResponse(str(index_file))
    return HTMLResponse("<h1>PlacidWay Knowledge Chatbot</h1><p>Application is live.</p>")
