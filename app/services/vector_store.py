import json
import logging
import re
from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from app.config import settings

logger = logging.getLogger(__name__)

class HybridVectorStore:
    """
    Lightweight, high-performance in-memory and file-persistent vector store
    combining dense semantic representations and sparse keyword matching (TF-IDF/BM25).
    Guarantees fast, offline-capable, zero-maintenance retrieval.
    """
    def __init__(self):
        self.chunks: List[Dict[str, Any]] = []
        self.vectorizer: TfidfVectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            stop_words="english",
            max_features=5000,
            sublinear_tf=True
        )
        self.tfidf_matrix = None
        self.dense_model = None
        self.dense_embeddings = None
        self._init_dense_model()
        self.load_index()

    def _init_dense_model(self):
        import os
        # Only enable dense downloads if explicitly requested
        if os.getenv("ENABLE_DENSE_EMBEDDINGS", "false").lower() == "true":
            try:
                from sentence_transformers import SentenceTransformer
                self.dense_model = SentenceTransformer("all-MiniLM-L6-v2")
                logger.info("SentenceTransformer (all-MiniLM-L6-v2) initialized.")
            except Exception as e:
                logger.warning(f"Could not load SentenceTransformer, running in TF-IDF sparse mode: {e}")
                self.dense_model = None
        else:
            logger.info("Running in fast, zero-dependency TF-IDF/BM25 retrieval mode.")
            self.dense_model = None

    def build_index(self, chunks: List[Dict[str, Any]]):
        """Indexes all chunks into both sparse TF-IDF and dense vector matrices."""
        if not chunks:
            logger.warning("No chunks provided to index.")
            return

        self.chunks = chunks
        corpus = [c["text"] for c in chunks]

        # 1. Fit sparse TF-IDF
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)

        # 2. Compute dense embeddings if model is active
        if self.dense_model:
            try:
                self.dense_embeddings = self.dense_model.encode(corpus, convert_to_numpy=True, normalize_embeddings=True)
            except Exception as e:
                logger.error(f"Error computing dense embeddings: {e}")
                self.dense_embeddings = None

        # 3. Save to disk for persistence across restarts
        self.save_index()
        logger.info(f"Vector store successfully indexed {len(self.chunks)} chunks.")

    def save_index(self):
        """Saves chunks and metadata to JSON file."""
        data = {
            "total_chunks": len(self.chunks),
            "chunks": self.chunks
        }
        with open(settings.VECTOR_INDEX_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_index(self) -> bool:
        """Loads chunks from persistent disk file if available."""
        if not settings.VECTOR_INDEX_PATH.exists():
            return False
        try:
            with open(settings.VECTOR_INDEX_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            chunks = data.get("chunks", [])
            if chunks:
                self.build_index(chunks)
                return True
        except Exception as e:
            logger.error(f"Failed to load vector index: {e}")
        return False

    def search(self, query: str, top_k: int = 4) -> List[Tuple[Dict[str, Any], float]]:
        """
        Hybrid search combining semantic similarity and keyword relevance.
        Returns sorted list of (chunk, composite_score).
        """
        if not self.chunks or self.tfidf_matrix is None:
            return []

        # 1. Sparse TF-IDF score
        q_sparse = self.vectorizer.transform([query])
        sparse_scores = cosine_similarity(q_sparse, self.tfidf_matrix)[0]

        # 2. Dense semantic score
        if self.dense_model and self.dense_embeddings is not None:
            try:
                q_dense = self.dense_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
                dense_scores = np.dot(self.dense_embeddings, q_dense[0])
                # Weighted hybrid score (60% dense + 40% sparse)
                composite_scores = 0.6 * dense_scores + 0.4 * sparse_scores
            except Exception as e:
                logger.warning(f"Dense scoring failed, falling back to sparse: {e}")
                composite_scores = sparse_scores
        else:
            composite_scores = sparse_scores

        # 3. Principled Entity & Price Intent Booster
        COMMON_TERMS = {
            "what", "when", "where", "which", "who", "whom", "this", "that", "these", "those",
            "have", "from", "with", "about", "into", "through", "during", "before", "after",
            "cost", "costs", "price", "prices", "much", "many", "does", "treat", "treatment",
            "treatments", "clinic", "clinics", "medical", "hospital", "mexico", "tijuana",
            "alternative", "cancer", "cancers", "they", "your", "tell", "show", "give", "help"
        }
        distinctive_terms = [
            w for w in re.findall(r"\b[a-zA-Z]{4,}\b", query.lower()) 
            if w not in COMMON_TERMS
        ]
        
        has_corpus_match = True
        if distinctive_terms:
            total_matches_across_corpus = 0
            for idx, chunk in enumerate(self.chunks):
                chunk_lower = chunk["text"].lower()
                matches = sum(1 for term in distinctive_terms if term in chunk_lower)
                if matches > 0:
                    composite_scores[idx] += 0.08 * min(matches, 3)
                    total_matches_across_corpus += matches
                    
            if total_matches_across_corpus == 0:
                has_corpus_match = False

        # Price / Cost Intent Booster
        # If user asks how much something costs and the query is in-domain, prioritize chunks with actual prices
        is_cost_query = bool(re.search(r"\b(cost|costs|price|prices|how much|usd|\$|pricing|expensive|cheaper|quote)\b", query, re.I))
        if is_cost_query and has_corpus_match:
            for idx, chunk in enumerate(self.chunks):
                chunk_text = chunk["text"]
                if re.search(r"(\$\d[\d,]+|\bUSD\b|### Cost|### Package Price|Total Cost)", chunk_text):
                    composite_scores[idx] += 0.15

        # If distinctive terms had zero matches in the entire corpus, penalize top scores to ensure honest fallback
        if not has_corpus_match:
            composite_scores = composite_scores * 0.5

        # Get top-k indices
        top_indices = np.argsort(composite_scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            score = float(composite_scores[idx])
            results.append((self.chunks[idx], score))

        return results

# Global singleton
vector_store = HybridVectorStore()
