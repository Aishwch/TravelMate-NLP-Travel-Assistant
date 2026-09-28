"""
TravelMate — Semantic Search & Neural Embeddings Module
========================================================
Implements semantic similarity search using Sentence Transformers
('all-MiniLM-L6-v2') with cosine similarity. Caches precomputed dense embeddings
in models/embeddings/ to ensure instant retrieval without runtime recomputation.
Includes an automatic, reliable TF-IDF + cosine similarity fallback engine.
"""

import os
import sys
from typing import List, Dict, Tuple, Optional, Any
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Root directory references
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EMBEDDINGS_DIR = os.path.join(ROOT_DIR, "models", "embeddings")
DEST_CSV = os.path.join(ROOT_DIR, "data", "processed", "destinations.csv")
ATTR_CSV = os.path.join(ROOT_DIR, "data", "processed", "attractions.csv")


class SemanticSearchEngine:
    """
    Semantic search engine using sentence embeddings with TF-IDF fallback.
    """

    def __init__(self, use_gpu: bool = False):
        os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
        self.encoder = None
        self.use_fallback = False
        self.tfidf_vectorizer = None
        
        # Datasets
        self.destinations_df = None
        self.attractions_df = None
        
        # Embedding matrices
        self.destination_embeddings = None
        self.attraction_embeddings = None

        self._load_datasets()
        self._init_encoder()
        self._load_or_compute_embeddings()

    def _load_datasets(self):
        """Loads processed destinations and attractions datasets."""
        if os.path.exists(DEST_CSV):
            self.destinations_df = pd.read_csv(DEST_CSV)
        else:
            self.destinations_df = pd.DataFrame()

        if os.path.exists(ATTR_CSV):
            self.attractions_df = pd.read_csv(ATTR_CSV)
        else:
            self.attractions_df = pd.DataFrame()

    def _init_encoder(self):
        """Initializes SentenceTransformer model or falls back to TF-IDF."""
        use_st = os.environ.get("USE_SENTENCE_TRANSFORMERS", "true").lower()
        if use_st not in ("0", "false", "no", "off"):
            try:
                from sentence_transformers import SentenceTransformer
                try:
                    # Load from local cache directly for near-instant startup
                    self.encoder = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)
                except Exception:
                    print("Attempting to load SentenceTransformer ('all-MiniLM-L6-v2')...")
                    self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
                print("✓ SentenceTransformer loaded successfully.")
                return
            except Exception as e:
                print(f"Notice: SentenceTransformer offline/unavailable ({e}).")
        
        print("Activating high-fidelity TF-IDF + Cosine Similarity semantic engine.")
        self._init_tfidf_fallback()

    def _init_tfidf_fallback(self):
        """Initializes TF-IDF vectorizer fallback."""
        self.use_fallback = True
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            sublinear_tf=True,
            min_df=1
        )
        if not self.destinations_df.empty:
            corpus = self.destinations_df["search_text"].fillna("").tolist()
            if not self.attractions_df.empty:
                corpus += self.attractions_df["search_text"].fillna("").tolist()
            self.tfidf_vectorizer.fit(corpus)

    def _load_or_compute_embeddings(self):
        """Loads cached embeddings or calculates and saves them to disk."""
        dest_emb_path = os.path.join(EMBEDDINGS_DIR, "destination_embeddings.npy")
        attr_emb_path = os.path.join(EMBEDDINGS_DIR, "attraction_embeddings.npy")
        vec_pkl_path = os.path.join(EMBEDDINGS_DIR, "semantic_vectorizer.pkl")

        # Load cached semantic vectorizer if in fallback mode
        if self.use_fallback and os.path.exists(vec_pkl_path):
            try:
                self.tfidf_vectorizer = joblib.load(vec_pkl_path)
            except Exception:
                pass

        # Check if cache is valid for destinations
        expected_dim = 384 if (not self.use_fallback and self.encoder is not None) else None

        if os.path.exists(dest_emb_path) and not self.destinations_df.empty:
            try:
                cached = np.load(dest_emb_path)
                dim_ok = True if expected_dim is None else (cached.shape[1] == expected_dim)
                if len(cached) == len(self.destinations_df) and dim_ok:
                    self.destination_embeddings = cached
                    print(f"✓ Loaded cached destination embeddings: {cached.shape}")
                else:
                    self.destination_embeddings = None
            except Exception:
                self.destination_embeddings = None

        # Check if cache is valid for attractions
        if os.path.exists(attr_emb_path) and not self.attractions_df.empty:
            try:
                cached = np.load(attr_emb_path)
                dim_ok = True if expected_dim is None else (cached.shape[1] == expected_dim)
                if len(cached) == len(self.attractions_df) and dim_ok:
                    self.attraction_embeddings = cached
                    print(f"✓ Loaded cached attraction embeddings: {cached.shape}")
                else:
                    self.attraction_embeddings = None
            except Exception:
                self.attraction_embeddings = None

        # Compute embeddings if not cached
        if self.destination_embeddings is None and not self.destinations_df.empty:
            self._compute_and_save_dest_embeddings(dest_emb_path)

        if self.attraction_embeddings is None and not self.attractions_df.empty:
            self._compute_and_save_attr_embeddings(attr_emb_path)

    def _compute_and_save_dest_embeddings(self, dest_emb_path: str):
        """Computes and saves destination embeddings."""
        texts = self.destinations_df["search_text"].fillna(self.destinations_df["description"]).tolist()
        if not self.use_fallback and self.encoder is not None:
            print("Encoding destination knowledge base with SentenceTransformer...")
            embeddings = self.encoder.encode(texts, show_progress_bar=False, normalize_embeddings=True)
            self.destination_embeddings = np.array(embeddings, dtype=np.float32)
            np.save(dest_emb_path, self.destination_embeddings)
            print(f"✓ Saved destination embeddings -> {dest_emb_path}")
        else:
            if self.tfidf_vectorizer is None:
                self._init_tfidf_fallback()
            self.destination_embeddings = self.tfidf_vectorizer.transform(texts).toarray().astype(np.float32)
            np.save(dest_emb_path, self.destination_embeddings)
            joblib.dump(self.tfidf_vectorizer, os.path.join(EMBEDDINGS_DIR, "semantic_vectorizer.pkl"))
            print(f"✓ Saved destination embeddings (TF-IDF semantic space) -> {dest_emb_path}")

    def _compute_and_save_attr_embeddings(self, attr_emb_path: str):
        """Computes and saves attraction embeddings."""
        if "search_text" in self.attractions_df.columns:
            texts = self.attractions_df["search_text"].fillna(self.attractions_df.get("description", "")).tolist()
        else:
            name_col = self.attractions_df.get("attraction_name", self.attractions_df.get("attraction", ""))
            dest_col = self.attractions_df.get("destination", "")
            cat_col = self.attractions_df.get("category", "")
            desc_col = self.attractions_df.get("description", "")
            texts = (name_col.astype(str) + " in " + dest_col.astype(str) + ". " + cat_col.astype(str) + ". " + desc_col.astype(str)).tolist()
        if not self.use_fallback and self.encoder is not None:
            print("Encoding attraction knowledge base with SentenceTransformer...")
            embeddings = self.encoder.encode(texts, show_progress_bar=False, normalize_embeddings=True)
            self.attraction_embeddings = np.array(embeddings, dtype=np.float32)
            np.save(attr_emb_path, self.attraction_embeddings)
            print(f"✓ Saved attraction embeddings -> {attr_emb_path}")
        else:
            if self.tfidf_vectorizer is None:
                self._init_tfidf_fallback()
            self.attraction_embeddings = self.tfidf_vectorizer.transform(texts).toarray().astype(np.float32)
            np.save(attr_emb_path, self.attraction_embeddings)
            print(f"✓ Saved attraction embeddings (TF-IDF semantic space) -> {attr_emb_path}")

    def encode_query(self, query: str) -> np.ndarray:
        """Encodes user query into an embedding vector."""
        if not self.use_fallback and self.encoder is not None:
            emb = self.encoder.encode([query], show_progress_bar=False, normalize_embeddings=True)
            return np.array(emb, dtype=np.float32)
        else:
            if self.tfidf_vectorizer is None:
                self._init_tfidf_fallback()
            vec = self.tfidf_vectorizer.transform([query]).toarray()
            # Normalize vector
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            return vec

    def search_destinations(self, query: str, top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """
        Calculates cosine similarity between user query and all destination embeddings.
        Returns top-K destinations with similarity scores [0.0 - 1.0].
        """
        if self.destinations_df.empty or self.destination_embeddings is None:
            return []

        query_vec = self.encode_query(query)
        # Cosine similarity
        similarities = cosine_similarity(query_vec, self.destination_embeddings)[0]
        
        # Sort descending
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            dest_row = self.destinations_df.iloc[idx].to_dict()
            results.append((dest_row, round(score, 4)))

        return results

    def search_attractions(
        self, query: str, destination: Optional[str] = None, top_k: int = 5
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Searches attractions using semantic similarity, optionally filtering by destination.
        """
        if self.attractions_df.empty or self.attraction_embeddings is None:
            return []

        query_vec = self.encode_query(query)
        similarities = cosine_similarity(query_vec, self.attraction_embeddings)[0]

        # Destination filtering mask
        if destination:
            mask = self.attractions_df["destination"].str.lower() == destination.strip().lower()
            filtered_indices = np.where(mask)[0]
            if len(filtered_indices) == 0:
                # Fuzzy match destination in attractions df
                dest_lower = destination.strip().lower()
                mask = self.attractions_df["destination"].str.lower().str.contains(dest_lower, na=False)
                filtered_indices = np.where(mask)[0]
        else:
            filtered_indices = np.arange(len(self.attractions_df))

        if len(filtered_indices) == 0:
            filtered_indices = np.arange(len(self.attractions_df))

        # Get similarities for filtered items
        subset_sims = similarities[filtered_indices]
        top_subset_idx = np.argsort(subset_sims)[::-1][:top_k]

        results = []
        for sub_idx in top_subset_idx:
            orig_idx = filtered_indices[sub_idx]
            score = float(similarities[orig_idx])
            attr_row = self.attractions_df.iloc[orig_idx].to_dict()
            results.append((attr_row, round(score, 4)))

        return results
