"""
TravelMate — Hybrid Information Retrieval Module
=================================================
Combines 4 distinct retrieval paradigms:
  1. Structured Filtering (budget, duration, state, crowd level, walking difficulty, season)
  2. Keyword / Preference Matching (Jaccard lexical overlap over tags and activities)
  3. TF-IDF Text Similarity
  4. Neural Semantic Similarity (Sentence-Transformers / Dense Embeddings)
Merges signals to return high-precision candidate pools for the Recommendation Engine.
"""

import os
from typing import List, Dict, Any, Optional, Set
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.preprocessing import clean_text


class HybridRetriever:
    """
    Executes multi-method retrieval across structured datasets and vector spaces.
    """

    def __init__(self, destinations_df: pd.DataFrame, semantic_engine):
        self.destinations_df = destinations_df.copy()
        self.semantic_engine = semantic_engine
        
        # Fit a dedicated TF-IDF vectorizer for Method 3
        self.tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        if not self.destinations_df.empty:
            corpus = self.destinations_df["search_text"].fillna("").tolist()
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(corpus)
        else:
            self.tfidf_matrix = None

    # Method 1: Structured Filtering
    def structured_filter(
        self,
        entities: Dict[str, Any],
        strict_budget: bool = False
    ) -> pd.DataFrame:
        """
        Filters destination DataFrame by hard and soft structured constraints:
        budget, duration, crowd density, physical exertion (walking), and season.
        """
        df = self.destinations_df.copy()
        if df.empty:
            return df

        # Filter by State / Region if mentioned
        if entities.get("destination") and entities["destination"] in df["state"].values:
            df = df[df["state"].str.lower() == entities["destination"].lower()]

        # Filter by Maximum Total Budget
        budget = entities.get("budget_inr")
        duration = entities.get("duration_days") or 2
        people = entities.get("people_count") or 1

        if budget and budget > 0:
            # Daily per-person budget threshold with 25% tolerance
            per_person_total = budget / max(people, 1)
            allowed_daily = (per_person_total / max(duration, 1)) * 1.25
            
            if strict_budget:
                df = df[df["estimated_cost_per_day"] <= allowed_daily]
            else:
                # Soft filter: prioritize within budget but keep close alternatives
                df["budget_diff"] = np.abs(df["estimated_cost_per_day"] - (per_person_total / max(duration, 1)))

        # Filter by Crowd Preference (e.g. 'Low' crowd / peaceful)
        crowd_pref = entities.get("crowd_preference")
        if crowd_pref and crowd_pref == "Low":
            # Prefer 'Low' and 'Moderate', demote 'High'
            df = df[df["crowd_level"].isin(["Low", "Moderate"])]

        # Filter by Walking / Accessibility (e.g. 'Low' walking for parents)
        walking_pref = entities.get("walking_preference")
        if walking_pref and walking_pref == "Low":
            df = df[df["walking_level"].isin(["Low", "Moderate"])]

        # Filter by Season if specified
        season = entities.get("season")
        if season:
            season_mask = df["best_season"].str.contains(season, case=False, na=False)
            if season_mask.sum() >= 2:
                df = df[season_mask]

        return df

    # Method 2: Keyword Matching
    def keyword_match_scores(self, preferences: List[str]) -> np.ndarray:
        """
        Computes lexical match score based on extracted preferences against
        destination categories, suitable_for tags, and famous_for descriptions.
        """
        n = len(self.destinations_df)
        if not preferences or n == 0:
            return np.zeros(n)

        scores = np.zeros(n)
        pref_set = set(p.lower() for p in preferences)

        for idx, row in self.destinations_df.iterrows():
            tags = str(row.get("suitable_for", "")).lower() + " " + str(row.get("category", "")).lower()
            famous = str(row.get("famous_for", "")).lower() + " " + str(row.get("activities", "")).lower()
            combined_tags = set(tags.replace(",", " ").split()) | set(famous.replace(",", " ").split())
            
            overlap = pref_set.intersection(combined_tags)
            # Weighted Jaccard-like score
            if pref_set:
                score = len(overlap) / len(pref_set)
                scores[idx] = min(score, 1.0)

        return scores

    # Method 3: TF-IDF Similarity
    def tfidf_similarity_scores(self, query: str) -> np.ndarray:
        """
        Computes cosine similarity between query TF-IDF vector and destination documents.
        """
        if self.tfidf_matrix is None or not query:
            return np.zeros(len(self.destinations_df))

        q_vec = self.tfidf_vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.tfidf_matrix)[0]
        return sims

    # Method 4: Semantic Embedding Similarity
    def semantic_similarity_scores(self, query: str) -> np.ndarray:
        """
        Uses dense sentence embeddings from the SemanticSearchEngine.
        """
        if self.semantic_engine is None or self.destinations_df.empty:
            return np.zeros(len(self.destinations_df))

        q_vec = self.semantic_engine.encode_query(query)
        if self.semantic_engine.destination_embeddings is not None:
            sims = cosine_similarity(q_vec, self.semantic_engine.destination_embeddings)[0]
            return sims
        return np.zeros(len(self.destinations_df))

    # Hybrid Candidate Pool Retrieval
    def hybrid_retrieve(
        self,
        query: str,
        entities: Dict[str, Any],
        preferences: List[str],
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Combines Structured Filtering + Keyword Matching + TF-IDF + Semantic Search.
        Returns candidate records populated with individual component scores.
        """
        n = len(self.destinations_df)
        if n == 0:
            return []

        # 1. Run Structured Filtering to identify qualified candidates
        filtered_df = self.structured_filter(entities, strict_budget=False)
        qualified_indices = set(filtered_df.index) if not filtered_df.empty else set(range(n))

        # 2. Compute individual similarity vector components
        sem_scores = self.semantic_similarity_scores(query)
        tfidf_scores = self.tfidf_similarity_scores(query)
        keyword_scores = self.keyword_match_scores(preferences)

        candidates = []
        for idx in range(n):
            row = self.destinations_df.iloc[idx].to_dict()
            dest_name = row["destination"]
            
            s_score = float(sem_scores[idx]) if idx < len(sem_scores) else 0.0
            t_score = float(tfidf_scores[idx]) if idx < len(tfidf_scores) else 0.0
            k_score = float(keyword_scores[idx]) if idx < len(keyword_scores) else 0.0
            is_structured_match = idx in qualified_indices

            # Normalized baseline blend
            combined_retrieval_score = (
                0.50 * s_score +
                0.25 * t_score +
                0.25 * k_score
            )
            # Boost if passes structured filters
            if is_structured_match:
                combined_retrieval_score *= 1.25

            # If user explicitly named this destination, grant maximum initial score
            if entities.get("destination") and dest_name.lower() == entities["destination"].lower():
                combined_retrieval_score += 2.0

            candidate_info = {
                "destination_data": row,
                "semantic_score": round(s_score, 4),
                "tfidf_score": round(t_score, 4),
                "keyword_score": round(k_score, 4),
                "structured_match": is_structured_match,
                "retrieval_score": round(combined_retrieval_score, 4)
            }
            candidates.append(candidate_info)

        # Sort candidates by combined retrieval score
        candidates.sort(key=lambda c: c["retrieval_score"], reverse=True)
        return candidates[:top_k]
