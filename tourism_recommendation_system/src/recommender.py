"""
TourMate AI - Core Recommendation Engine
Implements popularity-based, content-based, and hybrid recommendation systems.
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import joblib
import json
import os
import warnings
warnings.filterwarnings('ignore')


class PopularityRecommender:
    """Popularity-based recommendation baseline."""

    def __init__(self):
        self.catalog = None

    def fit(self, catalog: pd.DataFrame):
        self.catalog = catalog.copy()
        return self

    def recommend(self, category: str = None, province: str = None,
                  season: str = None, n: int = 10) -> pd.DataFrame:
        df = self.catalog.copy()
        if category and category != 'All':
            df = df[df['category'].str.lower() == category.lower()]
        if province and province != 'All':
            df = df[df['province'].str.lower() == province.lower()]
        if season and season != 'All':
            df = df[df['best_season'].str.lower() == season.lower()]
        if df.empty:
            df = self.catalog.copy()
        return df.nlargest(n, 'popularity_score').reset_index(drop=True)


class ContentBasedRecommender:
    """TF-IDF + Cosine Similarity content-based recommender."""

    def __init__(self):
        self.tfidf = None
        self.tfidf_matrix = None
        self.catalog = None
        self.nn_model = None

    def fit(self, catalog: pd.DataFrame, tfidf_vectorizer=None):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.catalog = catalog.copy().reset_index(drop=True)

        if tfidf_vectorizer is None:
            self.tfidf = TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.95
            )
        else:
            self.tfidf = tfidf_vectorizer

        text_corpus = self.catalog['combined_features'].fillna('')
        self.tfidf_matrix = self.tfidf.fit_transform(text_corpus)

        self.nn_model = NearestNeighbors(
            n_neighbors=min(20, len(self.catalog)),
            metric='cosine',
            algorithm='brute'
        )
        self.nn_model.fit(self.tfidf_matrix)
        return self

    def get_similar(self, attraction_name: str, n: int = 10) -> pd.DataFrame:
        """Find similar destinations to a given one."""
        matches = self.catalog[
            self.catalog['attraction_name'].str.lower() == attraction_name.lower()
        ]
        if matches.empty:
            # Partial match
            matches = self.catalog[
                self.catalog['attraction_name'].str.lower().str.contains(
                    attraction_name.lower(), na=False
                )
            ]
        if matches.empty:
            return pd.DataFrame()

        idx = matches.index[0]
        distances, indices = self.nn_model.kneighbors(
            self.tfidf_matrix[idx], n_neighbors=min(n + 1, len(self.catalog))
        )
        indices = indices[0][1:]  # Exclude self
        scores = 1 - distances[0][1:]

        result = self.catalog.iloc[indices].copy()
        result['similarity_score'] = scores
        return result.reset_index(drop=True)

    def search_by_query(self, query: str, n: int = 10) -> pd.DataFrame:
        """Search attractions by free-text query."""
        query_vec = self.tfidf.transform([query])
        distances, indices = self.nn_model.kneighbors(
            query_vec, n_neighbors=min(n, len(self.catalog))
        )
        result = self.catalog.iloc[indices[0]].copy()
        result['similarity_score'] = 1 - distances[0]
        return result.reset_index(drop=True)


class HybridRecommender:
    """
    Hybrid Recommendation System combining:
    - Content similarity (TF-IDF cosine)
    - Popularity score
    - Rating score
    - Budget compatibility
    - Season match
    - Category match
    - Province/location match
    """

    def __init__(self, weights: dict = None):
        self.default_weights = {
            'similarity': 0.30,
            'popularity': 0.25,
            'rating': 0.20,
            'budget': 0.10,
            'season': 0.10,
            'category': 0.05,
        }
        self.weights = weights or self.default_weights
        self.content_rec = None
        self.catalog = None

    def fit(self, catalog: pd.DataFrame, content_recommender: ContentBasedRecommender):
        self.catalog = catalog.copy().reset_index(drop=True)
        self.content_rec = content_recommender
        return self

    def _build_query_string(self, category: str = None, province: str = None,
                             season: str = None, budget_level: str = None,
                             activity: str = None) -> str:
        parts = []
        if category:
            parts.append(category)
        if province:
            parts.append(province)
        if season:
            parts.append(season)
        if budget_level:
            parts.append(budget_level)
        if activity:
            parts.append(activity)
        return ' '.join(parts) if parts else 'Natural Scenery China'

    def recommend(self, category: str = None, province: str = None,
                  season: str = None, budget_level: str = None,
                  min_rating: float = 3.0, n: int = 10,
                  activity: str = None, attraction_name: str = None,
                  age_group: str = None) -> pd.DataFrame:
        """
        Main hybrid recommendation function.
        """
        df = self.catalog.copy()

        # Get similarity scores via content-based
        if attraction_name and attraction_name.strip():
            similar = self.content_rec.get_similar(attraction_name, n=len(df))
        else:
            query = self._build_query_string(category, province, season, budget_level, activity)
            similar = self.content_rec.search_by_query(query, n=len(df))

        if not similar.empty:
            sim_lookup = dict(zip(
                similar['attraction_name'],
                similar.get('similarity_score', pd.Series([0.5] * len(similar)))
            ))
        else:
            sim_lookup = {}

        df['sim_score'] = df['attraction_name'].map(sim_lookup).fillna(0.3)

        # Normalize rating
        r_min, r_max = df['avg_rating'].min(), df['avg_rating'].max()
        df['rating_norm'] = (df['avg_rating'] - r_min) / (r_max - r_min + 1e-9)

        # Season match
        if season and season != 'All':
            df['season_match'] = (df['best_season'].str.lower() == season.lower()).astype(float)
        else:
            df['season_match'] = 0.5

        # Category match
        if category and category != 'All':
            df['cat_match'] = (df['category'].str.lower() == category.lower()).astype(float)
        else:
            df['cat_match'] = 0.5

        # Budget compatibility
        if budget_level and budget_level != 'All':
            df['budget_match'] = (df['budget_level'].astype(str).str.lower() ==
                                   budget_level.lower()).astype(float)
        else:
            df['budget_match'] = 0.5

        # Rating filter
        df = df[df['avg_rating'] >= min_rating]
        if df.empty:
            df = self.catalog.copy()
            df['sim_score'] = 0.3
            df['rating_norm'] = 0.5
            df['season_match'] = 0.5
            df['cat_match'] = 0.5
            df['budget_match'] = 0.5

        # Compute hybrid score
        w = self.weights
        df['hybrid_score'] = (
            w['similarity'] * df['sim_score'] +
            w['popularity'] * df['popularity_score'] +
            w['rating'] * df['rating_norm'] +
            w['budget'] * df['budget_match'] +
            w['season'] * df['season_match'] +
            w['category'] * df['cat_match']
        )

        # Add reasons
        df['match_reasons'] = df.apply(
            lambda r: self._generate_reason(r, category, season, budget_level, province), axis=1
        )

        result = df.nlargest(n, 'hybrid_score').reset_index(drop=True)
        result['match_pct'] = (result['hybrid_score'] / result['hybrid_score'].max() * 100).round(1)
        return result

    def _generate_reason(self, row, category, season, budget_level, province):
        reasons = []
        if category and str(row.get('category', '')).lower() == category.lower():
            reasons.append(f"Matches your {category} preference")
        if season and str(row.get('best_season', '')).lower() == season.lower():
            reasons.append(f"Best visited in {season}")
        if budget_level and str(row.get('budget_level', '')).lower() == budget_level.lower():
            reasons.append(f"Fits your {budget_level} budget")
        if row.get('avg_rating', 0) >= 4.5:
            reasons.append("Highly rated by visitors")
        if row.get('popularity_score', 0) >= 0.7:
            reasons.append("Very popular destination")
        if row.get('attraction_level', '') == '5A':
            reasons.append("Top-tier 5A rated attraction")
        if not reasons:
            reasons.append("Recommended by our AI model")
        return '; '.join(reasons)


class ActivityRecommender:
    """Recommend destinations based on activity search."""

    def __init__(self):
        self.catalog = None

    def fit(self, catalog: pd.DataFrame):
        self.catalog = catalog.copy()
        return self

    def recommend_by_activity(self, activity_keyword: str, n: int = 10) -> pd.DataFrame:
        """Find destinations matching activity keywords."""
        kw = activity_keyword.lower()
        df = self.catalog.copy()

        # Score based on category match and name match
        df['activity_score'] = 0.0
        df.loc[df['category'].str.lower().str.contains(kw, na=False), 'activity_score'] += 0.6
        df.loc[df['attraction_name'].str.lower().str.contains(kw, na=False), 'activity_score'] += 0.4

        matched = df[df['activity_score'] > 0]
        if matched.empty:
            matched = df.nlargest(n, 'popularity_score')

        return matched.nlargest(n, ['activity_score', 'avg_rating']).reset_index(drop=True)


def save_models(models_dict: dict, models_dir: str = 'models'):
    """Save all model artifacts."""
    os.makedirs(models_dir, exist_ok=True)
    for name, obj in models_dict.items():
        path = os.path.join(models_dir, f'{name}.pkl')
        joblib.dump(obj, path)
    print(f"Saved {len(models_dict)} model files to {models_dir}/")


def load_models(models_dir: str = 'models') -> dict:
    """Load all model artifacts."""
    models = {}
    pkl_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
    for fname in pkl_files:
        key = fname.replace('.pkl', '')
        models[key] = joblib.load(os.path.join(models_dir, fname))
    return models
