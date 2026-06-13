"""
TourMate AI - Data Preprocessing Module
Handles all data cleaning, transformation, and feature engineering.
"""

import pandas as pd
import numpy as np
import re
import warnings
warnings.filterwarnings('ignore')


def load_dataset(filepath: str) -> pd.DataFrame:
    """Load the tourism dataset with proper encoding handling."""
    try:
        df = pd.read_csv(filepath, encoding='utf-8-sig')
    except UnicodeDecodeError:
        df = pd.read_csv(filepath, encoding='latin-1')
    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Full cleaning pipeline for the tourism dataset."""
    df = df.copy()

    # Standardize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

    # Fix season column - standardize Chun Ji to Spring
    if 'season' in df.columns:
        df['season'] = df['season'].replace({'Chun Ji': 'Spring', 'chun ji': 'Spring'})

    # Fix attraction_category - map Chinese names to English
    category_map = {
        'Zi Ran Qi Guan': 'Natural Wonder',
        'Wen Hua Yi Shu': 'Cultural Arts',
        'Cheng Shi Gong Yuan': 'City Park',
        'Min Su Wen Hua': 'Folk Culture',
        'Wen Bo Yuan Guan': 'Museum',
    }
    if 'attraction_category' in df.columns:
        df['attraction_category'] = df['attraction_category'].replace(category_map)

    # Parse visit_date
    if 'visit_date' in df.columns:
        df['visit_date'] = pd.to_datetime(df['visit_date'], errors='coerce')
        df['visit_month'] = df['visit_date'].dt.month
        df['visit_year'] = df['visit_date'].dt.year

    # Create budget levels
    if 'ticket_price' in df.columns:
        df['budget_level'] = pd.cut(
            df['ticket_price'],
            bins=[-1, 0, 50, 100, 200, 999],
            labels=['Free', 'Budget', 'Mid-Range', 'Premium', 'Luxury']
        )

    # Create spend level
    if 'spend_amount' in df.columns:
        df['spend_level'] = pd.cut(
            df['spend_amount'],
            bins=[-1, 50, 200, 500, 1000, 99999],
            labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
        )

    # Fill group tour missing
    if 'is_group_tour' in df.columns:
        df['is_group_tour'] = df['is_group_tour'].fillna('No')

    # Fill numeric nulls with median
    for col in ['group_fee', 'trip_days']:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Encode binary columns
    if 'is_holiday' in df.columns:
        df['is_holiday_enc'] = (df['is_holiday'] == 'Yes').astype(int)
    if 'is_group_tour' in df.columns:
        df['is_group_tour_enc'] = (df['is_group_tour'] == 'Yes').astype(int)

    # Create popularity score per attraction
    if 'attraction_name' in df.columns and 'rating' in df.columns:
        popularity = df.groupby('attraction_name').agg(
            visit_count=('tourist_id', 'count'),
            avg_rating=('rating', 'mean'),
            avg_spend=('spend_amount', 'mean')
        ).reset_index()
        popularity['popularity_score'] = (
            0.5 * (popularity['visit_count'] / popularity['visit_count'].max()) +
            0.3 * ((popularity['avg_rating'] - popularity['avg_rating'].min()) /
                   (popularity['avg_rating'].max() - popularity['avg_rating'].min() + 1e-9)) +
            0.2 * ((popularity['avg_spend'] - popularity['avg_spend'].min()) /
                   (popularity['avg_spend'].max() - popularity['avg_spend'].min() + 1e-9))
        )
        df = df.merge(popularity[['attraction_name', 'visit_count', 'avg_rating',
                                   'avg_spend', 'popularity_score']],
                      on='attraction_name', how='left')

    return df


def build_attraction_catalog(df: pd.DataFrame) -> pd.DataFrame:
    """Build aggregated catalog of unique attractions."""
    agg = df.groupby('attraction_name').agg(
        category=('attraction_category', lambda x: x.mode()[0]),
        province=('province', lambda x: x.mode()[0]),
        city=('city', lambda x: x.mode()[0]),
        attraction_level=('attraction_level', lambda x: x.mode()[0]),
        avg_rating=('rating', 'mean'),
        avg_ticket_price=('ticket_price', 'mean'),
        avg_spend=('spend_amount', 'mean'),
        visit_count=('tourist_id', 'count'),
        best_season=('season', lambda x: x.mode()[0]),
        avg_duration=('visit_duration_hours', 'mean'),
        highly_recommended_pct=('recommendation_level',
                                 lambda x: (x == 'Highly Recommend').mean()),
        very_satisfied_pct=('satisfaction_level',
                             lambda x: (x == 'Very Satisfied').mean()),
    ).reset_index()

    agg['popularity_score'] = (
        0.4 * (agg['visit_count'] / agg['visit_count'].max()) +
        0.3 * ((agg['avg_rating'] - agg['avg_rating'].min()) /
               (agg['avg_rating'].max() - agg['avg_rating'].min() + 1e-9)) +
        0.2 * agg['highly_recommended_pct'] +
        0.1 * agg['very_satisfied_pct']
    )

    agg['budget_level'] = pd.cut(
        agg['avg_ticket_price'],
        bins=[-1, 0, 50, 100, 200, 999],
        labels=['Free', 'Budget', 'Mid-Range', 'Premium', 'Luxury']
    )

    # Add combined text feature for TF-IDF
    agg['combined_features'] = (
        agg['attraction_name'] + ' ' +
        agg['category'] + ' ' +
        agg['province'] + ' ' +
        agg['city'] + ' ' +
        agg['attraction_level'] + ' ' +
        agg['best_season'] + ' ' +
        agg['budget_level'].astype(str)
    )

    return agg


def encode_features(df: pd.DataFrame, label_col: str = 'attraction_category'):
    """Label-encode categorical target columns."""
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    df[label_col + '_enc'] = le.fit_transform(df[label_col].astype(str))
    return df, le


def get_feature_matrix(catalog: pd.DataFrame):
    """Build numerical feature matrix from catalog."""
    from sklearn.preprocessing import StandardScaler, LabelEncoder

    features = catalog.copy()

    # Encode categoricals
    for col in ['category', 'province', 'city', 'attraction_level', 'best_season', 'budget_level']:
        if col in features.columns:
            le = LabelEncoder()
            features[col + '_enc'] = le.fit_transform(features[col].astype(str))

    num_cols = ['avg_rating', 'avg_ticket_price', 'avg_spend', 'avg_duration',
                'popularity_score', 'highly_recommended_pct', 'very_satisfied_pct',
                'visit_count', 'category_enc', 'province_enc', 'city_enc',
                'attraction_level_enc', 'best_season_enc']

    num_cols = [c for c in num_cols if c in features.columns]
    X = features[num_cols].fillna(0).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, scaler, num_cols
