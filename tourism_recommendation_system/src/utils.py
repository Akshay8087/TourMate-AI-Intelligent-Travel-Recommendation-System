"""
TourMate AI - Utility Functions
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

_cache = {}


def load_all_models(models_dir: str = None) -> dict:
    """Load all model artifacts from disk with caching."""
    global _cache
    if 'models' in _cache:
        return _cache['models']

    if models_dir is None:
        models_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'models'
        )

    models = {}
    if not os.path.exists(models_dir):
        return models

    for fname in os.listdir(models_dir):
        if fname.endswith('.pkl'):
            key = fname.replace('.pkl', '')
            try:
                models[key] = joblib.load(os.path.join(models_dir, fname))
            except Exception as e:
                print(f"Warning: Could not load {fname}: {e}")

    for fname in ['model_metrics.json', 'model_metadata.json', 'feature_names.json']:
        path = os.path.join(models_dir, fname)
        if os.path.exists(path):
            with open(path) as f:
                models[fname.replace('.json', '')] = json.load(f)

    _cache['models'] = models
    return models


def clear_cache():
    """Clear the models cache (forces reload)."""
    global _cache
    _cache = {}


def format_currency(amount: float, symbol: str = '¥') -> str:
    """Format a number as currency."""
    if pd.isna(amount):
        return 'N/A'
    return f"{symbol}{amount:,.0f}"


def format_rating(rating: float) -> str:
    """Format rating with stars."""
    if pd.isna(rating):
        return 'N/A'
    stars = '★' * int(rating) + '☆' * (5 - int(rating))
    return f"{rating:.1f} {stars}"


def get_budget_badge(budget_level: str) -> dict:
    """Return badge color and icon for budget level."""
    badges = {
        'Free': {'color': '#10b981', 'icon': '🆓', 'text': 'Free Entry'},
        'Budget': {'color': '#3b82f6', 'icon': '💰', 'text': 'Budget-Friendly'},
        'Mid-Range': {'color': '#f59e0b', 'icon': '💳', 'text': 'Mid-Range'},
        'Premium': {'color': '#8b5cf6', 'icon': '💎', 'text': 'Premium'},
        'Luxury': {'color': '#ef4444', 'icon': '👑', 'text': 'Luxury'},
    }
    return badges.get(str(budget_level), {'color': '#6b7280', 'icon': '💲', 'text': 'Varies'})


def get_season_icon(season: str) -> str:
    """Return emoji for season."""
    icons = {
        'Spring': '🌸',
        'Summer': '☀️',
        'Autumn': '🍂',
        'Winter': '❄️',
        'Chun Ji': '🌸',
    }
    return icons.get(str(season), '🌍')


def get_category_icon(category: str) -> str:
    """Return emoji for category."""
    icons = {
        'Natural Scenery': '🏔️',
        'Natural Culture': '🌿',
        'Natural Wonder': '🌊',
        'Historical Culture': '🏛️',
        'Historical Site': '🗿',
        'Historical Architecture': '🏯',
        'Ancient Town': '🏘️',
        'Religious Culture': '⛩️',
        'Revolutionary Site': '🚩',
        'Theme Park': '🎡',
        'Sports & Leisure': '🏄',
        'Botanical Garden': '🌺',
        'Zoo': '🦁',
        'Urban Landscape': '🌆',
        'Urban Landmark': '🗼',
        'Cultural Arts': '🎨',
        'Folk Culture': '🎭',
        'City Park': '🌳',
        'Museum': '🏛️',
        'Food Street': '🍜',
    }
    return icons.get(str(category), '📍')


def safe_json(obj):
    """Convert numpy/pandas types to JSON-serializable Python types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, pd.Series):
        return obj.tolist()
    if pd.isna(obj):
        return None
    return obj


def dataframe_to_json(df: pd.DataFrame) -> list:
    """Convert DataFrame to JSON-safe list of dicts."""
    records = []
    for _, row in df.iterrows():
        record = {}
        for col, val in row.items():
            record[col] = safe_json(val)
        records.append(record)
    return records


def compute_match_percentage(hybrid_score: float, max_score: float = 1.0) -> int:
    """Compute match percentage from hybrid score."""
    if max_score <= 0:
        return 0
    pct = min(100, int((hybrid_score / max_score) * 100))
    return max(1, pct)


def get_travel_tips(category: str, season: str) -> list:
    """Return travel tips based on category and season."""
    general_tips = [
        "Book accommodations in advance during peak season.",
        "Carry a portable power bank for long days of exploration.",
        "Download offline maps before your trip.",
        "Try local street food for authentic culinary experiences.",
        "Check if photography is allowed at the attraction.",
    ]

    category_tips = {
        'Religious Culture': [
            "Dress modestly when visiting religious sites.",
            "Remove shoes before entering temples.",
            "Be respectful of ongoing religious ceremonies.",
        ],
        'Natural Scenery': [
            "Start your visit early to avoid crowds.",
            "Wear comfortable, non-slip footwear for trails.",
            "Check weather forecasts before hiking.",
        ],
        'Ancient Town': [
            "Explore side alleys for hidden gems.",
            "Bargain politely at local markets.",
            "Visit in the morning for fewer tourists.",
        ],
        'Historical Culture': [
            "Hire a local guide for deeper cultural context.",
            "Allow 3-4 hours for a thorough visit.",
            "Take notes or photos for later research.",
        ],
    }

    season_tips = {
        'Summer': ["Stay hydrated and carry sunscreen.", "Visit indoor attractions during peak heat."],
        'Winter': ["Layer your clothing for warmth.", "Check if outdoor attractions are fully operational."],
        'Spring': ["Perfect for photography — carry your best camera.", "Book early as Spring is popular."],
        'Autumn': ["Ideal weather — plan for longer outdoor visits.", "Autumn foliage photography is spectacular."],
    }

    tips = category_tips.get(category, []) + season_tips.get(season, []) + general_tips
    return tips[:5]
