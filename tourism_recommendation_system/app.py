"""
TourMate AI - Flask Web Application
Intelligent Tourism Recommendation System
"""

import os
import sys
import json
import warnings
warnings.filterwarnings('ignore')

from flask import Flask, render_template, request, jsonify, redirect, url_for
from dotenv import load_dotenv
import pandas as pd
import numpy as np

# Load environment variables
load_dotenv()

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.utils import (
    load_all_models, dataframe_to_json, safe_json,
    get_budget_badge, get_season_icon, get_category_icon, get_travel_tips
)
from src.data_preprocessing import load_dataset, clean_dataset, build_attraction_catalog
from src.eda_utils import (
    get_top_attractions_chart, get_category_distribution_chart,
    get_rating_distribution_chart, get_budget_distribution_chart,
    get_season_demand_chart, get_province_chart, get_rating_by_category_chart,
    get_age_group_chart, get_gender_chart, get_satisfaction_chart,
    get_spend_vs_rating_chart, get_summary_stats
)
from src.gemini_helper import generate_travel_insight

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'tourmate-ai-secret-2024')

# ─── Load Data & Models Once ────────────────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, 'data', 'tourism_recommendation_dataset_en.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

print("Loading dataset...")
_df_raw = load_dataset(DATA_PATH)
_df = clean_dataset(_df_raw)

print("Loading models...")
_models = load_all_models(MODELS_DIR)
_catalog = _models.get('attraction_catalog')
if _catalog is None:
    _catalog = _models.get('catalog')
_metadata = _models.get('model_metadata') or {}
_metrics = _models.get('model_metrics') or {}

print(f"✓ App ready — {len(_catalog)} attractions, {len(_df):,} records")

# ─── Helper: Get unique filter options ──────────────────────────────────────
def get_filter_options():
    categories = sorted(_df['attraction_category'].unique().tolist())
    provinces = sorted(_df['province'].unique().tolist())
    seasons = ['Spring', 'Summer', 'Autumn', 'Winter']
    budget_levels = ['Free', 'Budget', 'Mid-Range', 'Premium', 'Luxury']
    age_groups = ['18-25', '26-35', '36-45', '46-55', '56+']
    genders = ['Male', 'Female']
    attraction_levels = ['3A', '4A', '5A']
    return dict(
        categories=categories,
        provinces=provinces,
        seasons=seasons,
        budget_levels=budget_levels,
        age_groups=age_groups,
        genders=genders,
        attraction_levels=attraction_levels,
    )

# ─── Routes ─────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    stats = get_summary_stats(_df)
    return render_template('index.html', stats=stats, metadata=_metadata)


@app.route('/recommend', methods=['GET', 'POST'])
def recommend():
    opts = get_filter_options()

    if request.method == 'POST':
        data = request.form
        category = data.get('category', '')
        province = data.get('province', '')
        season = data.get('season', '')
        budget_level = data.get('budget_level', '')
        min_rating = float(data.get('min_rating', 3.5))
        attraction_name = data.get('attraction_name', '')
        n = int(data.get('n', 10))

        hybrid_rec = _models.get('hybrid_recommender')
        if hybrid_rec is None:
            return render_template('recommend.html', **opts,
                                   error="Model not loaded. Please run training first.")

        try:
            results = hybrid_rec.recommend(
                category=category or None,
                province=province or None,
                season=season or None,
                budget_level=budget_level or None,
                min_rating=min_rating,
                n=n,
                attraction_name=attraction_name or None,
            )

            recs = dataframe_to_json(results)

            # Enrich each recommendation
            for rec in recs:
                rec['budget_badge'] = get_budget_badge(rec.get('budget_level'))
                rec['season_icon'] = get_season_icon(rec.get('best_season'))
                rec['category_icon'] = get_category_icon(rec.get('category'))
                rec['travel_tips'] = get_travel_tips(
                    rec.get('category', ''), rec.get('best_season', '')
                )
                score = rec.get('hybrid_score', 0)
                max_s = max((r.get('hybrid_score', 0.01) for r in recs), default=0.01)
                rec['match_pct'] = min(99, max(40, int(score / max_s * 98)))

            return render_template(
                'result.html',
                recommendations=recs,
                query=dict(data),
                count=len(recs),
                **opts,
            )

        except Exception as e:
            return render_template('recommend.html', **opts,
                                   error=f"Recommendation error: {str(e)}")

    return render_template('recommend.html', **opts)


@app.route('/api/recommend', methods=['POST'])
def api_recommend():
    """JSON API endpoint for recommendations."""
    data = request.get_json() or request.form.to_dict()
    hybrid_rec = _models.get('hybrid_recommender')
    if hybrid_rec is None:
        return jsonify({'error': 'Model not loaded'}), 503

    try:
        results = hybrid_rec.recommend(
            category=data.get('category') or None,
            province=data.get('province') or None,
            season=data.get('season') or None,
            budget_level=data.get('budget_level') or None,
            min_rating=float(data.get('min_rating', 3.5)),
            n=int(data.get('n', 10)),
            attraction_name=data.get('attraction_name') or None,
        )
        recs = dataframe_to_json(results)
        return jsonify({'success': True, 'count': len(recs), 'recommendations': recs})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/destination/<path:name>')
def destination_detail(name):
    """Destination detail page."""
    catalog = _catalog
    if catalog is None:
        return redirect(url_for('index'))

    match = catalog[catalog['attraction_name'].str.lower() == name.lower()]
    if match.empty:
        match = catalog[catalog['attraction_name'].str.lower().str.contains(
            name.lower(), na=False
        )]
    if match.empty:
        return render_template('destination_explorer.html',
                               error=f'Destination "{name}" not found',
                               **get_filter_options())

    dest = match.iloc[0].to_dict()

    # Get similar destinations
    content_rec = _models.get('content_recommender')
    similar = []
    if content_rec:
        try:
            sim_df = content_rec.get_similar(name, n=6)
            similar = dataframe_to_json(sim_df)
            for s in similar:
                s['category_icon'] = get_category_icon(s.get('category', ''))
                s['season_icon'] = get_season_icon(s.get('best_season', ''))
        except Exception:
            pass

    # Get visitor stats from raw data
    visits = _df[_df['attraction_name'].str.lower() == name.lower()]
    visitor_stats = {}
    if not visits.empty:
        visitor_stats = {
            'total_visits': len(visits),
            'avg_rating': round(visits['rating'].mean(), 2),
            'avg_spend': round(visits['spend_amount'].mean(), 1),
            'gender_split': visits['gender'].value_counts().to_dict(),
            'age_groups': visits['age_group'].value_counts().to_dict(),
            'top_season': visits['season'].replace({'Chun Ji': 'Spring'}).value_counts().index[0]
            if len(visits) else 'N/A',
        }

    # Gemini insight
    insight = generate_travel_insight(
        destination=dest.get('attraction_name', name),
        category=str(dest.get('category', '')),
        province=str(dest.get('province', '')),
        avg_rating=float(dest.get('avg_rating', 4.0)),
        budget_level=str(dest.get('budget_level', 'Mid-Range')),
        best_season=str(dest.get('best_season', 'Spring')),
    )

    dest['budget_badge'] = get_budget_badge(dest.get('budget_level'))
    dest['season_icon'] = get_season_icon(dest.get('best_season'))
    dest['category_icon'] = get_category_icon(dest.get('category'))

    return render_template(
        'destination_explorer.html',
        destination=dest,
        similar=similar,
        visitor_stats=visitor_stats,
        insight=insight,
        **get_filter_options()
    )


@app.route('/explore')
def destination_explorer():
    """Destination explorer page."""
    catalog = _catalog
    if catalog is None:
        return render_template('destination_explorer.html', **get_filter_options())

    category = request.args.get('category', '')
    province = request.args.get('province', '')
    season = request.args.get('season', '')
    budget = request.args.get('budget', '')
    search = request.args.get('search', '')
    sort_by = request.args.get('sort_by', 'popularity_score')

    df = catalog.copy()
    if category:
        df = df[df['category'].str.lower() == category.lower()]
    if province:
        df = df[df['province'].str.lower() == province.lower()]
    if season:
        df = df[df['best_season'].str.lower() == season.lower()]
    if budget:
        df = df[df['budget_level'].astype(str).str.lower() == budget.lower()]
    if search:
        df = df[
            df['attraction_name'].str.lower().str.contains(search.lower(), na=False) |
            df['city'].str.lower().str.contains(search.lower(), na=False) |
            df['province'].str.lower().str.contains(search.lower(), na=False)
        ]

    valid_sorts = ['popularity_score', 'avg_rating', 'avg_ticket_price', 'visit_count']
    if sort_by not in valid_sorts:
        sort_by = 'popularity_score'
    df = df.sort_values(sort_by, ascending=False).head(60)

    destinations = dataframe_to_json(df)
    for d in destinations:
        d['budget_badge'] = get_budget_badge(d.get('budget_level'))
        d['season_icon'] = get_season_icon(d.get('best_season'))
        d['category_icon'] = get_category_icon(d.get('category'))

    return render_template(
        'destination_explorer.html',
        destinations=destinations,
        total=len(destinations),
        filters=dict(category=category, province=province, season=season,
                     budget=budget, search=search, sort_by=sort_by),
        metadata=_metadata,
        **get_filter_options()
    )


@app.route('/activities')
def activity_explorer():
    """Activity explorer page."""
    search = request.args.get('search', '')
    category = request.args.get('category', '')

    activity_rec = _models.get('activity_recommender')
    results = []

    if search and activity_rec:
        try:
            df = activity_rec.recommend_by_activity(search, n=20)
            results = dataframe_to_json(df)
            for r in results:
                r['category_icon'] = get_category_icon(r.get('category', ''))
                r['season_icon'] = get_season_icon(r.get('best_season', ''))
        except Exception:
            pass

    # Category stats
    cat_stats = _df.groupby('attraction_category').agg(
        count=('tourist_id', 'count'),
        avg_rating=('rating', 'mean'),
        avg_spend=('spend_amount', 'mean')
    ).reset_index().sort_values('count', ascending=False)
    cat_stats_list = dataframe_to_json(cat_stats)
    for c in cat_stats_list:
        c['icon'] = get_category_icon(c.get('attraction_category', ''))

    return render_template(
        'activity_explorer.html',
        results=results,
        search=search,
        category=category,
        category_stats=cat_stats_list,
        **get_filter_options()
    )


@app.route('/dashboard')
def dashboard():
    """EDA Dashboard with charts."""
    charts = {
        'top_attractions': get_top_attractions_chart(_df),
        'categories': get_category_distribution_chart(_df),
        'ratings': get_rating_distribution_chart(_df),
        'budget': get_budget_distribution_chart(_df),
        'season': get_season_demand_chart(_df),
        'provinces': get_province_chart(_df),
        'rating_by_cat': get_rating_by_category_chart(_df),
        'age_groups': get_age_group_chart(_df),
        'gender': get_gender_chart(_df),
        'satisfaction': get_satisfaction_chart(_df),
        'spend_rating': get_spend_vs_rating_chart(_df),
    }
    stats = get_summary_stats(_df)
    return render_template('dashboard.html', charts=charts, stats=stats)


@app.route('/model-performance')
def model_performance():
    """Model performance metrics page."""
    metrics = _metrics or {}
    metadata = _metadata or {}

    # Model comparison table
    models_table = [
        {'model': 'Hybrid Recommender', 'type': 'Recommendation', 'metric': 'Coverage',
         'value': '100%', 'status': 'production'},
        {'model': 'Content-Based (TF-IDF)', 'type': 'Recommendation', 'metric': 'Cosine Similarity',
         'value': '~0.85', 'status': 'production'},
        {'model': 'Popularity Baseline', 'type': 'Recommendation', 'metric': 'Popularity Score',
         'value': 'N/A', 'status': 'baseline'},
        {'model': 'Random Forest Classifier', 'type': 'Classification',
         'metric': 'Accuracy', 'value': f"{metrics.get('classification_accuracy', 'N/A')}",
         'status': 'production'},
        {'model': 'Gradient Boosting Regressor', 'type': 'Regression',
         'metric': 'R² Score', 'value': f"{metrics.get('regression_r2', 'N/A')}",
         'status': 'production'},
        {'model': 'KMeans Clustering', 'type': 'Clustering',
         'metric': 'Clusters', 'value': '6', 'status': 'production'},
    ]

    return render_template(
        'model_performance.html',
        metrics=metrics,
        metadata=metadata,
        models_table=models_table,
    )


@app.route('/about')
def about():
    metadata = _metadata or {}
    return render_template('about.html', metadata=metadata)


@app.route('/api/insight', methods=['POST'])
def api_insight():
    """Generate Gemini AI insight for a destination."""
    data = request.get_json() or {}
    insight = generate_travel_insight(
        destination=data.get('destination', ''),
        category=data.get('category', 'Natural Scenery'),
        province=data.get('province', 'China'),
        avg_rating=float(data.get('avg_rating', 4.0)),
        budget_level=data.get('budget_level', 'Mid-Range'),
        best_season=data.get('best_season', 'Spring'),
        user_preferences=data.get('preferences', {}),
    )
    return jsonify(insight)


@app.route('/api/stats')
def api_stats():
    stats = get_summary_stats(_df)
    return jsonify(stats)


@app.route('/api/suggest')
def api_suggest():
    """Autocomplete suggestions for attraction names."""
    q = request.args.get('q', '').lower().strip()
    if not q or len(q) < 2:
        return jsonify({'suggestions': []})
    catalog = _catalog
    if catalog is None:
        return jsonify({'suggestions': []})
    matches = catalog[
        catalog['attraction_name'].str.lower().str.contains(q, na=False)
    ].head(8)
    suggestions = []
    for _, row in matches.iterrows():
        suggestions.append({
            'name': row['attraction_name'],
            'category': row.get('category', ''),
            'province': row.get('province', ''),
            'icon': get_category_icon(row.get('category', '')),
        })
    return jsonify({'suggestions': suggestions})


@app.template_filter('currency')
def currency_filter(value):
    try:
        return f"¥{float(value):,.0f}"
    except Exception:
        return str(value)


@app.template_filter('rating_stars')
def rating_stars_filter(value):
    try:
        r = float(value)
        full = int(r)
        half = 1 if r - full >= 0.5 else 0
        empty = 5 - full - half
        return '★' * full + '½' * half + '☆' * empty
    except Exception:
        return '★★★★☆'


@app.context_processor
def inject_globals():
    return {
        'app_name': 'TourMate AI',
        'tagline': 'Intelligent Tourism Recommendation System',
        'nav_links': [
            {'url': url_for('index'), 'label': 'Home', 'icon': '🏠'},
            {'url': url_for('recommend'), 'label': 'Get Recommendations', 'icon': '🎯'},
            {'url': url_for('destination_explorer'), 'label': 'Explore', 'icon': '🗺️'},
            {'url': url_for('activity_explorer'), 'label': 'Activities', 'icon': '🎭'},
            {'url': url_for('dashboard'), 'label': 'Dashboard', 'icon': '📊'},
            {'url': url_for('model_performance'), 'label': 'Model Performance', 'icon': '🤖'},
            {'url': url_for('about'), 'label': 'About', 'icon': 'ℹ️'},
        ]
    }


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
