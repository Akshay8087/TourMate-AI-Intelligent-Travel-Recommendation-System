"""
TourMate AI - EDA Utilities
Chart generation and analysis helpers for the dashboard.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import warnings
warnings.filterwarnings('ignore')

PALETTE = {
    'primary': '#0ea5e9',
    'secondary': '#38bdf8',
    'accent1': '#d4a373',
    'accent2': '#52c27a',
    'accent3': '#f97316',
    'bg': '#f8fafc',
    'card': '#ffffff',
    'text': '#1e293b',
    'muted': '#64748b',
}

CHART_COLORS = [
    '#0ea5e9', '#38bdf8', '#52c27a', '#d4a373', '#f97316',
    '#8b5cf6', '#ec4899', '#14b8a6', '#ef4444', '#84cc16'
]


def _fig_to_json(fig) -> str:
    return fig.to_json()


def get_top_attractions_chart(df: pd.DataFrame, n: int = 15) -> str:
    top = df['attraction_name'].value_counts().head(n).reset_index()
    top.columns = ['Attraction', 'Visit Count']
    fig = px.bar(
        top, x='Visit Count', y='Attraction', orientation='h',
        color='Visit Count', color_continuous_scale='Blues',
        title=f'Top {n} Most Visited Attractions'
    )
    fig.update_layout(
        plot_bgcolor='white', paper_bgcolor='white',
        font_family='Poppins, sans-serif',
        yaxis={'categoryorder': 'total ascending'},
        coloraxis_showscale=False,
        height=500,
        margin=dict(l=10, r=10, t=50, b=10)
    )
    return _fig_to_json(fig)


def get_category_distribution_chart(df: pd.DataFrame) -> str:
    cat = df['attraction_category'].value_counts().reset_index()
    cat.columns = ['Category', 'Count']
    fig = px.pie(
        cat, names='Category', values='Count',
        color_discrete_sequence=CHART_COLORS,
        title='Tourism Category Distribution',
        hole=0.4
    )
    fig.update_layout(
        plot_bgcolor='white', paper_bgcolor='white',
        font_family='Poppins, sans-serif',
        height=450,
        margin=dict(l=10, r=10, t=50, b=10)
    )
    return _fig_to_json(fig)


def get_rating_distribution_chart(df: pd.DataFrame) -> str:
    fig = px.histogram(
        df, x='rating', nbins=30,
        color_discrete_sequence=[PALETTE['primary']],
        title='Rating Distribution'
    )
    fig.update_layout(
        plot_bgcolor='white', paper_bgcolor='white',
        font_family='Poppins, sans-serif',
        height=350,
        xaxis_title='Rating', yaxis_title='Count',
        margin=dict(l=10, r=10, t=50, b=10)
    )
    return _fig_to_json(fig)


def get_budget_distribution_chart(df: pd.DataFrame) -> str:
    fig = px.histogram(
        df[df['ticket_price'] <= 300], x='ticket_price', nbins=40,
        color_discrete_sequence=[PALETTE['accent1']],
        title='Ticket Price Distribution'
    )
    fig.update_layout(
        plot_bgcolor='white', paper_bgcolor='white',
        font_family='Poppins, sans-serif',
        height=350,
        xaxis_title='Ticket Price (¥)', yaxis_title='Count',
        margin=dict(l=10, r=10, t=50, b=10)
    )
    return _fig_to_json(fig)


def get_season_demand_chart(df: pd.DataFrame) -> str:
    season_order = ['Spring', 'Summer', 'Autumn', 'Winter']
    df2 = df.copy()
    df2['season'] = df2['season'].replace({'Chun Ji': 'Spring'})
    season = df2['season'].value_counts().reindex(season_order, fill_value=0).reset_index()
    season.columns = ['Season', 'Count']
    fig = px.bar(
        season, x='Season', y='Count',
        color='Season',
        color_discrete_map={
            'Spring': '#52c27a', 'Summer': '#f97316',
            'Autumn': '#d4a373', 'Winter': '#0ea5e9'
        },
        title='Visitor Demand by Season'
    )
    fig.update_layout(
        plot_bgcolor='white', paper_bgcolor='white',
        font_family='Poppins, sans-serif',
        height=350, showlegend=False,
        margin=dict(l=10, r=10, t=50, b=10)
    )
    return _fig_to_json(fig)


def get_province_chart(df: pd.DataFrame, n: int = 15) -> str:
    prov = df['province'].value_counts().head(n).reset_index()
    prov.columns = ['Province', 'Count']
    fig = px.bar(
        prov, x='Province', y='Count',
        color='Count', color_continuous_scale='Teal',
        title=f'Top {n} Provinces by Tourism Volume'
    )
    fig.update_layout(
        plot_bgcolor='white', paper_bgcolor='white',
        font_family='Poppins, sans-serif',
        xaxis_tickangle=-30, coloraxis_showscale=False,
        height=400,
        margin=dict(l=10, r=10, t=50, b=80)
    )
    return _fig_to_json(fig)


def get_rating_by_category_chart(df: pd.DataFrame) -> str:
    cat_rating = df.groupby('attraction_category')['rating'].mean().sort_values(ascending=False).reset_index()
    cat_rating.columns = ['Category', 'Avg Rating']
    fig = px.bar(
        cat_rating, x='Avg Rating', y='Category', orientation='h',
        color='Avg Rating', color_continuous_scale='RdYlGn',
        range_color=[3.5, 5],
        title='Average Rating by Category'
    )
    fig.update_layout(
        plot_bgcolor='white', paper_bgcolor='white',
        font_family='Poppins, sans-serif',
        yaxis={'categoryorder': 'total ascending'},
        height=500,
        margin=dict(l=10, r=10, t=50, b=10)
    )
    return _fig_to_json(fig)


def get_age_group_chart(df: pd.DataFrame) -> str:
    order = ['18-25', '26-35', '36-45', '46-55', '56+']
    age = df['age_group'].value_counts().reindex(order, fill_value=0).reset_index()
    age.columns = ['Age Group', 'Count']
    fig = px.bar(
        age, x='Age Group', y='Count',
        color='Age Group', color_discrete_sequence=CHART_COLORS,
        title='Visitors by Age Group'
    )
    fig.update_layout(
        plot_bgcolor='white', paper_bgcolor='white',
        font_family='Poppins, sans-serif',
        showlegend=False, height=350,
        margin=dict(l=10, r=10, t=50, b=10)
    )
    return _fig_to_json(fig)


def get_gender_chart(df: pd.DataFrame) -> str:
    gender = df['gender'].value_counts().reset_index()
    gender.columns = ['Gender', 'Count']
    fig = px.pie(
        gender, names='Gender', values='Count',
        color_discrete_sequence=['#0ea5e9', '#f472b6'],
        title='Visitor Gender Distribution', hole=0.5
    )
    fig.update_layout(
        plot_bgcolor='white', paper_bgcolor='white',
        font_family='Poppins, sans-serif',
        height=350,
        margin=dict(l=10, r=10, t=50, b=10)
    )
    return _fig_to_json(fig)


def get_satisfaction_chart(df: pd.DataFrame) -> str:
    sat = df['satisfaction_level'].value_counts().reset_index()
    sat.columns = ['Satisfaction', 'Count']
    color_map = {
        'Very Satisfied': '#10b981',
        'Satisfied': '#3b82f6',
        'Neutral': '#f59e0b'
    }
    fig = px.bar(
        sat, x='Satisfaction', y='Count',
        color='Satisfaction',
        color_discrete_map=color_map,
        title='Visitor Satisfaction Levels'
    )
    fig.update_layout(
        plot_bgcolor='white', paper_bgcolor='white',
        font_family='Poppins, sans-serif',
        showlegend=False, height=350,
        margin=dict(l=10, r=10, t=50, b=10)
    )
    return _fig_to_json(fig)


def get_spend_vs_rating_chart(df: pd.DataFrame) -> str:
    sample = df.sample(min(3000, len(df)), random_state=42)
    fig = px.scatter(
        sample, x='spend_amount', y='rating',
        color='attraction_category',
        color_discrete_sequence=CHART_COLORS,
        title='Spend Amount vs Rating',
        opacity=0.6,
        labels={'spend_amount': 'Total Spend (¥)', 'rating': 'Rating'}
    )
    fig.update_layout(
        plot_bgcolor='white', paper_bgcolor='white',
        font_family='Poppins, sans-serif',
        height=450,
        margin=dict(l=10, r=10, t=50, b=10)
    )
    return _fig_to_json(fig)


def get_summary_stats(df: pd.DataFrame) -> dict:
    """Return key summary statistics for the dashboard."""
    df2 = df.copy()
    df2['season'] = df2['season'].replace({'Chun Ji': 'Spring'})
    return {
        'total_records': len(df),
        'unique_attractions': df['attraction_name'].nunique(),
        'unique_provinces': df['province'].nunique(),
        'unique_categories': df['attraction_category'].nunique(),
        'avg_rating': round(df['rating'].mean(), 2),
        'avg_ticket_price': round(df['ticket_price'].mean(), 1),
        'avg_spend': round(df['spend_amount'].mean(), 1),
        'highly_recommended_pct': round((df['recommendation_level'] == 'Highly Recommend').mean() * 100, 1),
        'very_satisfied_pct': round((df['satisfaction_level'] == 'Very Satisfied').mean() * 100, 1),
        'group_tour_pct': round((df['is_group_tour'] == 'Yes').mean() * 100, 1),
        'top_category': df['attraction_category'].value_counts().index[0],
        'top_province': df['province'].value_counts().index[0],
        'top_season': df2['season'].value_counts().index[0],
        'top_attraction': df['attraction_name'].value_counts().index[0],
    }
