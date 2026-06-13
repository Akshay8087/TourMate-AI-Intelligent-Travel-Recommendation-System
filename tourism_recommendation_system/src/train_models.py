"""
TourMate AI - Model Training Script
Trains all ML models and recommendation systems, saves artifacts.
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_preprocessing import load_dataset, clean_dataset, build_attraction_catalog
from src.recommender import (
    PopularityRecommender, ContentBasedRecommender,
    HybridRecommender, ActivityRecommender
)


def train_classification_model(df: pd.DataFrame):
    """Train classification model to predict recommendation_level."""
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, accuracy_score
    from sklearn.pipeline import Pipeline

    print("Training classification model...")

    le_target = LabelEncoder()
    y = le_target.fit_transform(df['recommendation_level'].astype(str))

    # Features
    feature_cols = ['age', 'ticket_price', 'visit_duration_hours', 'spend_amount',
                    'other_spend', 'rating', 'is_holiday_enc', 'is_group_tour_enc']

    # Encode categoricals
    le_cat = LabelEncoder()
    le_season = LabelEncoder()
    le_gender = LabelEncoder()
    le_agegroup = LabelEncoder()
    le_atlevel = LabelEncoder()

    df2 = df.copy()
    df2['cat_enc'] = le_cat.fit_transform(df2['attraction_category'].astype(str))
    df2['season_enc'] = le_season.fit_transform(df2['season'].astype(str))
    df2['gender_enc'] = le_gender.fit_transform(df2['gender'].astype(str))
    df2['age_group_enc'] = le_agegroup.fit_transform(df2['age_group'].astype(str))
    df2['atlevel_enc'] = le_atlevel.fit_transform(df2['attraction_level'].astype(str))

    feature_cols += ['cat_enc', 'season_enc', 'gender_enc', 'age_group_enc', 'atlevel_enc']
    X = df2[feature_cols].fillna(0).values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                          random_state=42, stratify=y)

    clf = RandomForestClassifier(
        n_estimators=200, max_depth=15, min_samples_split=5,
        random_state=42, n_jobs=-1, class_weight='balanced'
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"  Classification Accuracy: {acc:.4f}")

    encoders = {
        'le_target': le_target,
        'le_cat': le_cat,
        'le_season': le_season,
        'le_gender': le_gender,
        'le_agegroup': le_agegroup,
        'le_atlevel': le_atlevel,
    }

    metrics = {
        'classification_accuracy': round(acc, 4),
        'classification_report': classification_report(y_test, y_pred,
                                                        target_names=le_target.classes_,
                                                        output_dict=True)
    }

    return clf, encoders, feature_cols, metrics


def train_regression_model(df: pd.DataFrame):
    """Train regression model to predict rating."""
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    from sklearn.preprocessing import LabelEncoder

    print("Training regression model...")

    df2 = df.copy()
    for col in ['attraction_category', 'season', 'gender', 'age_group',
                'attraction_level', 'province']:
        le = LabelEncoder()
        df2[col + '_enc'] = le.fit_transform(df2[col].astype(str))

    feature_cols = ['age', 'ticket_price', 'visit_duration_hours', 'spend_amount',
                    'is_holiday_enc', 'is_group_tour_enc',
                    'attraction_category_enc', 'season_enc', 'gender_enc',
                    'age_group_enc', 'attraction_level_enc', 'province_enc']

    feature_cols = [c for c in feature_cols if c in df2.columns]
    X = df2[feature_cols].fillna(0).values
    y = df2['rating'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    reg = GradientBoostingRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        random_state=42, subsample=0.8
    )
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"  Regression RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")

    metrics = {
        'regression_rmse': round(rmse, 4),
        'regression_mae': round(mae, 4),
        'regression_r2': round(r2, 4),
    }

    return reg, feature_cols, metrics


def train_clustering(catalog: pd.DataFrame):
    """Train KMeans clustering on attraction catalog."""
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    import warnings
    warnings.filterwarnings('ignore')

    print("Training clustering model...")

    num_cols = ['avg_rating', 'avg_ticket_price', 'avg_spend', 'avg_duration',
                'popularity_score', 'visit_count']
    num_cols = [c for c in num_cols if c in catalog.columns]

    X = catalog[num_cols].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=6, random_state=42, n_init=10, max_iter=300)
    clusters = kmeans.fit_predict(X_scaled)

    catalog = catalog.copy()
    catalog['cluster'] = clusters

    # Profile clusters
    cluster_names = {
        0: 'Budget Cultural Explorers',
        1: 'Luxury Nature Seekers',
        2: 'Family Heritage Travelers',
        3: 'Adventure Enthusiasts',
        4: 'Urban Experience Seekers',
        5: 'Spiritual & Wellness Tourists',
    }

    print(f"  Created {len(np.unique(clusters))} clusters")
    return kmeans, scaler, cluster_names, catalog


def main():
    """Main training pipeline."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'tourism_recommendation_dataset_en.csv')
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)

    print("=" * 60)
    print("TourMate AI - Model Training Pipeline")
    print("=" * 60)

    # 1. Load & Clean
    print("\n[1/7] Loading and cleaning dataset...")
    df = load_dataset(data_path)
    df = clean_dataset(df)
    print(f"  Dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")

    # 2. Build attraction catalog
    print("\n[2/7] Building attraction catalog...")
    catalog = build_attraction_catalog(df)
    print(f"  Catalog: {len(catalog)} unique attractions")

    # 3. Content-Based Recommender
    print("\n[3/7] Training content-based recommender...")
    content_rec = ContentBasedRecommender()
    content_rec.fit(catalog)
    print(f"  TF-IDF matrix: {content_rec.tfidf_matrix.shape}")

    # 4. Popularity Recommender
    print("\n[4/7] Training popularity recommender...")
    pop_rec = PopularityRecommender()
    pop_rec.fit(catalog)

    # 5. Hybrid Recommender
    print("\n[5/7] Building hybrid recommender...")
    hybrid_rec = HybridRecommender()
    hybrid_rec.fit(catalog, content_rec)

    # 6. Activity Recommender
    activity_rec = ActivityRecommender()
    activity_rec.fit(catalog)

    # 7. ML Models
    print("\n[6/7] Training ML classification/regression models...")
    clf, clf_encoders, clf_features, clf_metrics = train_classification_model(df)
    reg, reg_features, reg_metrics = train_regression_model(df)

    # 8. Clustering
    print("\n[7/7] Training clustering...")
    kmeans, cluster_scaler, cluster_names, catalog_with_clusters = train_clustering(catalog)

    # Save everything
    print("\nSaving model artifacts...")
    artifacts = {
        'popularity_recommender': pop_rec,
        'content_recommender': content_rec,
        'hybrid_recommender': hybrid_rec,
        'activity_recommender': activity_rec,
        'tfidf_vectorizer': content_rec.tfidf,
        'similarity_model': content_rec.nn_model,
        'classification_model': clf,
        'regression_model': reg,
        'clustering_model': kmeans,
        'cluster_scaler': cluster_scaler,
        'clf_encoders': clf_encoders,
        'catalog': catalog_with_clusters,
        'attraction_catalog': catalog,
    }

    for name, obj in artifacts.items():
        joblib.dump(obj, os.path.join(models_dir, f'{name}.pkl'))

    # Save feature names
    feature_names = {
        'clf_features': clf_features,
        'reg_features': reg_features,
        'cluster_names': cluster_names,
    }
    with open(os.path.join(models_dir, 'feature_names.json'), 'w') as f:
        json.dump(feature_names, f, indent=2)

    # Save metrics
    all_metrics = {**clf_metrics, **reg_metrics}
    with open(os.path.join(models_dir, 'model_metrics.json'), 'w') as f:
        json.dump(all_metrics, f, indent=2)

    # Save metadata
    metadata = {
        'n_attractions': len(catalog),
        'n_training_records': len(df),
        'categories': sorted(catalog['category'].unique().tolist()),
        'provinces': sorted(catalog['province'].unique().tolist()),
        'seasons': ['Spring', 'Summer', 'Autumn', 'Winter'],
        'budget_levels': ['Free', 'Budget', 'Mid-Range', 'Premium', 'Luxury'],
        'attraction_levels': ['3A', '4A', '5A'],
        'hybrid_weights': hybrid_rec.weights,
        'tfidf_vocab_size': len(content_rec.tfidf.vocabulary_),
        'n_clusters': 6,
        'model_version': '1.0.0',
    }
    with open(os.path.join(models_dir, 'model_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 60)
    print("✓ Training Complete!")
    print(f"  Classification Accuracy: {clf_metrics['classification_accuracy']:.4f}")
    print(f"  Regression R²: {reg_metrics['regression_r2']:.4f}")
    print(f"  Models saved to: {models_dir}/")
    print("=" * 60)

    return artifacts, metadata


if __name__ == '__main__':
    main()
