import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# ==============================
# 1. load and split train data
# ==============================
def load_and_split_train(csv_path):
    print("Loading full dataset...")
    df = pd.read_csv(csv_path)

    dt = pd.to_datetime(df['time'], errors='coerce', utc=False)
    if dt.isna().all():
        raise ValueError("Column 'time' cannot be parsed as datetime")

    base_day = dt.dt.normalize().min()
    df['sensor_day'] = (dt.dt.normalize() - base_day).dt.days.astype('int32')

    valid_boundary = 22
    train_df = df[df['sensor_day'] < valid_boundary]

    return train_df


# ==============================
# 2. Sample from training data
# ==============================
def sample_train_data(train_df, sample_size=50000):
    if len(train_df) > sample_size:
        sampled_df = train_df.sample(sample_size)
    else:
        sampled_df = train_df.copy()

    return sampled_df


# ==============================
# 3. Feature selection preprocessing
# ==============================
def preprocess_for_feature_selection(df, target_col='values'):
    essential_cols = [
        'station_id', 'time', 'values', 'arrivals', 'departures',
        'hour', 'dow', 'is_weekend', 'sensor_day', 'latitude',
        'longitude', 'station_name'
    ]

    feature_cols = [col for col in df.columns if col not in essential_cols and col != target_col]

    print(f"Essential columns (not ranked): {len(essential_cols)}")
    print(f"Candidate features for ranking: {len(feature_cols)}")

    df_processed = df.copy()
    label_encoders = {}

    for col in feature_cols + essential_cols:
        if col in df_processed.columns and df_processed[col].dtype == 'object':
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col].astype(str))
            label_encoders[col] = le

    return df_processed, feature_cols, essential_cols


# ==============================
# 4. Feature importance calculation
# ==============================
def calculate_feature_importance(df, feature_cols, target_col='values', method='random_forest'):
    X = df[feature_cols].fillna(df[feature_cols].median())
    y = df[target_col]

    if method == 'random_forest':
        rf = RandomForestRegressor(n_estimators=100, n_jobs=-1, max_depth=10)
        rf.fit(X, y)
        importance_scores = rf.feature_importances_

    elif method == 'mutual_info':
        selector = SelectKBest(score_func=mutual_info_regression, k='all')
        selector.fit(X, y)
        importance_scores = selector.scores_

    return pd.DataFrame({
        'feature': feature_cols,
        'importance': importance_scores
    }).sort_values('importance', ascending=False)


# ==============================
# 5. select top features
# ==============================
def select_top_features(importance_df, essential_cols, top_k=20):
    top_features = importance_df.head(top_k)['feature'].tolist()
    final_features = essential_cols + top_features

    print(f"Essential columns: {len(essential_cols)}")
    print(f"Top selected features: {len(top_features)}")
    print(f"Total features: {len(final_features)}")

    return final_features, top_features


# ==============================
# 6. main execution
# ==============================
def main():
    csv_path = "202507-citibike-tft-features-cleaned-500000.csv"

    train_df = load_and_split_train(csv_path)

    sampled_df = sample_train_data(train_df, sample_size=50000)

    df_processed, feature_cols, essential_cols = preprocess_for_feature_selection(sampled_df)

    print("\nCalculating RF importance...")
    importance_rf = calculate_feature_importance(df_processed, feature_cols, method='random_forest')

    print("Calculating MI importance...")
    importance_mi = calculate_feature_importance(df_processed, feature_cols, method='mutual_info')

    importance_combined = importance_rf.copy()
    importance_combined['importance_mi'] = importance_mi['importance']

    for col in ['importance', 'importance_mi']:
        importance_combined[col + '_norm'] = (
            importance_combined[col] - importance_combined[col].min()
        ) / (importance_combined[col].max() - importance_combined[col].min())

    importance_combined['combined_score'] = (
        importance_combined['importance_norm'] +
        importance_combined['importance_mi_norm']
    ) / 2

    importance_combined = importance_combined.sort_values('combined_score', ascending=False)

    final_features, selected_features = select_top_features(importance_combined, essential_cols)

    print("\nTop features:")
    print(selected_features)

    return final_features, selected_features, importance_combined


if __name__ == "__main__":
    final_features, selected_features, importance_df = main()
