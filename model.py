import nfl_data_py as nfl
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import streamlit as st

# Load schedule and weekly stats
schedule = nfl.import_schedules(years=list(range(2020, 2025)))
weekly = nfl.import_weekly_data(years=list(range(2022, 2025)))  # Use 2022–2024 for team stats
weekly = weekly.rename(columns={'recent_team': 'team'})

# Filter for regular season games only
schedule = schedule[schedule['game_type'] == 'REG']

# Weighted team stats: favor recent seasons
weights = {2022: 1, 2023: 2, 2024: 3}
weekly['weight'] = weekly['season'].map(weights)
for col in ['fantasy_points', 'passing_yards', 'rushing_yards', 'receiving_yards', 'interceptions']:
    weekly[col] = weekly[col] * weekly['weight']

team_stats = weekly.groupby('team').agg({
    'fantasy_points': 'sum',
    'passing_yards': 'sum',
    'rushing_yards': 'sum',
    'receiving_yards': 'sum',
    'interceptions': 'sum',
    'weight': 'sum'
}).reset_index()

# Normalize by total weight
for col in ['fantasy_points', 'passing_yards', 'rushing_yards', 'receiving_yards', 'interceptions']:
    team_stats[col] = team_stats[col] / team_stats['weight']

# Merge team stats into historical schedule for training
data = schedule.merge(team_stats, left_on='home_team', right_on='team', suffixes=('', '_home'))
data = data.merge(team_stats, left_on='away_team', right_on='team', suffixes=('', '_away'))

# Feature engineering
data['spread_line'] = pd.to_numeric(data['spread_line'], errors='coerce')
data['home_win'] = data['result'] > 0
data['home_points'] = data['fantasy_points']
data['away_points'] = data['fantasy_points_away']
data['home_yards'] = data['passing_yards'] + data['rushing_yards'] + data['receiving_yards']
data['away_yards'] = data['passing_yards_away'] + data['rushing_yards_away'] + data['receiving_yards_away']
data['home_turnovers'] = data['interceptions']
data['away_turnovers'] = data['interceptions_away']
data['yard_diff'] = data['home_yards'] - data['away_yards']
data['turnover_diff'] = data['away_turnovers'] - data['home_turnovers']  # fewer turnovers is better

# Fill missing values
data[['home_points', 'away_points', 'home_yards', 'away_yards', 'home_turnovers', 'away_turnovers',
      'spread_line', 'yard_diff', 'turnover_diff']] = data[[
    'home_points', 'away_points', 'home_yards', 'away_yards', 'home_turnovers', 'away_turnovers',
    'spread_line', 'yard_diff', 'turnover_diff'
]].fillna(0)

# Prepare training data
feature_cols = ['home_points', 'away_points', 'yard_diff', 'turnover_diff', 'spread_line']
features = data[feature_cols]
labels = data['home_win'].astype(int)

# Train model
if len(features) > 0 and labels.nunique() > 1:
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    model = XGBClassifier(base_score=0.5)
    model.fit(X_train, y_train)

    # Load 2025 schedule and filter for future games
    schedule_2025 = nfl.import_schedules(years=[2025])
    future_games = schedule_2025[(schedule_2025['game_type'] == 'REG') & (schedule_2025['result'].isna())]

    # Streamlit dashboard
    st.title("🏈 NFL Moneyline Predictor")
    st.write("This dashboard uses weighted team stats and spread lines to predict winners for upcoming 2025 NFL matchups.")

    # Week selector and confidence slider
    available_weeks = sorted(future_games['week'].dropna().unique())
    selected_week = st.selectbox("Select Week", available_weeks)
    confidence_cutoff = st.slider("Minimum Confidence", min_value=0.50, max_value=1.00, value=0.60)

    upcoming = future_games[future_games['week'] == selected_week]

    # Merge team stats
    upcoming = upcoming.merge(team_stats, left_on='home_team', right_on='team', suffixes=('', '_home'))
    upcoming = upcoming.merge(team_stats, left_on='away_team', right_on='team', suffixes=('', '_away'))

    # Feature engineering
    upcoming['home_points'] = upcoming['fantasy_points']
    upcoming['away_points'] = upcoming['fantasy_points_away']
    upcoming['home_yards'] = upcoming['passing_yards'] + upcoming['rushing_yards'] + upcoming['receiving_yards']
    upcoming['away_yards'] = upcoming['passing_yards_away'] + upcoming['rushing_yards_away'] + upcoming['receiving_yards_away']
    upcoming['home_turnovers'] = upcoming['interceptions']
    upcoming['away_turnovers'] = upcoming['interceptions_away']
    upcoming['yard_diff'] = upcoming['home_yards'] - upcoming['away_yards']
    upcoming['turnover_diff'] = upcoming['away_turnovers'] - upcoming['home_turnovers']
    upcoming['spread_line'] = pd.to_numeric(upcoming['spread_line'], errors='coerce')

    # Fill missing values
    upcoming[feature_cols] = upcoming[feature_cols].fillna(0)

    # Predict
    X_upcoming = upcoming[feature_cols]
    upcoming['home_win_pred'] = model.predict(X_upcoming)
    upcoming['confidence'] = model.predict_proba(X_upcoming).max(axis=1)
    upcoming['recommended_moneyline'] = upcoming.apply(
        lambda row: row['home_team'] if row['home_win_pred'] == 1 else row['away_team'], axis=1
    )

    # Display predictions above confidence threshold
    filtered = upcoming[upcoming['confidence'] >= confidence_cutoff]
    if len(filtered) == 0:
        st.write("No matchups meet the confidence threshold.")
    else:
        for _, row in filtered.iterrows():
            st.subheader(f"{row['away_team']} @ {row['home_team']}")
            st.write(f"**Recommended Pick**: {row['recommended_moneyline']}")
            st.write(f"**Confidence**: {row['confidence']:.2f}")
            st.write("---")

else:
    st.title("🏈 NFL Moneyline Predictor")
    st.write("Insufficient data to train the model. Weekly stats may be missing or incomplete for recent seasons.")
