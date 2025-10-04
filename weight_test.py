import nfl_data_py as nfl
from nfl_data_py import import_pbp_data
import pandas as pd
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import streamlit as st
from team_advanced_metrics import compute_team_advanced_metrics

# Define feature columns (pre-game only)
feature_cols = [
    'home_points', 'away_points', 'yard_diff', 'turnover_diff', 'spread_line',
    "epa_diff_home_adv", "epa_diff_away_adv",
    "success_rate_diff_home_adv", "success_rate_diff_away_adv",
    "redzone_diff_home_adv", "redzone_diff_away_adv",
    "third_down_diff_home_adv", "third_down_diff_away_adv",
    "pressure_diff_home_adv", "pressure_diff_away_adv",
    "injury_diff"
]

# Load schedule and weekly stats
schedule = nfl.import_schedules(years=[2023, 2024])
weekly = nfl.import_weekly_data(years=[2023, 2024])
weekly = weekly.rename(columns={'recent_team': 'team'})
schedule = schedule[schedule['game_type'] == 'REG']

# Precompute all play-by-play data
pbp_all = import_pbp_data([2023, 2024])

# Define weight schemes to test
weight_schemes = {
    "linear": {2023: 1, 2024: 2},
    "flat": {2023: 1, 2024: 1},
    "decay": {2023: 0.5, 2024: 1.0},
    "aggressive": {2023: 0.2, 2024: 1.5}
}

results = []

# Loop through each weight scheme
for scheme_name, weights in weight_schemes.items():
    weekly['weight'] = weekly['season'].map(weights)

    # Apply weights to key stats
    weighted = weekly.copy()
    for col in ['fantasy_points', 'passing_yards', 'rushing_yards', 'receiving_yards', 'interceptions']:
        weighted[col] *= weighted['weight']

    # Aggregate team stats
    team_stats = weighted.groupby('team').agg({
        'fantasy_points': 'sum',
        'passing_yards': 'sum',
        'rushing_yards': 'sum',
        'receiving_yards': 'sum',
        'interceptions': 'sum',
        'weight': 'sum'
    }).reset_index()

    for col in ['fantasy_points', 'passing_yards', 'rushing_yards', 'receiving_yards', 'interceptions']:
        team_stats[col] /= team_stats['weight']

    # Simulate predictions week-by-week
    week_results = []
    for season in [2023, 2024]:
        for week in range(2, 18):  # Start from week 2 to allow week 1 training
            # Filter schedule for current week
            games = schedule[(schedule['season'] == season) & (schedule['week'] == week)].copy()
            if games.empty:
                continue

            # Compute advanced metrics using data up to previous week
            pbp_subset = pbp_all[(pbp_all['season'] == season) & (pbp_all['week'] < week)]
            adv_metrics = compute_team_advanced_metrics(season=season, week=week, pbp=pbp_subset)

            # Merge team stats
            games = games.merge(team_stats, left_on='home_team', right_on='team', how='left', suffixes=('', '_home'))
            games = games.merge(team_stats, left_on='away_team', right_on='team', how='left', suffixes=('', '_away'))

            # Merge advanced metrics
            games = games.merge(adv_metrics, left_on='home_team', right_on='team', how='left', suffixes=('', '_home_adv'))
            home_cols = [col for col in adv_metrics.columns if col not in ['team', 'week', 'season']]
            games.rename(columns={col: f"{col}_home_adv" for col in home_cols}, inplace=True)

            games = games.merge(adv_metrics, left_on='away_team', right_on='team', how='left', suffixes=('', '_away_adv'))
            away_cols = [col for col in adv_metrics.columns if col not in ['team', 'week', 'season']]
            games.rename(columns={col: f"{col}_away_adv" for col in away_cols}, inplace=True)

            # Feature engineering
            games['spread_line'] = pd.to_numeric(games['spread_line'], errors='coerce')
            games['home_win'] = games['result'] > 0
            games['home_points'] = games['fantasy_points']
            games['away_points'] = games['fantasy_points_away']
            games['home_yards'] = games['passing_yards'] + games['rushing_yards'] + games['receiving_yards']
            games['away_yards'] = games['passing_yards_away'] + games['rushing_yards_away'] + games['receiving_yards_away']
            games['home_turnovers'] = games['interceptions']
            games['away_turnovers'] = games['interceptions_away']
            games['yard_diff'] = games['home_yards'] - games['away_yards']
            games['turnover_diff'] = games['away_turnovers'] - games['home_turnovers']
            games['injury_diff'] = 0
            games[feature_cols] = games[feature_cols].fillna(0)

            # Prepare training data from previous weeks
            train = schedule[(schedule['season'] == season) & (schedule['week'] < week)].copy()
            train = train.merge(team_stats, left_on='home_team', right_on='team', how='left', suffixes=('', '_home'))
            train = train.merge(team_stats, left_on='away_team', right_on='team', how='left', suffixes=('', '_away'))
            train = train.merge(adv_metrics, left_on='home_team', right_on='team', how='left', suffixes=('', '_home_adv'))
            train.rename(columns={col: f"{col}_home_adv" for col in home_cols}, inplace=True)
            train = train.merge(adv_metrics, left_on='away_team', right_on='team', how='left', suffixes=('', '_away_adv'))
            train.rename(columns={col: f"{col}_away_adv" for col in away_cols}, inplace=True)

            train['spread_line'] = pd.to_numeric(train['spread_line'], errors='coerce')
            train['home_win'] = train['result'] > 0
            train['home_points'] = train['fantasy_points']
            train['away_points'] = train['fantasy_points_away']
            train['home_yards'] = train['passing_yards'] + train['rushing_yards'] + train['receiving_yards']
            train['away_yards'] = train['passing_yards_away'] + train['rushing_yards_away'] + train['receiving_yards_away']
            train['home_turnovers'] = train['interceptions']
            train['away_turnovers'] = train['interceptions_away']
            train['yard_diff'] = train['home_yards'] - train['away_yards']
            train['turnover_diff'] = train['away_turnovers'] - train['home_turnovers']
            train['injury_diff'] = 0
            train[feature_cols] = train[feature_cols].fillna(0)

            if len(train) > 0 and games['home_win'].nunique() > 1:
                model = XGBClassifier()
                model.fit(train[feature_cols], train['home_win'])
                preds = model.predict(games[feature_cols])
                acc = accuracy_score(games['home_win'], preds)
                week_results.append(acc)

    # Average accuracy across all weeks
    avg_acc = sum(week_results) / len(week_results) if week_results else 0
    results.append((scheme_name, avg_acc))

# Display results
st.title("📊 True Predictive Weight Scheme Backtest")
df_results = pd.DataFrame(results, columns=["Scheme", "Avg Accuracy"])
df_results = df_results.sort_values(by="Avg Accuracy", ascending=False)
st.dataframe(df_results)
st.bar_chart(df_results.set_index("Scheme"))
