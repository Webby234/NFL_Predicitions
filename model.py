import nfl_data_py as nfl
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor
import streamlit as st
from injury_model import scrape_espn_injuries, compute_team_injury_scores, add_injury_features

# Load schedule and weekly stats
schedule = nfl.import_schedules(years=list(range(2020, 2025)))
weekly = nfl.import_weekly_data(years=list(range(2022, 2025)))
weekly = weekly.rename(columns={'recent_team': 'team'})
schedule = schedule[schedule['game_type'] == 'REG']

# Weighted team stats
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

for col in ['fantasy_points', 'passing_yards', 'rushing_yards', 'receiving_yards', 'interceptions']:
    team_stats[col] = team_stats[col] / team_stats['weight']

# Merge team stats into historical schedule
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
data['turnover_diff'] = data['away_turnovers'] - data['home_turnovers']
data['injury_diff'] = 0 # Placeholder for injury_diff

# Final feature set
feature_cols = ['home_points', 'away_points', 'yard_diff', 'turnover_diff', 'spread_line']
if 'injury_diff' in data.columns:
    feature_cols.append('injury_diff')
data[feature_cols] = data[feature_cols].fillna(0)
features = data[feature_cols]
labels = data['home_win'].astype(int)

# Train team-level model
team_model = None
if len(features) > 0 and labels.nunique() > 1:
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    team_model = XGBClassifier(base_score=0.5)
    team_model.fit(X_train, y_train)

# WR Prop Model
wr_data = weekly[weekly['position'] == 'WR']
wr_data = wr_data[['season', 'week', 'player_name', 'team', 'receiving_yards']].dropna()
wr_data = wr_data.merge(team_stats, on='team', suffixes=('', '_team'))

# Merge spread_line for home and away teams
home_merge = wr_data.merge(schedule[['season', 'week', 'home_team', 'spread_line']],
                           left_on=['season', 'week', 'team'],
                           right_on=['season', 'week', 'home_team'], how='left')
away_merge = wr_data.merge(schedule[['season', 'week', 'away_team', 'spread_line']],
                           left_on=['season', 'week', 'team'],
                           right_on=['season', 'week', 'away_team'], how='left')
wr_data['spread_line'] = home_merge['spread_line'].combine_first(away_merge['spread_line'])

# Train WR model
wr_feature_cols = ['passing_yards', 'rushing_yards', 'fantasy_points', 'interceptions', 'spread_line']
wr_data[wr_feature_cols] = wr_data[wr_feature_cols].fillna(0)
X_wr = wr_data[wr_feature_cols]
y_wr = wr_data['receiving_yards']
X_wr_train, X_wr_test, y_wr_train, y_wr_test = train_test_split(X_wr, y_wr, test_size=0.2, random_state=42)
wr_model = XGBRegressor()
wr_model.fit(X_wr_train, y_wr_train)

# Streamlit dashboard
st.title("🏈 NFL Prediction Dashboard")
tab1, tab2 = st.tabs(["💰 Moneyline Predictor", "📈 WR Prop Predictor"])

with tab1:
    # Load future schedule and team stats
    schedule_2025 = nfl.import_schedules(years=[2025])
    future_games = schedule_2025[(schedule_2025['game_type'] == 'REG') & (schedule_2025['result'].isna())]
    available_weeks = sorted(future_games['week'].dropna().unique())
    selected_week = st.selectbox("Select Week", available_weeks)
    confidence_cutoff = st.slider("Minimum Confidence", min_value=0.50, max_value=1.00, value=0.60)

    upcoming = future_games[future_games['week'] == selected_week]
    upcoming = upcoming.merge(team_stats, left_on='home_team', right_on='team', suffixes=('', '_home'))
    upcoming = upcoming.merge(team_stats, left_on='away_team', right_on='team', suffixes=('', '_away'))

    upcoming['home_points'] = upcoming['fantasy_points']
    upcoming['away_points'] = upcoming['fantasy_points_away']
    upcoming['home_yards'] = upcoming['passing_yards'] + upcoming['rushing_yards'] + upcoming['receiving_yards']
    upcoming['away_yards'] = upcoming['passing_yards_away'] + upcoming['rushing_yards_away'] + upcoming['receiving_yards_away']
    upcoming['home_turnovers'] = upcoming['interceptions']
    upcoming['away_turnovers'] = upcoming['interceptions_away']
    upcoming['yard_diff'] = upcoming['home_yards'] - upcoming['away_yards']
    upcoming['turnover_diff'] = upcoming['away_turnovers'] - upcoming['home_turnovers']
    upcoming['spread_line'] = pd.to_numeric(upcoming['spread_line'], errors='coerce')

    # Scrape live injury data for current season/week
    injuries_2025 = scrape_espn_injuries(season=2025, week=selected_week)
    print(f"Scraped {len(injuries_2025)} injuries for Week {selected_week}")
    injury_scores_2025 = compute_team_injury_scores(injuries_2025)
    upcoming = add_injury_features(upcoming, injury_scores_2025)
    upcoming['injury_diff'] = upcoming['injury_diff'].fillna(0)

    upcoming[feature_cols] = upcoming[feature_cols].fillna(0)
    X_upcoming = upcoming[feature_cols]
    upcoming['home_win_pred'] = team_model.predict(X_upcoming)
    upcoming['confidence'] = team_model.predict_proba(X_upcoming).max(axis=1)
    upcoming['recommended_moneyline'] = upcoming.apply(
        lambda row: row['home_team'] if row['home_win_pred'] == 1 else row['away_team'], axis=1
    )

    filtered = upcoming[upcoming['confidence'] >= confidence_cutoff]
    filtered = filtered.sort_values(by='confidence', ascending=False)

    if len(filtered) == 0:
        st.write("No matchups meet the confidence threshold.")
    else:
        for _, row in filtered.iterrows():
            st.subheader(f"{row['away_team']} @ {row['home_team']}")
            st.write(f"**Recommended Pick**: {row['recommended_moneyline']}")
            st.write(f"**Confidence**: {row['confidence']:.2f}")
            st.write("---")

with tab2:
    st.write("Predict whether a wide receiver will go over or under a receiving yards betting line.")
    selected_week_wr = st.selectbox("Select Week", sorted(wr_data['week'].unique()), key="wr_week")
    available_wrs = wr_data[wr_data['week'] == selected_week_wr]['player_name'].unique()
    selected_wr = st.selectbox("Select WR", sorted(available_wrs), key="wr_player")
    betting_line = st.number_input("Enter Betting Line (Receiving Yards)", min_value=0, max_value=300, value=50)

    wr_filtered = wr_data[(wr_data['week'] == selected_week_wr) & (wr_data['player_name'] == selected_wr)]
    if not wr_filtered.empty:
        wr_row = wr_filtered.iloc[0]
        X_input_wr = pd.DataFrame([wr_row[wr_feature_cols]])
        predicted_yards = wr_model.predict(X_input_wr)[0]
        recommendation = "Over" if predicted_yards > betting_line else "Under"

        st.subheader(f"{selected_wr} - Week {selected_week_wr}")
        st.write(f"**Predicted Receiving Yards**: {predicted_yards:.1f}")
        st.write(f"**Betting Line**: {betting_line}")
        st.write(f"**Recommendation**: {recommendation}")
    else:
        st.write("No data available for this player and week.")

   