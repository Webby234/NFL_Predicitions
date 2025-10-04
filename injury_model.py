import pandas as pd
import requests
from selenium import webdriver
from bs4 import BeautifulSoup

# Position-based importance weights
POSITION_WEIGHTS = {
    'QB': 1.0,
    'RB': 0.8,
    'WR': 0.8,
    'TE': 0.6,
    'OL': 0.5,
    'DL': 0.5,
    'LB': 0.5,
    'CB': 0.6,
    'S': 0.6,
    'K': 0.2,
    'P': 0.2
}

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0 Safari/537.36"
}

def scrape_espn_injuries(season: int, week: int) -> pd.DataFrame:
    """
    Scrapes ESPN NFL injury report and returns a DataFrame with OUT players.
    """
    url = "https://www.espn.com/nfl/injuries"
    response = requests.get(url, headers=headers)
    print(response.text[:1000])  # Print first 1000 characters of the page

    soup = BeautifulSoup(response.text, "html.parser")

    teams = soup.find_all("section", class_="ResponsiveTable")
    injury_data = []

    for team_section in teams:
        team_name = team_section.find("h2").text.strip()
        rows = team_section.find_all("tr")[1:]  # Skip header

        for row in rows:
            cols = row.find_all("td")
            if len(cols) >= 4:
                player = cols[0].text.strip()
                position = cols[1].text.strip().upper()
                injury = cols[2].text.strip()
                status = cols[3].text.strip().lower()

                if status == "out":
                    injury_data.append({
                        "season": season,
                        "week": week,
                        "team": team_name,
                        "player_name": player,
                        "position": position,
                        "injury": injury,
                        "status": status,
                        "weight": POSITION_WEIGHTS.get(position, 0.3)
                    })

    return pd.DataFrame(injury_data)

def compute_team_injury_scores(injury_df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame with injury scores per team per week.
    """
    if injury_df.empty or not all(col in injury_df.columns for col in ['season', 'week', 'team', 'weight']):
        return pd.DataFrame(columns=['season', 'week', 'team', 'injury_score'])

    team_scores = injury_df.groupby(['season', 'week', 'team'])['weight'].sum().reset_index()
    team_scores.rename(columns={'weight': 'injury_score'}, inplace=True)
    return team_scores

def add_injury_features(schedule_df: pd.DataFrame, injury_scores: pd.DataFrame) -> pd.DataFrame:
    """
    Adds home_injury_score, away_injury_score, and injury_diff to schedule_df.
    """
    df = schedule_df.copy()

    # Merge home team injury scores
    df = df.merge(injury_scores, left_on=['season', 'week', 'home_team'],
                  right_on=['season', 'week', 'team'], how='left')
    df.rename(columns={'injury_score': 'home_injury_score'}, inplace=True)
    df.drop(columns=['team'], inplace=True, errors='ignore')

    # Merge away team injury scores
    df = df.merge(injury_scores, left_on=['season', 'week', 'away_team'],
                  right_on=['season', 'week', 'team'], how='left')
    df.rename(columns={'injury_score': 'away_injury_score'}, inplace=True)
    df.drop(columns=['team'], inplace=True, errors='ignore')

    # Fill missing values and compute injury differential
    df['home_injury_score'] = df['home_injury_score'].fillna(0)
    df['away_injury_score'] = df['away_injury_score'].fillna(0)
    df['injury_diff'] = df['away_injury_score'] - df['home_injury_score']

    return df
