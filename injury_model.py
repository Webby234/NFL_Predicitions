import pandas as pd
import requests
from bs4 import BeautifulSoup

# Position-based importance weights
POSITION_WEIGHTS = {
    'QB': 1.0, 'RB': 0.8, 'WR': 0.8, 'TE': 0.6,
    'OL': 0.5, 'DL': 0.5, 'LB': 0.5, 'CB': 0.6,
    'S': 0.6, 'K': 0.2, 'P': 0.2
}

def scrape_fantasypros_injuries(season: int, week: int) -> pd.DataFrame:
    """
    Scrapes FantasyPros NFL injury report across all positions and returns a DataFrame of OUT players.
    """
    url = "https://www.fantasypros.com/nfl/players/injuries.php"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")

    injury_data = []

    # Loop through all positional sections
    for section in soup.find_all("div", class_="mobile-table-report-page"):
        position_header = section.find_previous_sibling("h4")
        position = position_header.text.strip().upper() if position_header else "UNK"

        table = section.find("table")
        if not table or not table.find("tbody"):
            continue

        for row in table.find("tbody").find_all("tr"):
            cols = row.find_all("td")
            if len(cols) >= 3:
                player_cell = cols[0]
                player_name_tag = player_cell.find("a")
                team_tag = player_cell.find("small")

                player_name = player_name_tag.text.strip() if player_name_tag else "Unknown"
                team = team_tag.text.strip() if team_tag else "UNK"
                status = cols[1].text.strip().lower()
                injury = cols[2].text.strip()

                if status in ["out", "ir", "pup"]:
                    weight = POSITION_WEIGHTS.get(position, 0.3)
                    if status != "out":
                        weight *= 0.5
                    injury_data.append({
                        "season": season,
                        "week": week,
                        "team": team,
                        "player_name": player_name,
                        "position": position,
                        "injury": injury,
                        "status": status,
                        "weight": POSITION_WEIGHTS.get(position, 0.3)
                    })

    df = pd.DataFrame(injury_data)
    print(f"✅ Scraped {len(df)} injuries from FantasyPros for Week {week}")
    return df

def compute_team_injury_scores(injury_df: pd.DataFrame) -> pd.DataFrame:
    if injury_df.empty or not all(col in injury_df.columns for col in ['season', 'week', 'team', 'weight']):
        return pd.DataFrame(columns=['season', 'week', 'team', 'injury_score'])

    team_scores = injury_df.groupby(['season', 'week', 'team'])['weight'].sum().reset_index()
    team_scores.rename(columns={'weight': 'injury_score'}, inplace=True)
    return team_scores

def add_injury_features(schedule_df: pd.DataFrame, injury_scores: pd.DataFrame) -> pd.DataFrame:
    df = schedule_df.copy()

    df = df.merge(injury_scores, left_on=['season', 'week', 'home_team'],
                  right_on=['season', 'week', 'team'], how='left')
    df.rename(columns={'injury_score': 'home_injury_score'}, inplace=True)
    df.drop(columns=['team'], inplace=True, errors='ignore')

    df = df.merge(injury_scores, left_on=['season', 'week', 'away_team'],
                  right_on=['season', 'week', 'team'], how='left')
    df.rename(columns={'injury_score': 'away_injury_score'}, inplace=True)
    df.drop(columns=['team'], inplace=True, errors='ignore')

    df['home_injury_score'] = df['home_injury_score'].fillna(0)
    df['away_injury_score'] = df['away_injury_score'].fillna(0)
    df['injury_diff'] = df['away_injury_score'] - df['home_injury_score']

    return df
