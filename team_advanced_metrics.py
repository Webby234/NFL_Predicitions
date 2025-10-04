from nfl_data_py import import_pbp_data
import pandas as pd
import numpy as np

def compute_team_advanced_metrics(season: int, week: int, pbp: pd.DataFrame) -> pd.DataFrame:
    pbp_week = pbp[pbp['week'] == week]

    valid_plays = pbp_week[pbp_week['epa'].notnull()].copy()
    valid_plays['success'] = valid_plays.apply(lambda row: (
        row['yards_gained'] >= 0.5 * row['ydstogo'] if row['down'] == 1 else
        row['yards_gained'] >= 0.7 * row['ydstogo'] if row['down'] == 2 else
        row['yards_gained'] >= row['ydstogo'] if row['down'] in [3, 4] else False
    ), axis=1)

    redzone = valid_plays[valid_plays['yardline_100'] <= 20]

    offense = valid_plays.groupby('posteam').agg({
        'epa': 'mean',
        'success': 'mean',
        'third_down_converted': 'mean'
    }).rename(columns={
        'epa': 'epa_offense',
        'success': 'success_rate_offense',
        'third_down_converted': 'third_down_pct_offense'
    })

    redzone_offense = redzone.groupby('posteam')['touchdown'].mean().rename('redzone_td_pct_offense')

    defense = valid_plays.groupby('defteam').agg({
        'epa': 'mean',
        'success': 'mean',
        'third_down_converted': 'mean'
    }).rename(columns={
        'epa': 'epa_defense',
        'success': 'success_rate_defense',
        'third_down_converted': 'third_down_pct_defense'
    })

    redzone_defense = redzone.groupby('defteam')['touchdown'].mean().rename('redzone_td_pct_defense')

    teams = pd.DataFrame({'team': sorted(set(offense.index) | set(defense.index))})
    teams = teams.set_index('team')
    teams = teams.join(offense).join(defense).join(redzone_offense).join(redzone_defense)
    teams = teams.fillna(0)

    teams['epa_diff'] = teams['epa_offense'] - teams['epa_defense']
    teams['success_rate_diff'] = teams['success_rate_offense'] - teams['success_rate_defense']
    teams['redzone_diff'] = teams['redzone_td_pct_offense'] - teams['redzone_td_pct_defense']
    teams['third_down_diff'] = teams['third_down_pct_offense'] - teams['third_down_pct_defense']
    teams['pressure_rate_allowed'] = np.nan
    teams['pressure_rate_generated'] = np.nan
    teams['pressure_diff'] = np.nan

    teams.reset_index(inplace=True)
    teams['week'] = week
    teams['season'] = season

    return teams
