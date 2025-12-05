import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.image as mpimg
from scipy.ndimage import rotate
import ift6758.data.tidy_data as td
from scipy.ndimage import gaussian_filter
import os
import matplotlib.colors as colors

def get_offense_events(df_all_events : pd.DataFrame) -> pd.DataFrame:
    '''
    Function to filter the dataframe to only include events that happened in the offense zone.
    Input: dataframe of all the events.
    Output: dataframe with only events that happened in the offense zone.
    '''
    mask = (
        ((df_all_events['event_owner_team_id'] == df_all_events['home_id']) & (df_all_events['x'] > 0) & (df_all_events['home_defending_side'] == 'left')) |
        ((df_all_events['event_owner_team_id'] == df_all_events['home_id']) & (df_all_events['x'] < 0) & (df_all_events['home_defending_side'] == 'right')) |
        ((df_all_events['event_owner_team_id'] == df_all_events['away_id']) & (df_all_events['x'] < 0) & (df_all_events['home_defending_side'] == 'left')) |
        ((df_all_events['event_owner_team_id'] == df_all_events['away_id']) & (df_all_events['x'] > 0) & (df_all_events['home_defending_side'] == 'right'))
    )

    return df_all_events.loc[mask].copy()
                    
def compare_team_to_avg(team_df : pd.DataFrame, avg_df : pd.DataFrame, team : str = "_", season : str = "_") -> np.ndarray:
    '''
    Function to compare the average shots per hour per location to the shots per hour per location for a specific team.
    '''
    # convert each df into a 2D array with shape (width, height) where each cell is the number of shots per hour in that location
    league_avg = np.zeros((100, 85)) 
    team_avg = np.zeros((100, 85))
    for (x, y), value in avg_df.items():
        league_avg[int(x), int(y)+42] = value
    for (x, y), value in team_df.items():
        team_avg[int(x), int(y)+42] = value
    #league_avg = gaussian_filter(league_avg, sigma=1.5)
    #team_avg = gaussian_filter(team_avg, sigma=1.5)
    comparison = team_avg - league_avg/2
    np.save(f"ift6758/data/graph_data/comparison-{team}-{season}.npy", comparison)
    return comparison

def get_avg_shot_per_location_per_hour(df_shots : pd.DataFrame) -> pd.DataFrame:
    '''
    Function to get average number of shots per hour for each coordinate point in the offense zone.
    Args:
        df: dataframe of all the events to include in the average.
    Returns:
        dataframe with index (x_offense, y_offense) and value avg_shots_per_hour at that location.
    Note: We assume that each match is 60 minutes long and don't consider the power play situations.
    '''
    num_games = df_shots['game_id'].nunique()
    # On drop les na dans les matchs. i.e. on assume que tous les matchs ont at least 1 event de bien associé.
    df = get_offense_events(df_shots)
    # Apply coord change to half rink
    #df[['x_positive', 'y_positive']] = df.apply(lambda row: pd.Series(coord_tuple_to_half_rink(row['x'], row['y'])), axis=1)
    df['x_positive'] = np.where(df['x'] < 0, -df['x'], df['x'])
    df['y_positive'] = np.where(df['x'] < 0, -df['y'], df['y'])


    # Get total number of shots per location from dataframe
    # Group by x_offense and y_offense and count number of shots
    df_grouped = df.groupby(['x_positive', 'y_positive']).size()
    # Divide each data point by the number of matches (each match is 60 minutes)
    return 100*df_grouped / num_games

def plot_half_rink(array : np.ndarray, title : str, save_figure : bool = False, season : int = "_", team : str = "_", max : int = 10) -> Figure:
    '''
    Function to plot the array values in the half rink image.
    Input:
    array of shape (0, 0) # Change to the correct shape with each cell representing the value at that location.
    vmin and vmax: The min and max values for the color scale.
    Title: Title of the plot.
    Returns nothing, but shows the heat map on a half rink.
    '''
    assert array.shape == (100,85)
    
    rink_img = mpimg.imread("figures/nhl_rink.png")
    half_rink_img = rink_img[:, rink_img.shape[1]//2:, :]
    half_rink_img = rotate(half_rink_img, 90, reshape=True)
    image_ratio_loL = 85 / 100
    image_size = 12
    fig, ax = plt.subplots(figsize=(image_size * image_ratio_loL, image_size))
    ax.imshow(half_rink_img, extent=[-42.5, 42.5, 0, 100], aspect='auto')

    #norm = TwoSlopeNorm(vmin=-10, vcenter=0, vmax= 10)
    norm = colors.Normalize(vmin=-1, vmax=1)
    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap='bwr', norm=norm), ax=ax, orientation='vertical', fraction=0.02, pad=0.04)
    cbar.set_label('Différence de tirs moyens par heure', labelpad=15, rotation=270)

    ticks = [-1,-.5,0,.5,1]
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{t:.2f}" for t in ticks])

    #sns.kdeplot(data=array, fill=True)
    array = array/max
    array = gaussian_filter(array, sigma=1.5)
    ax.imshow(array, extent=[-42.5, 42.5, 0, 100], alpha=0.6, origin='lower', cmap='bwr', norm=norm)
    
    levels = np.linspace(-1, 1, 51)
    cf = ax.contourf(
        array,
        levels=levels,
        cmap='bwr',
        norm=norm,
        alpha=0.5,
        origin='lower',
        extent=[-42.5, 42.5, 0, 100]
    )

    contours = ax.contour(
        array,
        levels=levels,
        colors='black',
        linewidths=0.5,
        alpha=0.3,
        origin='lower',
        extent=[-42.5, 42.5, 0, 100]
    )

    zero_levels = [-0.1, 0.1]  # Adjust thickness of white line
    ax.contour(
        array,
        levels=zero_levels,
        colors='white',
        linewidths=1.2,
        origin='lower',
        extent=[-42.5, 42.5, 0, 100]
    )


    ax.set_xlim(-42.5, 42.5)
    ax.set_ylim(0, 100)
    ax.set_xlabel('Distance du centre de la patinoire (pieds)')
    ax.set_ylabel('Distance de la ligne centrale (pieds)')
    ax.legend()
    ax.set_title(title)

    if save_figure:
        plt.savefig(f"figures/{season}-{team}.png", bbox_inches='tight')
    #plt.show()
    fig = plt.gcf()
    return fig

def visualisation_avancee(start_season : int, end_season : int = -1, team : str = None, save_figure : bool = False) -> Figure:
    '''
    Function that creates the 'visualisation avancée' of the average number of shots for a given season and calculates the difference between a given team and the league's season average.
    Args:
        start_season: int representing the start season to analyze (e.g. 2016 for "2016-2017")
        end_season: int representing the end season to analyze (e.g. 2017 for "2016-2017"). If -1, only the start_season is analyzed.
        team: string representing the name of team to analyze (e.g. "Montreal Canadiens"). If None, only the league average is plotted.
    Returns: 
        None, but shows the heat map(s) as described above.
    '''
    df_shot = td.get_shot_events_by_season(start_season, end_season, bool_list=False, save=save_figure)
    df_avg = get_avg_shot_per_location_per_hour(df_shot)
    if team:
        df_team = df_shot[df_shot['event_owner_team_name'] == team].copy()
        if df_team.empty:
            print(f"Team {team} not found in the data for the given season(s). Please check the team name and try again.")
            return
        df_team_avg = get_avg_shot_per_location_per_hour(df_team)
        diff_array = compare_team_to_avg(team_df = df_team_avg, avg_df = df_avg, team=team, season = start_season)
        fig = plot_half_rink(diff_array, title = f"Difference de tir moyen par heure entre l'équipe {team} et la moyenne de la ligue", save_figure=save_figure, season=start_season, team=team)
        return fig
