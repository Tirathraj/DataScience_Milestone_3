import streamlit as st
import pandas as pd
import numpy as np
from ift6758.client.serving_client import ServingClient
import ift6758.client.game_client as GameClient
import model_variables as mv
import os

import matplotlib.pyplot as plt

###### IMPORTANT #######
# Modification par Issam pour que le client pointe vers le service flask

serving_ip = os.getenv("SERVING_HOST", "127.0.0.1")
serving_client = ServingClient(ip=serving_ip)

# serving_client = ServingClient()

#Tirath bonus part
def cumulative_xg(game_df: pd.DataFrame):
    """
    Calculates cumulative xG over time for both teams.

    Returns:
        home_df: df with event_index, time, cumulative_xg for home
        away_df: df with event_index, time, cumulative_xg for away
    """

    # Determines time index
    game_df = game_df.copy()
    game_df["idx"] = range(len(game_df))

    # Gets total elapsed time in seconds
    game_df["time_sec"] = convert_game_time_seconds(game_df)

    # team names
    home = game_df.iloc[0]["home_name"]
    away = game_df.iloc[0]["away_name"]

    home_mask = game_df["event_owner_team_name"] == home
    away_mask = game_df["event_owner_team_name"] == away

    # Get row data
    home_df = game_df.loc[home_mask, ["time_sec", "Model output"]].copy()
    away_df = game_df.loc[away_mask, ["time_sec", "Model output"]].copy()

    # cumulative xG for both teams
    home_df["cumulative_xg"] = home_df["Model output"].cumsum()
    away_df["cumulative_xg"] = away_df["Model output"].cumsum()

    return home_df, away_df

import matplotlib.pyplot as plt
import pandas as pd

def extract_actual_data(game_df, home_name, away_name):
        """
        Get the data of the actual goals for the plotting step
        """
        
        teams = [home_name, away_name]
        game_df["time_sec"] = convert_game_time_seconds(game_df)
        
        # Determine the end of the game (max time) to extend the plot lines to the finish
        max_time = game_df['time_sec'].max()

        data = [[[0],[0]],[[0],[0]]]
        for i in range(2):
            team_goals = game_df[
                (game_df['event_owner_team_name'] == teams[i]) & 
                (game_df['event_type'] == 'goal')
            ].sort_values('time_sec')
            
            for t in team_goals['time_sec']:
                data[i][0].append(t)                 # time of goal
                data[i][1].append(data[i][1][-1] + 1)  # resulting score
                
            if data[i][0][-1] < max_time:
                data[i][0].append(max_time)
                data[i][1].append(data[i][1][-1])
        home_data = data[0]
        away_data = data[1]
        return np.array(home_data), np.array(away_data)

def plot_cumulative_xg(home_df: pd.DataFrame, away_df: pd.DataFrame, home_actual_data : list, away_actual_data : list, home_name: str, away_name: str):
    """
    Plots cumulative xG over time for both teams.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot teams xG
    #Convert seconds into minutes by dividing by 60 for display
    
    ax.plot(home_df["time_sec"]/60, home_df["cumulative_xg"], label=f"Home: {home_name} xG",color="red")
    ax.plot(away_df["time_sec"]/60, away_df["cumulative_xg"],label=f"Away: {away_name} xG",color="blue")
    ax.step(home_actual_data[0]/60, home_actual_data[1], where='post', label=f"{home_name} actual", linestyle='--',color="red")
    ax.step(away_actual_data[0]/60, away_actual_data[1], where='post', label=f"{away_name} actual", linestyle='--',color="blue")

    ax.set_xlabel("Elapsed Time (minutes)")
    ax.set_ylabel("Cumulative xG")
    ax.set_title("Cumulative xG over Time")
    ax.grid()
    ax.legend()

    return fig

def convert_game_time_seconds(game_df: pd.DataFrame) -> pd.Series:
    """
    Converts (period_number, time_left MM:SS) into seconds elapsed for game

    Example:
        Period 1 time_left = 20 -> 0 seconds
        Period 2 time_left = 20 -> 7200 seconds
    """
    df = game_df.copy()

    time = []
    for _, row in df.iterrows():
                
        PERIOD_LENGTH = 20 * 60  # 1200 seconds
        
        period = int( row["period_number"] )
        
        mins, secs = map(int, row["time_left"].split(":") )
        seconds_left = (mins * 60) + secs
        
        # Time elapsed inside the current period
        period_elapsed = PERIOD_LENGTH - seconds_left

        # Total time since game start
        total_time_elapsed = ( (period - 1) * PERIOD_LENGTH ) + period_elapsed
        
        time.append(total_time_elapsed)

    return pd.Series(time, index = game_df.index)
    
def highlight_xg_cell(val):
    
    """
    Highlight xG cells based on value:
    0.10+ -> green (High chance)
    0.05+ -> yellow (Medium chance)
    """
    x = float(val)

    if x > 0.10:
        return "background-color: lightgreen"
    elif x > 0.05:
        return "background-color: yellow"
    else:
        return ""

#Simon Main + Bonus Part

def get_game_year(game_id : int):
    return int(str(game_id)[:4])

def download_model():
    try:
        serving_client.download_registry_model(
            workspace=st.session_state['workspace'], 
            model=st.session_state['model'], 
            version=st.session_state['version'])
        st.success("Model downloaded.")
    except Exception as e:
        st.error(f"Failed to download model: {str(e)}")
        return False
    
    model = st.session_state['model']
    if model == "Distance":
        st.session_state['features'] = ['distance_net']
    elif model == "Angle":
        st.session_state['features'] = ['angle_rad']
    elif model == "Distance_Angle":
        st.session_state['features'] = ['distance_net','angle_rad']
    else:
        st.error("Model not available")
        return False
    return True

def seperate_preditcted_goals(predictions : pd.Series):
    game_df = st.session_state['game_df']
    preds = predictions.astype(float)
    thresh = mv.PREDICTION_THRESH
    mask = preds > thresh
    home_mask = game_df["event_owner_team_name"] == game_df["home_name"]
    away_mask = game_df["event_owner_team_name"] == game_df["away_name"]
    home_goals = game_df[mask & home_mask]
    away_goals = game_df[mask & away_mask]
    return home_goals, away_goals

def predict_game(start_index : int = 0):
    game_df = st.session_state['game_df']
    features = st.session_state["features"]

    to_predict = game_df.loc[start_index:, features]

    if to_predict.empty:
        return
    
    preds = pd.Series(serving_client.predict(to_predict), index=to_predict.index)
    st.session_state["Model output"] = pd.concat([st.session_state['Model output'], preds], ignore_index=True)
    st.session_state['home_pred_goals'], st.session_state['away_pred_goals'] = seperate_preditcted_goals(st.session_state['Model output'])

def fetch_game():
    print(f"Fetching game {st.session_state['game_id']}")
    game_id = st.session_state["game_id"]
    prev_id = st.session_state.get("previous_game_id")
    try:
        game_df = GameClient.get_df_by_game(game_id)
    except Exception as e:
        print("Failed to get game data " + str(e))
        st.warning('Failed to get game data. Make sure this is a valid game id. Returning to previous game')
        return
    if game_df is None:
        game_df = GameClient.get_df_by_game(prev_id)
    
    st.session_state['game_df'] = game_df
    prev_len = st.session_state.get("prev_game_length", 0)
    new_len = game_df.shape[0]

    if game_id != prev_id:
        print(f"Calculating predictions for new game")
        predict_game(0)
    elif game_df.iloc[0]["game_state"] == "LIVE" and new_len > prev_len:
        print(f"Calculating predictions for rest of the live game")
        predict_game(prev_len)
    st.session_state['game_df']['Model output'] = st.session_state["Model output"]
    st.session_state['previous_game_id'] =game_id
    st.session_state['prev_game_length'] = new_len
    return
    
    

def initialize_data():
    print("Initializing streamlit state")
    st.session_state['game_id'] = 2024020001
    st.session_state['workspace'] = "IFT6758-2025-A10/Logistic Regression"
    st.session_state['model'] = "Distance"
    st.session_state['features'] = ['distance_net']
    st.session_state['version'] = "v1"
    st.session_state['previous_game_id'] = None
    st.session_state['prev_game_length'] = 0 # Since initialization game is not live
    st.session_state["Model output"] = pd.Series(dtype=float)
    download_model()
    fetch_game()

if "game_df" not in st.session_state: initialize_data()

# --------------------------------
#       Page frame
#---------------------------------

st.title("Hockey Visualization App")

with st.sidebar:
    st.session_state["workspace"] = st.selectbox(label="Workspace",options=mv.WORKSPACES, index=0)
    st.session_state["model"] = st.selectbox(label="Model", options=mv.MODELS, index=0)
    st.session_state["version"] = st.selectbox(label="Version", options=mv.VERSIONS, index=0)
    st.button("Get model", type="secondary", on_click=lambda: (download_model() and fetch_game()))

with st.container():
    st.session_state["game_id"] = st.text_input("Game ID", st.session_state["game_id"])
    st.button("Ping game", type="secondary", on_click= fetch_game)

game_df = st.session_state["game_df"]
home_name = game_df.iloc[0]["home_name"]
away_name = game_df.iloc[0]["away_name"]

# Cumulative xG (Tirath)
st.subheader("Cumulative xG Progression Over Time")

# Calculate cumulative xG for both teams
home_xg_df, away_xg_df = cumulative_xg(game_df)
home_actual_df = game_df[(game_df['event_owner_team_name'] == home_name) & (game_df['event_type'] == 'goal')]
away_actual_df = game_df[(game_df['event_owner_team_name'] == away_name) & (game_df['event_type'] == 'goal')]

home_actual_data, away_actual_data = extract_actual_data(game_df=game_df, home_name=home_name, away_name=away_name)

#Cumulative scores are the last row values for both teams
cumulative_home_xg = home_xg_df["cumulative_xg"].iloc[-1] #series indexing
cumulative_away_xg = away_xg_df["cumulative_xg"].iloc[-1]

# End of cumulative xG 


st.subheader(f"Game {st.session_state['game_id']}: {home_name} vs {away_name}")

with st.container():
    if game_df.iloc[0]['game_state'] == 'LIVE':
        f"Period {game_df.iloc[-1]['period_number']} - {game_df.iloc[-1]['time_left']} left"
    left, right = st.columns(2)

    with left.container():
        st.metric(
            label=f"{home_name} xG (actual)",
            #value=f"{st.session_state['home_pred_goals'].shape[0]} ({game_df.iloc[-1]['home_score']})",
            value=f"{cumulative_home_xg:.2f} ({game_df.iloc[-1]['home_score']})",
            delta=round(cumulative_home_xg,2) - game_df.iloc[-1]['home_score'],
            delta_color='off'
        )
    with right.container():
        st.metric(
            label=f"{away_name} xG (actual)",
            #value=f"{st.session_state['away_pred_goals'].shape[0]} ({game_df.iloc[-1]['away_score']})",
            value=f"{cumulative_away_xg:.2f} ({game_df.iloc[-1]['away_score']})",
            delta= round(cumulative_away_xg,2) - game_df.iloc[-1]['away_score'],
            delta_color='off'
        )
    
    st.write("This is our bonus contribution. We plotted the goals (predicted and actual) as a function of time simply by getting the time stamps of each true goal and plotting the cumulative expected goal with their time stamps. \
             This helps to get an idea of the performance of each team and to see how many \"unlikely\" goals were made. \
             Game 2025020425 is a good example of a game where the Capitals scored many more goals than expected.")
    # Plot
    fig = plot_cumulative_xg(home_xg_df, away_xg_df, home_actual_data, away_actual_data, home_name, away_name)
    st.pyplot(fig)

with st.container():
    st.subheader("Data used for predictions (and predictions)")

    #Tirath Part
    #Add columns for nicer display
    game_df['event_owner'] = game_df['event_owner_team_name']
    game_df['period'] = game_df['period_number']
    game_df["distance"] = game_df["distance_net"]
    game_df["Predicted xG"] = game_df["Model output"]
    
    cols = ['event_owner','period', 'time_left','distance', 'angle_rad','event_type', 'Predicted xG']
    
    styled_df = game_df[cols].style.applymap(highlight_xg_cell, subset=['Predicted xG'])

    st.dataframe(styled_df)
    #End of tirath
    
    #Simon part
    #st.table(game_df[['event_owner_team_name', 'home_name', 'away_name', 'period_number', 'time_left', 'distance_net','angle_rad', 'event_type', 'Model output']])

#with st.container():
#    #Tirath
#    st.write(f"Bonus part 1: We calculate cumulative xG for both teams over all shot events and plot a graph of cumulative xG over time. Huge jumps in such graps tend to highlight big chances such as goals, or a sustained time period of attacking play.")
#    #Simon
#    st.write("This is the bonus part. We use the advanced visualization part from Milestone 1 where we compared the average shot per hour of a given team to the rest of the league. \n \
#             Here, we plot the average shot per hour (per location) of each team compared to the rest of the league for the season where the match occured. This gives insight on how the different teams play their offense and how their playstyles match eachother.")    
#    fig_home = GameClient.get_avg_shot_per_hour_fig(get_game_year(st.session_state["game_id"]), game_df.iloc[0]["home_name"])
#    fig_away = GameClient.get_avg_shot_per_hour_fig(get_game_year(st.session_state["game_id"]), game_df.iloc[0]["away_name"])
#    left, right = st.columns(2)
#
#    st.subheader(f"Shot per hour per location of each team this year")
#    with left.container():
#        st.write(f"{game_df.iloc[0]['home_name']}")
#        st.pyplot(fig_home)
#    with right.container():
#        st.write(f"{game_df.iloc[0]['away_name']}")
#        st.pyplot(fig_away)