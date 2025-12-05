import streamlit as st
import pandas as pd
from ift6758.client.serving_client import ServingClient
import ift6758.client.game_client as GameClient
import model_variables as mv
import os

###### IMPORTANT #######
# Modification par Issam pour que le client pointe vers le service flask

serving_ip = os.getenv("SERVING_HOST", "127.0.0.1")
serving_client = ServingClient(ip=serving_ip)

# serving_client = ServingClient()

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
home = game_df.iloc[0]["home_name"]
away = game_df.iloc[0]["away_name"]

st.subheader(f"Game {st.session_state['game_id']}: {home} vs {away}")

with st.container():
    if game_df.iloc[0]['game_state'] == 'LIVE':
        f"Period {game_df.iloc[-1]['period_number']} - {game_df.iloc[-1]['time_left']} left"
    left, right = st.columns(2)

    with left.container():
        st.metric(
            label=f"{home} xG (actual)",
            value=f"{st.session_state['home_pred_goals'].shape[0]} ({game_df.iloc[-1]['home_score']})",
            delta=st.session_state['home_pred_goals'].shape[0] - game_df.iloc[-1]['home_score'],
            delta_color='off'
        )
    with right.container():
        st.metric(
            label=f"{away} xG (actual)",
            value=f"{st.session_state['away_pred_goals'].shape[0]} ({game_df.iloc[-1]['away_score']})",
            delta=st.session_state['away_pred_goals'].shape[0] - game_df.iloc[-1]['away_score'],
            delta_color='off'
        )

with st.container():
    st.subheader("Data used for predictions (and predictions)")
    st.table(game_df[['event_owner_team_name', 'home_name', 'away_name', 'period_number', 'time_left', 'distance_net','angle_rad', 'event_type', 'Model output']])

with st.container():
    st.write("This is the bonus part. We use the advanced visualization part from Milestone 1 where we compared the average shot per hour of a given team to the rest of the league. \n \
             Here, we plot the average shot per hour (per location) of each team compared to the rest of the league for the season where the match occured. This gives insight on how the different teams play their offense and how their playstyles match eachother.")
    fig_home = GameClient.get_avg_shot_per_hour_fig(get_game_year(st.session_state["game_id"]), game_df.iloc[0]["home_name"])
    fig_away = GameClient.get_avg_shot_per_hour_fig(get_game_year(st.session_state["game_id"]), game_df.iloc[0]["away_name"])
    left, right = st.columns(2)

    st.subheader(f"Shot per hour per location of each team this year")
    with left.container():
        st.write(f"{game_df.iloc[0]["home_name"]}")
        st.pyplot(fig_home)
    with right.container():
        st.write(f"{game_df.iloc[0]["away_name"]}")
        st.pyplot(fig_away)