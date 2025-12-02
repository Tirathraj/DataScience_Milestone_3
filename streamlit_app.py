import streamlit as st
import pandas as pd
from ift6758.client.serving_client import ServingClient
import ift6758.client.game_client as GameClient
import model_variables as mv

serving_client = ServingClient()

def download_model():
    try:
        serving_client.download_registry_model(
            workspace=st.session_state['workspace'], 
            model=st.session_state['model'], 
            version=st.session_state['version'])
        st.success("Model downloaded.")
    except Exception as e:
        st.error(f"Failed to download model: {str(e)}")

def update_model():
    download_model()
    if st.session_state['model'] == "Distance":
        st.session_state['features'] = ['distance_net']
    elif st.session_state['model'] == "Angle":
        st.session_state['features'] = ['angle_rad']
    elif st.session_state['model'] == "Distance_Angle":
        st.session_state['features'] = ['distance_net','angle_rad']
    else:
        st.error("Model not available")
        return
    fetch_game()

def seperate_preditcted_goals(predictions : list):
    preds = pd.Series(predictions).astype(float)
    goal_idx = preds[preds > mv.PREDICTION_THRESH].index
    is_pred_goal = preds.index.isin(goal_idx)
    is_home = st.session_state['game_df']["event_owner_team_name"] == st.session_state['game_df']["home_name"]
    is_away = st.session_state['game_df']["event_owner_team_name"] == st.session_state['game_df']["away_name"]
    home_goals = st.session_state['game_df'].loc[is_pred_goal & is_home]
    away_goals = st.session_state['game_df'].loc[is_pred_goal & is_away]
    return home_goals, away_goals

def predict_game():
    predicted_goals = serving_client.predict(st.session_state['game_df'][st.session_state['features']])
    st.session_state['game_df']['Model output'] = pd.Series(predicted_goals)
    st.session_state['home_pred_goals'], st.session_state['away_pred_goals'] = seperate_preditcted_goals(predicted_goals)

def fetch_game():
    try:
        st.session_state['game_df'] = GameClient.get_df_by_game(st.session_state['game_id'])
    except Exception as e:
        print("Failed to get game data " + str(e))
        st.error('Failed to get game data. Make sure this is a valid game id')
        return
    predict_game()

def initialize_data():
    st.session_state["game_df"] = GameClient.get_df_by_game(2025020001)
    st.session_state['game_id'] = 2025020001
    st.session_state['workspace'] = "IFT6758-2025-A10/Logistic Regression"
    st.session_state['model'] = "Distance"
    st.session_state['features'] = ['distance_net']
    st.session_state['version'] = "v1"
    download_model()
    fetch_game()

initialize_data()

# --------------------------------
#       Page frame
#---------------------------------

st.title("Hockey Visualization App")

with st.sidebar:
    st.session_state["workspace"] = st.selectbox(label="Workspace",options=mv.WORKSPACES, index=0)
    st.session_state["model"] = st.selectbox(label="Model", options=mv.MODELS, index=0)
    st.session_state["version"] = st.selectbox(label="Version", options=mv.VERSIONS, index=0)
    st.button("Get model", type="secondary", on_click=update_model)

with st.container():
    st.session_state["game_id"] = st.text_input("Game ID", st.session_state["game_id"])
    st.button("Ping game", type="secondary", on_click= fetch_game)

st.subheader(f"Game {st.session_state['game_id']}: {st.session_state['game_df'].iloc[0]['home_name']} vs {st.session_state['game_df'].iloc[0]['away_name']}")
with st.container():
    if st.session_state['game_df'].iloc[0]['game_state'] == 'LIVE':
        f"Period {st.session_state['game_df'].iloc[-1]['period_number']} - {st.session_state['game_df'].iloc[-1]['time_left']} left"
    left, right = st.columns(2)

    with left.container():
        st.metric(
            label=f"{st.session_state['game_df'].iloc[0]['home_name']} xG (actual)",
            value=f"{st.session_state['home_pred_goals'].shape[0]} ({st.session_state['game_df'].iloc[-1]['home_score']})",
            delta=st.session_state['home_pred_goals'].shape[0] - st.session_state['game_df'].iloc[-1]['home_score'],
            delta_color='off'
        )
    with right.container():
        st.metric(
            label=f"{st.session_state['game_df'].iloc[0]['away_name']} xG (actual)",
            value=f"{st.session_state['away_pred_goals'].shape[0]} ({st.session_state['game_df'].iloc[-1]['away_score']})",
            delta=st.session_state['away_pred_goals'].shape[0] - st.session_state['game_df'].iloc[-1]['away_score'],
            delta_color='off'
        )

with st.container():
    st.subheader("Data used for predictions (and predictions)")
    st.table(st.session_state['game_df'][['home_name', 'away_name', 'period_number', 'time_left', 'home_score', 'away_score', 'Model output']])