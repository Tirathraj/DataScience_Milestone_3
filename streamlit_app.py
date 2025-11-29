import streamlit as st
import pandas as pd
import numpy as np

st.title("Hockey Visualization App")

with st.sidebar:
    # TODO: Add input for the sidebar
    workspace = st.selectbox(
        "Workspace",
        ("Home", "Work", "Train"),
        )
    model = st.selectbox(
        "Model",
        ("Distance", "Angle", "Distance + Angle"),
    )
    version = st.selectbox(
        "Version",
        ("1", "2"),
    )
    st.button("Get model", type="secondary")

with st.container():
    # TODO: Add Game ID input
    game_id = st.selectbox(
        "Game ID",
        ("1", "2"),
    )
    st.button("Ping game", type="secondary")

st.subheader("Game {ID}: {Team_1} vs {Team_2}")
with st.container():
    # TODO: Add Game info and predictions
    "Period {period_num} - {time} left"
    left, right = st.columns(2)
    with left.container():
        "{team_1} xG (actual)"
    with right.container():
        "{team_2} xG (actual)"
    pass

with st.container():
    # TODO: Add data used for predictions
    st.subheader("Data used for predictions (and predictions)")
    pass