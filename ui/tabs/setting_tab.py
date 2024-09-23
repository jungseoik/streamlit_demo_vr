import streamlit as st
from ui.component.param import video_param_input, model_select_input , text_param_input

def setting_tab():
    col_video_param, col_text_parma = st.columns(2, gap="medium")
    with col_video_param:
        video_param_input()
        model_select_input()

    with col_text_parma:
        text_param_input()

