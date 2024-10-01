import streamlit as st
from ui.component.video_input import video_input
from ui.component.video_output import video_output
from ui.component.text_output import text_output
from ui.component.start_button import start_btn
import plotly.express as px
import pandas as pd
from manage.state_init_manage import init_session_states

from utils.graph import similarity_graph_output



def sampling_count_output():
    st.session_state.frame_sampling_count = st.empty()


def main_tab():
    init_session_states()

    col_video, col_text = st.columns(2, gap="medium")
    with col_video:
        st.subheader("Video Input")
        video_input()
    with col_text:
        st.subheader("Text Prompt Input")

    st.divider()
    sampling_count_output()
    start_btn()
    st.divider()
    
    col_video_output, col_text_output = st.columns(2, gap="medium")

    with col_video_output:
        st.subheader("Video Output")
        video_output()

    with col_text_output:
        st.subheader("Visual Output")
        text_output()

    st.divider()

    similarity_graph_output()

    # Streamlit의 기본 프로그레스 바(progress bar) 색상을 커스텀 그라디언트 색상변경
    st.markdown(
        """<style>
            .stProgress > div > div > div > div {
                background-image: linear-gradient(#FF9999, #FF0000);
            }
        </style>""",
        unsafe_allow_html=True,
    )
