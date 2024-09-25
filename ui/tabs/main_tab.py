import streamlit as st
from ui.component.video_input import video_input
from ui.component.video_output import video_output
from ui.component.text_output import text_output
from ui.component.start_button import start_btn
import plotly.express as px
import pandas as pd

def similarity_graph_output():
    # Generate graph object
    graph = st.empty()
    st.session_state.graph_component = graph

def final_graph():
    final_graph_bt = st.button('View Full Graphs')
    if final_graph_bt:
        fig = px.line(pd.DataFrame(st.session_state.prompt_score_dict), title='Similarity Graph')
        st.plotly_chart(fig,use_container_width=True)

def main_tab():
    col_video, col_text = st.columns(2, gap="medium")
    with col_video:
        st.subheader("Video Input")
        video_input()
    with col_text:
        st.subheader("Text Prompt Input")
    st.divider()
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
    st.markdown(
        """<style>
            .stProgress > div > div > div > div {
                background-image: linear-gradient(#FF9999, #FF0000);
            }
        </style>""",
        unsafe_allow_html=True,
    )