import streamlit as st
from ui.component.video_input import video_input
from ui.component.video_output import video_output
from ui.component.text_output import text_output
from ui.component.start_button import start_btn
import plotly.express as px
import pandas as pd
from manage.state_init_manage import init_session_states

def similarity_graph_output():
    # Generate graph object
    graph = st.empty()
    st.session_state.graph_component = graph

def final_graph():
    final_graph_bt = st.button('View Full Graphs')
    if final_graph_bt:
        fig = px.line(pd.DataFrame(st.session_state.prompt_score_dict), title='Similarity Graph')
        st.plotly_chart(fig,use_container_width=True)

def test_output():
    st.session_state.tiled_frame_com = st.empty()


def main_tab():
    init_session_states()
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
            # 빈 그래프 컴포넌트를 선언 (빈 컨테이너)
        test_output()
        # text_output()

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