import streamlit as st
from ui.component.video_input import video_input
from ui.component.video_output import video_output
from ui.component.text_output import text_output

def main_tab():
    col_video, col_text = st.columns(2, gap="medium")
    with col_video:
        st.subheader("Video Input")
        video_input()
    with col_text:
        st.subheader("Text Prompt Input")

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