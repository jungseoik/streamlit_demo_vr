import streamlit as st

def video_output():
    st.subheader('Source Frame')
    st.session_state.source_frame_com = st.empty()

    st.subheader('Tiles')
    st.session_state.tiled_frame_com = st.empty()

    # TODO : gradCAM visualize