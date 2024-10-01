import streamlit as st

def video_output():
    # st.subheader('Source Frame')
    st.session_state.video_output_frame = st.empty()

    # st.subheader('Tiles')
    # st.session_state.frame_sampling_count = st.empty()

    # TODO : gradCAM visualize