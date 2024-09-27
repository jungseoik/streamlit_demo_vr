import streamlit as st
from manage.thread_manage import ThreadManager

def kill_model():
    if st.session_state.model is not None:
        del st.session_state.model
        st.session_state.model = None

def kill_model_thread(thread_var: ThreadManager):
    if thread_var.thread is not None:
        thread_var.thread_enabled = False
        thread_var.thread.join()
        thread_var.thread = None

    capture = st.session_state.video_capture
    if capture is not None:
        capture.release()
        st.session_state.video_capture = None
