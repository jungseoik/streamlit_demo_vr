import streamlit as st
import copy
from config import *

def init_session_states():
    # shared variables for multi thread
    if 'source_frame_com' not in st.session_state:
        st.session_state.source_frame_com = None
    if 'tiled_frame_com' not in st.session_state:
        st.session_state.tiled_frame_com = None
    if 'tiled_images' not in st.session_state:
        st.session_state.tiled_images = None
    if 'model_thread' not in st.session_state:
        st.session_state.model_thread = None
    if 'thread_enabled' not in st.session_state:
        st.session_state.thread_enabled = False

    # video
    if 'video_path' not in st.session_state:
        st.session_state.video_path = None
    if 'save_path' not in st.session_state:
        st.session_state.save_path = None
    if 'video_capture' not in st.session_state:
        st.session_state.video_capture = None

    # video parameter
    if 'frame_len' not in st.session_state:
        st.session_state.frame_len = None
    if 'frame_int' not in st.session_state:
        st.session_state.frame_int = None
    if 'tile_size' not in st.session_state:
        st.session_state.tile_size = None
    if 'tile_margin' not in st.session_state:
        st.session_state.tile_margin = None

    # model info
    if 'model_type' not in st.session_state:
        st.session_state.model_type = None
    if 'model_weight_path' not in st.session_state:
        st.session_state.model_weight_path = None
    if 'model' not in st.session_state:
        st.session_state.model = None

    # prompt info
    if 'prompt_category_list' not in st.session_state:
        st.session_state.prompt_category_list = copy.deepcopy(PROMPT_CATEGORY)
    if 'prompt_text_dict' not in st.session_state:
        st.session_state.prompt_text_dict = copy.deepcopy(PROMPT_TEXT)
    if 'prompt_range_init' not in st.session_state:
        st.session_state.prompt_range_init = copy.deepcopy(PROMPT_RANGE)
    if 'prompt_all_text_list' not in st.session_state:
        st.session_state.prompt_all_text_list = None

    # analyzer outputs
    if 'previous_score' not in st.session_state:
        st.session_state.previous_score = None
    if 'prompt_score_dict' not in st.session_state:
        st.session_state.prompt_score_dict = {}
    if 'final_text_output' not in st.session_state:
        st.session_state.final_text_output = {}
    if 'graph_component' not in st.session_state:
        st.session_state.graph_component = None
    if 'event_flag' not in st.session_state:
        for i in range(len(st.session_state.prompt_text_dict)):
            st.session_state[f'event_flag_{i}'] = False
    if 'alert_start_time' not in st.session_state:
        for i in range(len(st.session_state.prompt_text_dict)):
            st.session_state[f'alert_start_time_{i}'] = None
    if 'logs' not in st.session_state:
        st.session_state.logs = {}
    if 'log_number' not in st.session_state:
        st.session_state.log_number = 0
    if 'log_component' not in st.session_state:
        st.session_state.log_component = None
