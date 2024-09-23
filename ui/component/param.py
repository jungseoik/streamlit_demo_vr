import streamlit as st
from config import *

def video_param_input():
    frame_len = st.number_input("Frame length", key="st_frame_len", value=6, min_value=1, max_value=24)
    frame_int = st.number_input("Frame interval", key="st_frame_int", value=.5, min_value=.1, max_value=5., step=.1)
    tile_size = st.number_input("Tile size", value=448)
    tile_margin = st.number_input("Tile margin", value=0.25, step=0.05)

    st.session_state.frame_len = frame_len
    st.session_state.frame_int = frame_int
    st.session_state.tile_size = tile_size
    st.session_state.tile_margin = tile_margin

def model_select_input():
    model_options = [f"{m} | {w}" for m in MODEL_ASSETS for w in MODEL_ASSETS[m]]
    model_selected = st.selectbox("Model", key="st_model_name_weight", options=model_options)
    model_type, model_weight_path = (m.strip() for m in model_selected.split('|'))
    st.session_state.model_type = model_type
    st.session_state.model_weight_path = model_weight_path

def text_param_input():
    category_list = st.session_state.prompt_category_list
    tabs = st.tabs(["Category"] + category_list + ["Settings"])

    pr_mins = st.session_state.prompt_range_init.get(CAM_ID, None)

    # prompt category setting
    with tabs[0]:
        n_cate = st.number_input("number of category", key="st_num_category", value=len(category_list), min_value=1)

        cate_temp = category_list[:n_cate]
        cate_temp = cate_temp + [""] * (n_cate - len(cate_temp))
        for k in range(n_cate):
            cate_temp[k] = st.text_input("category name", key=f"st_category_name{k+1}", value=cate_temp[k])

        save_category = st.button("Apply", key="st_apply_category")
        if save_category:
            st.session_state.prompt_category_list = cate_temp
            st.rerun()

    # prompt text setting
    text_dict = st.session_state.prompt_text_dict
    for i in range(len(category_list)):
        text_list = text_dict.setdefault(i, [""])
        with tabs[i+1]:
            n_text = st.number_input("number of prompt", key=f"st_num_text{i+1}", value=len(text_list), min_value=1)

            text_temp = text_list[:n_text]
            text_temp = text_temp + [""] * (n_text - len(text_temp))
            for j in range(n_text):
                text_temp[j] = st.text_input("prompt", key=f"st_text{i+1}_{j+1}", value=text_temp[j])

            save_text = st.button("Apply", key=f"st_apply_text{i+1}")
            if save_text:
                text_dict[i] = text_temp
                st.rerun()

    with tabs[-1]:
        for i, key in enumerate(text_dict):
            st.info(category_list[i])
            for j in range(len(text_dict[key])):
                prompt_area, min_gauge, max_gauge = st.columns([3,1,1])
                with prompt_area:
                    st.write(text_dict[key][j])
                with min_gauge:
                    st.number_input('Min', key=f'category_{i}_prompt{j}_min', value=pr_mins[i][j] if pr_mins else 0.25)
                with max_gauge:
                    st.number_input('Max', key=f'category_{i}_prompt{j}_max', value=pr_mins[i][j] + 0.03 if pr_mins else 0.3)

    # write to session state
    st.session_state.prompt_all_text_list = [text for k in range(len(category_list)) for text in text_dict[k]]

