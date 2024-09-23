import streamlit as st


def text_output():
    st.subheader('Text output')
    category_list = st.session_state.prompt_category_list

    for i, category in enumerate(category_list):
        # make container
        text_output_container(i)

def text_output_container(category_num):
    category_area, prompt_area, gauge_area = st.columns([1, 2.5, 2.5])
    with category_area:
        category_com = st.empty()
    with prompt_area:
        prompt_com = st.empty()
    with gauge_area:
        gauge_com = st.empty()

    st.session_state.final_text_output[category_num] = (category_com, prompt_com, gauge_com)
