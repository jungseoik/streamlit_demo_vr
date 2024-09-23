import streamlit as st
from ui.tabs.main_tab import main_tab
from ui.tabs.setting_tab import setting_tab

from utils.state_init import init_session_states

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
    layout='wide',
    initial_sidebar_state ='collapsed',
)


st.title('Devmacs-Demo v0.0')
st.write("데모 페이지입니다. 👋")
init_session_states()


tab_main, tab_setting = st.tabs(["Main", "Settings"])

with tab_main:
     main_tab()

with tab_setting:
    st.subheader("Settings")
    setting_tab()

