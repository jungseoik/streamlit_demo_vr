import streamlit as st
from config import *
from ui.component.local import local_input
from ui.component.rtsp import rtsp_input

def video_input()-> None:

    select_input = st.radio(
            "Video input 형태를 명시해주세요",
    ["RTSP", "Local"],
    captions=[
        "RTSP 주소를 입력해주세요",
        "비디오 파일을 입력해주세요",
    ],
    horizontal=True
    )

    if select_input == "RTSP":
        rtsp_input()
    elif select_input == "Local":
        local_input()
    
    