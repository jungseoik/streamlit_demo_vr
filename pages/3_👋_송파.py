import streamlit as st
import random
import datetime
import streamlit as st
from config import *
import os
import tempfile
import streamlit as st
import time
import cv2

st.title(':sparkles:로또 생성기:sparkles:')

def temp_file_save(file_upload_obj:  st.runtime.uploaded_file_manager.UploadedFile) -> str:
    """
    업로드된 파일을 임시 디렉토리에 저장하고 파일 경로를 반환하는 함수.
    동일한 파일을 다시 업로드 할 경우 체크 한 뒤 해당 파일 활용

    Args:
        file_upload_obj ( st.runtime.uploaded_file_manager.UploadedFile): Streamlit에서 업로드된 파일 객체.
    
    Returns:
        str: 임시 디렉토리에 저장된 파일의 경로.
    """
    temp_video_file_path = os.path.join(tempfile.gettempdir(), file_upload_obj.name)
    if not os.path.exists(temp_video_file_path):
        with open(temp_video_file_path, 'wb') as f:
            f.write(file_upload_obj.getvalue())
            print(f"video file has been copied to {temp_video_file_path}")
    
    return temp_video_file_path


video_file_path = st.file_uploader("동영상 파일 업로드", key="st_video_file_path", 
                                   type=["mp4", "avi"], 
                                    accept_multiple_files=False)
print("video file", type(video_file_path))
print("video file", video_file_path)

st.session_state.source_frame_com = st.empty()


# with st.empty():
#     for seconds in range(60):
#         st.write(f"⏳ {seconds} seconds have passed")
#         time.sleep(0.1)
#     st.write("✔️ 1 minute over!")



# placeholder = st.empty()
# Replace the placeholder with some text:
# placeholder.text("Hello")
# # Replace the text with a chart:
# placeholder.line_chart({"data": [1, 5, 2, 6]})

# Replace the chart with several elements:
# with placeholder.container():
#     st.write("This is one element")
#     st.write("This is another")

# # Clear all those elements:
# placeholder.empty()

if video_file_path:
    video_file = temp_file_save(video_file_path)
    st.session_state.video_path = video_file
    print("세션 비디오", st.session_state.video_path)

import streamlit as st
import cv2
import numpy as np
import tempfile

# Streamlit 앱 헤더
st.title("비디오 업로드 및 재생")

# 비디오 업로드 위젯
uploaded_file = st.file_uploader("비디오 파일을 업로드하세요", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    # 업로드된 파일을 임시 파일로 저장
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    # OpenCV를 사용하여 비디오 파일 열기
    cap = cv2.VideoCapture(tfile.name)

    # 비디오가 열렸는지 확인
    if cap.isOpened():
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("break!!!!!!!!!")
                break
            # 프레임을 RGB로 변환 (Streamlit에서 이미지를 출력하기 위해)
            # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # st.session_state.source_frame_com.image(frame_rgb, use_column_width=True)
            frame = cv2.resize(frame, (854, 480 ), interpolation=cv2.INTER_LINEAR)
            st.session_state.source_frame_com.image(frame, use_column_width=True)

            time.sleep(0.1)
            # st.image(frame_rgb)



    cap.release()
else:
    st.write("비디오 파일을 업로드하세요.")