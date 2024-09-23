import streamlit as st
from config import *
import cv2 

def check_rtsp_url(path: str) -> bool:
    """
    RTSP URL을 받아 해당 URL의 연결 가능 여부를 확인하는 함수.

    Args:
        path (str): RTSP 스트림 경로. 기본 형식은 'username:password@ip_address/stream_path'.
    
    Returns:
        bool: URL이 유효하면 True, 유효하지 않으면 False를 반환.
    """
    video_path = "rtsp://" + path
    if cv2.VideoCapture(video_path).isOpened():
        print("RTSP url has been detected")
        return True
    else:
        print("CHECK RTSP URL")
        return False

def rtsp_input() -> None:
    """
    Streamlit 텍스트 입력 필드를 통해 사용자가 입력한 RTSP 경로를 체크하는 UI

    Returns:
        None: Streamlit 컴포넌트를 사용하여 UI 요소를 생성하고, 성공/경고 메시지를 표시.
    """
    rtsp_url = st.text_input(label="RTSP 경로", key="st_rtsp_path", 
                             placeholder="admin:admin@192.168.1.xxx/stream1", value=DEFAULT_RTSP)
    
    if rtsp_url:
        ret = check_rtsp_url(rtsp_url)
        if ret:
            video_file = "rtsp://" + rtsp_url
            st.success(body="올바른 주소입니다.", icon="✨")
        else:
            video_file = None
            st.warning(body="CHECK RTSP URL, PLEASE", icon="🚨")
