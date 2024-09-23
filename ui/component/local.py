import streamlit as st
from config import *
import os
import tempfile

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

def local_input() -> None:
    """
    Streamlit 파일 업로드 컴포넌트를 사용하여 동영상 파일을 업로드 받고, 
    세션 상태에 파일 경로를 저장하는 함수.

    Returns:
        None: Streamlit을 사용한 UI 요소 생성 및 세션 상태 업데이트.
    """
    video_file_path = st.file_uploader("동영상 파일 업로드", key="st_video_file_path", type=["mp4", "avi"], 
                                       accept_multiple_files=False)
    print("video file", type(video_file_path))

    if video_file_path:
        video_file = temp_file_save(video_file_path)
        st.session_state.video_path = video_file
        print("세션 비디오", st.session_state.video_path)
    else:
        video_file = None
        st.session_state.video_path = None
