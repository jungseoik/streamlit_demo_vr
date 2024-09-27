import streamlit as st
from manage.thread_manage import ThreadManager
from manage.model_manage import load_model, download_model, prepare_model
import cv2
from typing import Optional
import threading
from collections import deque
from queue import Queue
import plotly.express as px

import time
import torch
import numpy as np
from typing import List
from config import *
from typing import Tuple

import os
import warnings

from datetime import datetime
from pathlib import Path 
import torch.nn.functional as F
from utils.kill import kill_model, kill_model_thread
from utils.thread.inference_thread import start_model_thread
from utils.thread.video_thread import video_processing_thread

def softmax(x):
    y = np.exp(x - np.max(x))
    return y / np.sum(y)


def start_process(thread_var: ThreadManager, model_type, weight_path):
    
    # check video
    video_path = st.session_state.video_path
    if not video_path:
        print('error: video is not uploaded')
    
    # kill thread if running
    if thread_var.thread_enabled or st.session_state.model_thread is not None:
        kill_model_thread(thread_var)

    # kill model if changed
    if model_type != st.session_state.model_type or weight_path != st.session_state.model_weight_path:
        st.session_state.model_type = model_type
        st.session_state.model_weight_path = weight_path
        kill_model(thread_var)

    # load model
    model = prepare_model()

    thread_var.model = model

    # encode all prompts and save
    texts = st.session_state.prompt_all_text_list
    if texts:
        thread_var.text_vectors = thread_var.model(text = texts)
        print("텍스트 벡터 쉐입",thread_var.text_vectors.shape)

    # get video settings
    frame_len = st.session_state.frame_len
    frame_int = st.session_state.frame_int
    tile_size = st.session_state.tile_size
    tile_margin = st.session_state.tile_margin
    
    # make Video capture
    video_path = st.session_state.video_path
    delay_state = not isinstance(video_path, str) or not video_path.startswith("rtsp")

    video_capture = cv2.VideoCapture(video_path)
    st.session_state.video_capture = video_capture

    fps = video_capture.get(cv2.CAP_PROP_FPS)
    counter = 0
 # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # start model thread
    start_model_thread(model, thread_var)
    video_processing_thread(video_capture, thread_var, fps, frame_int, tile_size, tile_margin, frame_len, MAX_WIDTH, MAX_HEIGHT, delay_state)



def record_video(queue:Queue, save_dir_path:str, fps):
    frame = queue.get() 
    h,w = frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    frame_cnt = 0 
    Path(save_dir_path).mkdir(exist_ok=True, parents=True)
    while True :
        if frame_cnt % int(fps * 60 * 30) == 0 :  # 30분 마다 저장 
            save_file_name = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
            save_path = os.path.join(save_dir_path, save_file_name + ".avi")
            vid_out = cv2.VideoWriter(save_path, fourcc=fourcc, 
                              fps=fps, frameSize=(w,h))
            frame_cnt = 1
        
        frame = queue.get()
        vid_out.write(frame)
        frame_cnt += 1


def start_btn():
    model_type = st.session_state.model_type
    model_weight_path = st.session_state.model_weight_path

    st.markdown(
        """
        <style>
        button {
            height: auto;
            padding-top: 20px !important;
            padding-bottom: 20px !important;
            width: 100% !important; /* 버튼을 가로로 길게 늘리기 */
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    start_btn = st.button(f"# Start", key="st_btn_run", type="primary", disabled=st.session_state.thread_enabled)
    
    if start_btn:
        # Initializing the session state
        st.session_state.prompt_score_dict = {}
        # Start process
        thread_var = ThreadManager()
        start_process(thread_var, model_type, model_weight_path)

