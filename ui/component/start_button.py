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

from queue import Queue
import cv2 
from datetime import datetime
from pathlib import Path 
import torch.nn.functional as F
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


def softmax(x):
    y = np.exp(x - np.max(x))
    return y / np.sum(y)


def run_model(model: object, frames: torch.Tensor, thread_manager: ThreadManager) -> Tuple[np.ndarray, np.ndarray]:
    """
    모델을 실행하여 프레임 벡터와 텍스트 벡터 간 유사도를 계산합니다.

    Args:
        model (object): 사용할 모델 객체
        frames (torch.Tensor): 처리할 프레임 텐서
        thread_manager (ThreadManager): 상태 관리를 담당하는 ThreadManager 인스턴스

    Returns:
        Tuple[np.ndarray, np.ndarray]: 유사도 점수 배열과 소프트맥스 결과 배열
    """
    print(f"frames type: {type(frames)}")
    print(f"frames dtype: {frames.dtype}")
    print(f"frames shape: {frames.shape}")
    print(f"frames device: {frames.device}")
    frames = frames.cuda() 
    print(f"frames device???: {frames.device}")
    
     # frames를 GPU로 이동
    txt_vectors = thread_manager.text_vectors  # 텍스트 벡터를 ThreadManager로부터 가져옴

    vid_vector = thread_manager.model(video = frames)
    print("비디오 벡터:" ,vid_vector.shape, "  텍스트 벡터 :", txt_vectors.shape)
    sim_scores = thread_manager.model.model._loose_similarity(sequence_output=txt_vectors, visual_output=vid_vector)
    print(f"Original shape: {sim_scores.shape}") 
    print(f"Original shape: {sim_scores.device}") 
    print(f"Original shape: {type(sim_scores)}") 
    print(f"Original shape: {sim_scores.dtype}") 

    # 최대값 계산
    max_values, max_indices = sim_scores.max(dim=1)
    print(f"Max values: {max_values.shape}") 

    softmax_values = F.softmax(max_values, dim=0)
    print(f"Max values22222: {softmax_values.shape}") 

    # 텐서를 CPU로 이동 후 NumPy 배열로 변환
    sim_scores_np = sim_scores.cpu().numpy()
    softmax_values_np = softmax_values.cpu().numpy()

    # NumPy 배열로 변환된 값 리턴
    return sim_scores_np, softmax_values_np

def model_loop_thread(thread_manager: ThreadManager) -> None:
    """
    모델 루프 스레드 함수.
    프레임을 가져와 모델을 실행하고, 유사도 점수와 소프트맥스 결과를 업데이트합니다.

    Args:
        thread_manager (ThreadManager): 스레드와 상태 관리를 담당하는 ThreadManager 인스턴스

    Returns:
        None
    """
    prompts_list = [prompt for category_id, prompt_list in PROMPT_TEXT.items() for prompt in prompt_list]
    print("여기부터 모델 스레드 로그 시작입니다! ")

    while thread_manager.thread_enabled:
        # if thread_manager.frame_queue_updated:
        #     thread_manager.frame_queue_updated = False
        # else:
        #     time.sleep(0.02)  # 프레임이 업데이트될 때까지 대기
        #     continue

        # 프레임 큐 업데이트 이벤트 대기
        thread_manager.frame_queue_updated_event.wait()

        with thread_manager.thread_lock:
            # 큐에서 프레임 가져오기
            frames = torch.tensor(np.array(thread_manager.frame_queue))
            print("여기 차원 봐야돼! ",frames.shape)

        # 텐서 차원 순서 변경 (batch, channel, height, width, frames)
        frames = frames.permute((1, 0, 2, 3, 4)).contiguous()
        
        # 모델 실행
        sim_scores, sim_softmax = run_model(thread_manager.model, frames, thread_manager)
        print("어쩔껀데")
        print(sim_scores.shape)
        print(sim_softmax.shape)
        # 각 프롬프트와 해당 유사도 점수 로그 기록
        sim_scores_list = list(sim_scores)
        # demo_sim_logger.log_sim_scores(prompts_list, sim_scores_list)

        with thread_manager.thread_lock:
            # 유사도 점수와 소프트맥스 결과 업데이트
            thread_manager.out_sim_scores = sim_scores
            thread_manager.out_sim_softmax = sim_softmax
            print("lock놓는다!")


        # 유사도 점수 그래프 업데이트
        # update_graph(thread_manager.out_sim_scores)
        # 모델 처리 완료를 알림
        thread_manager.frame_queue_updated_event.clear()
        thread_manager.model_processing_done_event.set()

        time.sleep(0.02)  # 다음 반복을 위한 대기



def _calc_tile_cnt(origin_size, tile_size, margin_size):
    tile_cnt = (origin_size - margin_size) / (tile_size - margin_size)
    return int(tile_cnt)

def _make_tiled_images(tile_size: int, frame: np.array, margin_min=0.25) -> Tuple[List[np.array], int, int]:
    """
    이미지 또는 프레임을 타일 크기(tile_size)만큼 분할하여 타일 이미지들을 반환.
    
    Args:
        tile_size (int): 각 타일의 크기.
        frame (np.array): 타일로 분할할 원본 이미지 또는 프레임.
        margin_min (float): 타일 간의 최소 여백 비율 (기본값은 0.25).
    
    Returns:
        Tuple[List[np.array], int, int]:
            - List[np.array]: 자른 타일 이미지들의 리스트.
            - int: x축 방향 타일의 개수.
            - int: y축 방향 타일의 개수.
    """
    h, w = frame.shape[:2]
    margin_size = int(tile_size * margin_min)
    margin_minus_tile_size = tile_size - margin_size

    # 타일들을 저장할 리스트 초기화
    return_frames = []

    # 타일이 없을 경우 원본 프레임을 리턴
    if margin_minus_tile_size * 2 >= h and margin_minus_tile_size * 2 >= w:
        return [frame], 1, 1

    # x축과 y축 타일 개수 계산
    x_tile_cnt = _calc_tile_cnt(w, tile_size=tile_size, margin_size=margin_size)
    y_tile_cnt = _calc_tile_cnt(h, tile_size=tile_size, margin_size=margin_size)
    end_x_coord = tile_size
    start_x_coord = 0
    y_tile_start_end_coord = []

    # 타일을 자르는 루프
    for _x in range(x_tile_cnt):
        end_y_coord = tile_size
        start_y_coord = 0
        for _y in range(y_tile_cnt):
            if _x == 0:
                y_tile_start_end_coord.append([start_y_coord, end_y_coord])
            tile = frame[start_y_coord:end_y_coord, start_x_coord:end_x_coord]
            return_frames.append(tile)

            start_y_coord = end_y_coord - margin_size
            end_y_coord = end_y_coord - margin_size + tile_size
        
        # y축 마지막 타일
        tile = frame[-tile_size:, start_x_coord:end_x_coord]
        return_frames.append(tile)

        start_x_coord = end_x_coord - margin_size
        end_x_coord = end_x_coord - margin_size + tile_size

    # x축 마지막 타일
    for start_y_coord, end_y_coord in y_tile_start_end_coord:
        tile = frame[start_y_coord:end_y_coord, -tile_size:]
        return_frames.append(tile)

    # 끝부분 타일
    tile = frame[-tile_size:, -tile_size:]
    return_frames.append(tile)

    # 타일 개수 보정
    x_tile_cnt += 1
    y_tile_cnt += 1

    # 원본 프레임도 타일 리스트에 추가
    return_frames.append(frame)

    # 타일 리스트와 x, y축 타일 개수 반환
    return return_frames, x_tile_cnt, y_tile_cnt

from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode
from PIL import Image


def process_video( video: np.ndarray, size: int) -> torch.Tensor:
    """
    for each frame in video, to resize, crop, and normalize
    then combine them into one 4D tensor (#frame, 3, size, size)
    @param video: ndarray, (#frame, h, w, 3), rgb
    @param size: int, target pixel size
    """
    process = Compose([
        Resize(size, interpolation=InterpolationMode.BICUBIC),
        CenterCrop(size),
        lambda img: img.convert("RGB"),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

    video_tensor = torch.stack([process(Image.fromarray(v)) for v in video], dim=0)
    return video_tensor

def preprocess_input_image(model, origin_image, tile_size, margin_size):
    # tiling 
    tiles, wn, hn = _make_tiled_images(tile_size=tile_size, frame=origin_image, margin_min=margin_size)
    # resize 
    tiles = np.array(process_video(tiles, size=224))
    return tiles, wn, hn


def start_model_thread(model, thread_manager: ThreadManager) -> None:
    """
    모델 스레드를 시작하고 초기 상태를 설정합니다.

    Args:
        model (object): 사용할 모델 객체
        thread_manager (ThreadManager): 스레드와 상태 관리를 담당하는 ThreadManager 인스턴스

    Returns:
        None
    """
    # 초기 상태 설정
    thread_manager.model = model
    thread_manager.frame_queue = deque(maxlen=st.session_state.frame_len)  # session_state에 따라 최대 길이 설정
    thread_manager.frame_queue_updated = False
    thread_manager.thread_enabled = True

    # 모델 스레드 시작
    worker = threading.Thread(target=model_loop_thread, args=(thread_manager,))
    thread_manager.thread = worker
    worker.start()




def combine_image_tiles(tiles, wn, hn):
    tiles_view = np.transpose(tiles, (0, 2, 3, 1))
    h, w = tiles_view.shape[1:3]

    t_min = np.min(tiles_view)
    t_max = np.max(tiles_view)
    tiles_view = ((tiles_view - t_min) / (t_max - t_min) * 255).astype(np.uint8)
    tiles_comb = np.hstack([np.vstack([tiles_view[i * hn + j] for j in range(hn)]) for i in range(wn)])

    # make grid
    tiles_comb[[j for j in range(0, tiles_comb.shape[0], h)] + [-1], :] = [0, 255, 255]
    tiles_comb[:, [i for i in range(0, tiles_comb.shape[1], w)] + [-1]] = [0, 255, 255]
    return tiles_comb

import pandas as pd


def update_text_output(category_num, prompt, sim_score):
    # get component
    category_com, prompt_com, sim_score_com = st.session_state.final_text_output[category_num]

    # get category list
    category_list = st.session_state.prompt_category_list
    text_dict = st.session_state.prompt_text_dict

    # get prompt index
    prompt_idx = text_dict[category_num].index(prompt)
    GAUGE_MIN = st.session_state[f'category_{category_num}_prompt{prompt_idx}_min']
    GAUGE_MAX = st.session_state[f'category_{category_num}_prompt{prompt_idx}_max']

    # scaling sim_score  todo : scaling 기능 함수화
    sim_score_scaled = (sim_score - GAUGE_MIN) / (GAUGE_MAX - GAUGE_MIN)

    # update category_com, prompt_com
    prompt_com.write(prompt if sim_score_scaled > 0 else '')
    sim_score_com.progress(min(max(int(sim_score_scaled * 100), 0), 100))

    # Alert logic
    alert_start_time_key = f'alert_start_time_{category_num}'
    event_flag = st.session_state[f'event_flag_{category_num}']
    last_log_index_key = f'last_log_index_{category_num}'

    # Initializing session state
    if f'last_log_index_{category_num}' not in st.session_state:
        st.session_state[f'last_log_index_{category_num}'] = None

    if sim_score_scaled > 0.7:
        category_com.error(category_list[category_num], icon="🚨")
        current_time = datetime.datetime.now()
        if st.session_state[alert_start_time_key] is None:
            st.session_state[alert_start_time_key] = current_time
        else:
            duration = (current_time - st.session_state[alert_start_time_key]).total_seconds()
            if duration >= 3:
                log_number = st.session_state.log_number
                # Log the event if not already logged
                if not event_flag:
                    st.session_state.logs[log_number] = {
                        'category_num': category_list[category_num],
                        'prompt': prompt,
                        'start_time': st.session_state[alert_start_time_key].strftime('%Y-%m-%d %H:%M:%S'),
                        'duration': f'{duration:.1f}',
                    }
                    st.session_state[last_log_index_key] = log_number
                    st.session_state.log_number += 1
                    st.session_state[f'event_flag_{category_num}'] = True
                else:
                    # update the duration of the ongoing event
                    last_log_index = st.session_state[last_log_index_key]
                    st.session_state.logs[last_log_index]['duration'] = f'{duration:.1f}'

                # Make table
                logs = pd.DataFrame(st.session_state.logs).transpose()
                st.session_state.log_component.table(logs)

    elif sim_score_scaled >= 0.5:
        st.session_state[alert_start_time_key] = None  # Reset start time for new alerts
        st.session_state[f'event_flag_{category_num}'] = False  # Reset event logged flag
        category_com.warning(category_list[category_num], icon="⚠️")
    else:
        st.session_state[alert_start_time_key] = None  # Reset start time for new alerts
        st.session_state[f'event_flag_{category_num}'] = False  # Reset event logged flag
        category_com.info(category_list[category_num])

def make_text_output(sim_scores):
    # check sim_scores
    print(f'시발 여기냐?!')

    if sim_scores is None:
        print(f'sim_score is None')
        return
    else:
        prompt_all_text_list = st.session_state.prompt_all_text_list
        prompt_text_dict = st.session_state.prompt_text_dict

        # prompt and gauge 생성
        prompt_gauge_dict = {}

        print(f"Length of sim_scores: {len(sim_scores)}")
        print(f"Length of prompt_all_text_list: {len(st.session_state.prompt_all_text_list)}")



        
        for i, prompt in enumerate(prompt_all_text_list):
            prompt_gauge_dict[prompt] = sim_scores[i]

########################################
        if prompt in prompt_gauge_dict:
            sim_score = prompt_gauge_dict[prompt]
        else:
            print(f"Prompt '{prompt}' not found in prompt_gauge_dict")
########################################
        # 최고값 갱신
        for category_num, prompts in prompt_text_dict.items():
            max_similarity_prompt = None
            max_sim_score = 0

            for prompt in prompts:
                sim_score = prompt_gauge_dict[prompt]


############################
                # 배열일 경우 최대값 또는 평균값 선택 (최대값을 사용 예시)
                if isinstance(sim_score, np.ndarray):
                    sim_score = sim_score.max()  # 또는 sim_score.mean()으로 평균값 사용 가
############################
                if sim_score is None:
                    print(f"sim_score is None for prompt: {prompt}")
                elif not isinstance(sim_score, (int, float)):
                    print(f"sim_score is not a number: {sim_score} for prompt: {prompt}")
##############################

                if sim_score > max_sim_score:
                    max_similarity_prompt = prompt
                    max_sim_score = sim_score

            update_text_output(category_num, max_similarity_prompt, max_sim_score)
            # print(f'Category: {category_num}, prompt: {max_similarity_prompt}, score : {max_sim_score}')


def update_graph(scores, max_length=100):
    if scores is not None:
        # 초기화
        prompt_score_dict = st.session_state.prompt_score_dict
        print("시바 여기냐?", prompt_score_dict)
        prompts = st.session_state.prompt_all_text_list

        for i, prompt in enumerate(prompts):
            print(f"scores[{i}]: {scores[i]}, type: {type(scores[i])}")
            print(f"scores[{i}] shape: {scores[i].shape}")

            # 배열의 평균값을 계산합니다.
            value = np.mean(scores[i])

            if prompt in prompt_score_dict:
                # 리스트 길이가 max_length를 초과하면 첫 번째 요소를 제거합니다.
                if len(prompt_score_dict[prompt]) >= max_length:
                    prompt_score_dict[prompt].pop(0)
                prompt_score_dict[prompt].append(value)
            else:
                prompt_score_dict[prompt] = [value]

        # 데이터 업데이트
        st.session_state.prompt_score_dict = prompt_score_dict

    # 그래프 업데이트
    graph_component = st.session_state.get('graph_component', None)

    if graph_component is not None:
        fig = px.line(pd.DataFrame(st.session_state.prompt_score_dict), title='Similarity Graph')
        graph_component.plotly_chart(fig, use_container_width=True)


def video_processing_thread(
    video_capture: cv2.VideoCapture,
    thread_var: ThreadManager,
    fps: float,
    frame_int: float,
    tile_size: int,
    tile_margin: float,
    frame_len: int,
    MAX_WIDTH: int,
    MAX_HEIGHT: int,
    delay_state: bool
) -> None:
    """
    스레드에서 실행되는 비디오 스트림 처리 함수. 각 프레임을 읽고 전처리한 후 타일을 생성하며,
    프레임 큐를 업데이트하고, 그래프를 업데이트하는 작업을 수행합니다.
    
    Args:
        video_capture (cv2.VideoCapture): 비디오 캡처 객체.
        thread_var (ThreadManager): 스레드 및 상태 관리를 담당하는 ThreadManager 객체.
        fps (float): 비디오의 초당 프레임 수.
        frame_int (float): 프레임 간 간격.
        tile_size (int): 타일의 크기.
        tile_margin (float): 타일 간의 여백 크기.
        frame_len (int): 프레임 큐의 최대 길이.
        MAX_WIDTH (int): 출력 프레임의 최대 너비.
        MAX_HEIGHT (int): 출력 프레임의 최대 높이.
        delay_state (bool): 파일을 읽을 때 딜레이를 설정할지 여부.

    Returns:
        None
    """
    counter = 0  # 프레임 카운터 초기화

    try:
        while True:
            # 비디오에서 프레임을 읽음
            res, frame = video_capture.read()
            if not res:
                print("비디오 끝남")
                break

            # 프레임을 RGB로 변환
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 타일 결합 이미지 초기화
            tiles_comb = None

            # 지정된 프레임 간격에 맞춰 타일을 생성
            if counter % int(fps * frame_int) == 0:
                tiles, wn, hn = preprocess_input_image(thread_var.model, origin_image=frame, tile_size=tile_size, margin_size=tile_margin)

                if wn > 1 or hn > 1:
                    tiles_comb = combine_image_tiles(tiles, wn, hn)
                else:
                    tiles_comb = frame

                st.session_state.tiled_images = tiles

                # 스레드 락을 사용하여 프레임 큐를 업데이트
                print("타일이 어떤 형태냐!" , tiles[0].shape ,"--", len(tiles))
                with thread_var.thread_lock:
                    if len(thread_var.frame_queue) != frame_len:
                        mask = np.zeros_like(tiles)
                        for _ in range(frame_len):
                            thread_var.frame_queue.append(mask)
                    thread_var.frame_queue.append(tiles)
                    thread_var.frame_queue_updated = True

                # 모델 스레드에 프레임 큐가 업데이트되었음을 알림
                thread_var.frame_queue_updated_event.set()

                # 모델 처리 완료될 때까지 대기
                thread_var.model_processing_done_event.wait()
                thread_var.model_processing_done_event.clear()
                st.session_state.tiled_frame_com.write(f"{counter}")
                update_graph(thread_var.out_sim_scores)

            # 프레임 크기 조정 후 UI 업데이트
            source_frame = st.session_state.video_output_frame
            frame = cv2.resize(frame, (MAX_WIDTH, MAX_HEIGHT), interpolation=cv2.INTER_LINEAR)
            source_frame.image(frame, use_column_width=True)

            # 딜레이 설정이 되어 있을 경우 시간 대기
            if delay_state:
                # time.sleep(1.0 / fps - 0.01)
                time.sleep(0.05)

            # 프레임 카운터 증가
            counter += 1

    except Exception as e:
        print(f'error in : {e}')

    finally:
        # 비디오 캡처 객체 해제
        video_capture.release()

# Execute model
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

    # st.markdown(
    #     """
    #     <style>
    #     button {
    #         height: auto;
    #         padding-top: 20px !important;
    #         padding-bottom: 20px !important;
    #         width: 100% !important; /* 버튼을 가로로 길게 늘리기 */
    #     }
    #     </style>
    #     """,
    #     unsafe_allow_html=True,
    # )
    start_btn = st.button(f"# Start", key="st_btn_run", type="primary", disabled=st.session_state.thread_enabled)
    
    if start_btn:
        # Initializing the session state
        st.session_state.prompt_score_dict = {}
        # Start process
        thread_var = ThreadManager()
        start_process(thread_var, model_type, model_weight_path)

