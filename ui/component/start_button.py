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
    txt_vectors = thread_manager.text_vectors  # 텍스트 벡터를 ThreadManager로부터 가져옴

    vid_vector = thread_manager.model(video = frames)
    sim_scores = thread_manager.model.model._loose_similarity(sequence_output=txt_vectors, visual_output=vid_vector)
    sim_scores = sim_scores.max(axis=1)
    sim_softmax = softmax(sim_scores)

    return sim_scores, sim_softmax

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

    while thread_manager.thread_enabled:
        if thread_manager.frame_queue_updated:
            thread_manager.frame_queue_updated = False
        else:
            time.sleep(0.02)  # 프레임이 업데이트될 때까지 대기
            continue

        with thread_manager.thread_lock:
            # 큐에서 프레임 가져오기
            frames = torch.tensor(np.array(thread_manager.frame_queue))

        # 텐서 차원 순서 변경 (batch, channel, height, width, frames)
        frames = frames.permute((1, 0, 2, 3, 4)).contiguous()

        # 모델 실행
        sim_scores, sim_softmax = run_model(thread_manager.model, frames, thread_manager)

        # 각 프롬프트와 해당 유사도 점수 로그 기록
        sim_scores_list = list(sim_scores)
        # demo_sim_logger.log_sim_scores(prompts_list, sim_scores_list)

        with thread_manager.thread_lock:
            # 유사도 점수와 소프트맥스 결과 업데이트
            thread_manager.out_sim_scores = sim_scores
            thread_manager.out_sim_softmax = sim_softmax

        time.sleep(0.02)  # 다음 반복을 위한 대기

def _calc_tile_cnt(origin_size, tile_size, margin_size):
    tile_cnt = (origin_size - margin_size) / (tile_size - margin_size)
    return int(tile_cnt)

def _make_tiled_images(tile_size, frame: np.array, margin_min=0.25):
    h, w = frame.shape[:2]
    margin_size = int(tile_size * margin_min)
    margin_minus_tile_size = tile_size - margin_size

    # Initializing
    return_frames = []

    # Available test
    if margin_minus_tile_size * 2 >= h and margin_minus_tile_size * 2 >= w:
        return [frame], 1, 1

    # Count calculatable tile
    x_tile_cnt = _calc_tile_cnt(w, tile_size=tile_size, margin_size=margin_size)
    y_tile_cnt = _calc_tile_cnt(h, tile_size=tile_size, margin_size=margin_size)
    end_x_coord = tile_size
    start_x_coord = 0
    y_tile_start_end_coord = []
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
        tile = frame[-tile_size:, start_x_coord:end_x_coord]
        return_frames.append(tile)

        start_x_coord = end_x_coord - margin_size
        end_x_coord = end_x_coord - margin_size + tile_size

    for start_y_coord, end_y_coord in y_tile_start_end_coord:
        tile = frame[start_y_coord:end_y_coord, -tile_size:]
        return_frames.append(tile)

    tile = frame[-tile_size:, -tile_size:]
    return_frames.append(tile)

    x_tile_cnt += 1
    y_tile_cnt += 1

    # add original frame
    return_frames.append(frame)
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
    if sim_scores is None:
        print(f'sim_score is None')
        return
    else:
        prompt_all_text_list = st.session_state.prompt_all_text_list
        prompt_text_dict = st.session_state.prompt_text_dict

        # prompt and gauge 생성
        prompt_gauge_dict = {}

        for i, prompt in enumerate(prompt_all_text_list):
            prompt_gauge_dict[prompt] = sim_scores[i]

        # 최고값 갱신
        for category_num, prompts in prompt_text_dict.items():
            max_similarity_prompt = None
            max_sim_score = 0

            for prompt in prompts:
                sim_score = prompt_gauge_dict[prompt]

                if sim_score > max_sim_score:
                    max_similarity_prompt = prompt
                    max_sim_score = sim_score

            update_text_output(category_num, max_similarity_prompt, max_sim_score)
            # print(f'Category: {category_num}, prompt: {max_similarity_prompt}, score : {max_sim_score}')


def update_graph(scores, max_length=100):
    if scores is not None:
        # initializing
        prompt_score_dict = st.session_state.prompt_score_dict
        print("시바 여기냐?",prompt_score_dict)
        prompts = st.session_state.prompt_all_text_list

        for i,prompt in enumerate(prompts):
            if prompt in prompt_score_dict:
                # Check if the list length is at max_length
                if len(prompt_score_dict[prompt]) >= max_length:
                    prompt_score_dict[prompt].pop(0)
                prompt_score_dict[prompt].append(scores[i].item())
            else:
                prompt_score_dict[prompt] = [scores[i].item()]

        # update data
        st.session_state.prompt_score_dict = prompt_score_dict

    # update graph
    graph_component = st.session_state.get('graph_component', None)

    if graph_component is not None:
        fig = px.line(pd.DataFrame(st.session_state.prompt_score_dict), title='Similarity Graph')
        graph_component.plotly_chart(fig,use_container_width=True)



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
    print("모델 뭐야 :", type(model))

    thread_var.model = model

    # encode all prompts and save
    texts = st.session_state.prompt_all_text_list
    if texts:
        thread_var.text_vectors = thread_var.model(text= texts)
        print(thread_var.text_vectors.shape)

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

    # start model thread
    start_model_thread(model, thread_var)

    # 비디오 저장을 위한 queue 생성
    video_record_queue = Queue(maxsize=100)

    # 비디오 저장 쓰레드 생성
    record_dir_path = st.session_state.save_path
    video_record_thread = threading.Thread(
        target=record_video,
        args=(video_record_queue, record_dir_path, fps),
        daemon=True
    )
    video_record_thread.start()

    # main loop
    try:
        while True:
            res, frame = video_capture.read()
            if not res:
                print("stream terminated. escaping...")
                break

            # put frame to queue if path is available
            if record_dir_path:
                video_record_queue.put(frame)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            tiles_comb = None
            if counter % int(fps * frame_int) == 0:
                tiles, wn, hn = preprocess_input_image(thread_var.model, origin_image=frame, tile_size=tile_size, margin_size=tile_margin)

                if wn > 1 or hn > 1:
                    tiles_comb = combine_image_tiles(tiles, wn, hn)
                else:
                    tiles_comb = frame

                # print("비디오 쉐입33333 : ", tiles.shape)
                # st.session_state.tiled_images = tiles


                # with thread_var.thread_lock:
                #     if len(thread_var.frame_queue) != frame_len:
                #         mask = np.zeros_like(tiles)
                #         for _ in range(frame_len):
                #             thread_var.add_frame(mask)
                #     thread_var.add_frame(tiles)

                try:
                    # print("비디오 쉐입33333 : ", tiles.shape)
                    st.session_state.tiled_images = tiles

                    with thread_var.thread_lock:
                        if len(thread_var.frame_queue) != frame_len:
                            mask = np.zeros_like(tiles)
                            for _ in range(frame_len):
                                thread_var.frame_queue.append(mask)
                        thread_var.frame_queue.append(tiles)
                        thread_var.frame_queue_updated = True

                    print("프레임 추가 완료")
                except Exception as e:
                    print(f"에러 발생: {e}")


            # update output ui - current frame
            source_frame = st.session_state.source_frame_com
            # print("비디오 쉐입22222 : ", source_frame.shpae)

            # Resize frame
            frame = cv2.resize(frame, (MAX_WIDTH, MAX_HEIGHT), interpolation=cv2.INTER_LINEAR)
            source_frame.image(frame, use_column_width=True)

            # tiled
            if tiles_comb is not None:
                tiled_frame = st.session_state.tiled_frame_com
                tiles_comb = cv2.resize(tiles_comb, (MAX_WIDTH, MAX_HEIGHT), interpolation=cv2.INTER_LINEAR)
                tiled_frame.image(tiles_comb, use_column_width=True)

            # 값에 변동 없으면 update 제한
            previous_score = st.session_state.previous_score
            # print(f"previous_score 타입: {type(previous_score)}")
            # print(f"thread_var.out_sim_scores 타입: {type(thread_var.out_sim_scores)}")

            if previous_score is None and thread_var.out_sim_scores is None:
            # if not np.array_equal(previous_score, thread_var.out_sim_scores):
            # if not np.array_equal(previous_score, thread_var.out_sim_scores):
                # make text output
                make_text_output(thread_var.out_sim_scores)
                # update graph
                update_graph(thread_var.out_sim_scores)

            st.session_state.previous_score = thread_var.out_sim_scores
            # print("비디오 쉐입2 : ", thread_var.out_sim_scores)

            if delay_state:     # 파일 읽어올 때는 딜레이 설정
                time.sleep(1.0 / fps - 0.01)

            counter += 1


    except Exception as e:
        print(f'error in : {e}')

    finally:
        video_capture.release()

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

