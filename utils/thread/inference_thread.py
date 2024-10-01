
import streamlit as st
from manage.thread_manage import ThreadManager
from config import *
import torch
import numpy as np
import time
from collections import deque
import threading

from typing import Tuple

import torch.nn.functional as F


def inference_model(model: object, frames: torch.Tensor, thread_manager: ThreadManager) -> Tuple[np.ndarray, np.ndarray]:
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
    sim_scores_np = max_values.cpu().numpy()
    softmax_values_np = softmax_values.cpu().numpy()

    # NumPy 배열로 변환된 값 리턴
    return sim_scores_np, softmax_values_np

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

    # frames = frames.cuda() 
    
    txt_vectors = thread_manager.text_vectors  # 텍스트 벡터를 ThreadManager로부터 가져옴

    vid_vector = thread_manager.model(video = frames)

    sim_scores = thread_manager.model.model._loose_similarity(sequence_output=txt_vectors, visual_output=vid_vector)

    # 최대값 계산
    max_values, max_indices = sim_scores.max(dim=1)
    softmax_values = F.softmax(max_values, dim=0)

    # 텐서를 CPU로 이동 후 NumPy 배열로 변환
    sim_scores_np = max_values.cpu().numpy()
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
            frames = torch.tensor(np.array(thread_manager.frame_queue)).cuda()
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

        # time.sleep(0.02)  # 다음 반복을 위한 대기


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


