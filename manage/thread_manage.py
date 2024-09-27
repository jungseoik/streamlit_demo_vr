# state_manager.py
import numpy as np
import threading
from collections import deque

class ThreadManager:
    def __init__(self):
        self.thread_enabled = False
        self.frame_queue: deque = deque()
        self.frame_queue_updated = False
        self.text_vectors: np.ndarray = None
        self.out_sim_scores = None
        self.out_sim_softmax = None
        self.thread_lock = threading.Lock()
        self.thread: threading.Thread = None
        self.model = None

        # 이벤트 플래그를 클래스 속성으로 추가
        self.frame_queue_updated_event = threading.Event()  # 프레임 큐 업데이트 이벤트
        self.model_processing_done_event = threading.Event()  # 모델 처리 완료 이벤트
        
    def reset(self):
        """상태를 초기화"""
        self.thread_enabled = False
        self.frame_queue.clear()
        self.frame_queue_updated = False
        self.text_vectors = None
        self.out_sim_scores = None
        self.out_sim_softmax = None
        self.thread = None
        self.model = None
