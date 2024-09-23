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

    def get_frame(self):
        """프레임 가져오기"""
        with self.thread_lock:
            if self.frame_queue:
                return self.frame_queue.popleft()
            return None

    def add_frame(self, frame):
        """프레임 추가하기"""
        with self.thread_lock:
            self.frame_queue.append(frame)
            self.frame_queue_updated = True
