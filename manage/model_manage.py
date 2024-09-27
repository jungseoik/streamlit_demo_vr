import os
from pia.ai.tasks.T2VRet.base import T2VRetConfig
from pia.model import PiaTorchModel
import numpy as np
import torch
import streamlit as st
import os.path as osp
from config import *

from pathlib import Path
from pia.utils.load_utils import google_drive_get_model

def download_model():
    large_files_dir = Path("models")
    large_files_dir.mkdir(parents=True, exist_ok=True)

    # Clip4Clip 모델 다운로드
    model_filename = "clip4clip.pt"
    model_filepath = large_files_dir / model_filename
    if not model_filepath.exists():
        google_drive_get_model(
            save_dir=large_files_dir,
            save_file_name=model_filename,
            file_id="1SZI2mQJxE9yewv3WpRNlAkxMVI6OM0nc",  # 실제 Google Drive 모델 파일 ID 입력
        )


def load_model(model_name: str, 
               model_weight_path: str = None, 
               device: str = None):
    
    os.makedirs("assets", exist_ok=True)
    
    if model_name == "C4-fine-tuned":
        # 기본 설정으로 T2VRetConfig 초기화
        config = T2VRetConfig(model_path="models/clip4clip.pt", device="cuda")
        # config = T2VRetConfig(model_path=model_weight_path, device="cuda")

        model = PiaTorchModel(target_task="RET", target_model="clip4clip", config=config)
    else:
        raise ValueError(f"Model {model_name} not supported")

    return model

def prepare_model():
    model_type = st.session_state.model_type
    model_weight_path = st.session_state.model_weight_path

    if not osp.exists(model_weight_path):
        download_model()
    # use cpu for debugging
    model = load_model(model_name=model_type, model_weight_path=model_weight_path, device=DEVICE)
    return model