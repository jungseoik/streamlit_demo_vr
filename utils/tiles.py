from manage.thread_manage import ThreadManager
import streamlit as st
import numpy as np
from typing import Tuple
from typing import List
import torch

from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode
from PIL import Image


def process_video(video: np.ndarray, size: int) -> torch.Tensor:
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