import streamlit as st
from manage.thread_manage import ThreadManager
import cv2
import time
import datetime
import plotly.express as px
import numpy as np
import pandas as pd

from utils.tiles import preprocess_input_image

def update_text_output(category_num, prompt, sim_score):
    # get component
    print("뜸?22222222222")
    st.session_state.final_text_output[category_num]
    category_com, prompt_com, sim_score_com = st.session_state.final_text_output[category_num]
    # get category list
    print("뜸?11111111111111")
  
    category_list = st.session_state.prompt_category_list
    text_dict = st.session_state.prompt_text_dict

    # get prompt index
    prompt_idx = text_dict[category_num].index(prompt)
    GAUGE_MIN = st.session_state[f'category_{category_num}_prompt{prompt_idx}_min']
    GAUGE_MAX = st.session_state[f'category_{category_num}_prompt{prompt_idx}_max']

    # scaling sim_score  todo : scaling 기능 함수화
    sim_score_scaled = (sim_score - GAUGE_MIN) / (GAUGE_MAX - GAUGE_MIN)
    print("뜸?")

    # update category_com, prompt_com

    prompt_com.write(prompt if sim_score_scaled > 0 else '')
    print("뜸?33333333333")

    sim_score_com.progress(min(max(int(sim_score_scaled * 100), 0), 100))
    print("뜸?44444444444")

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
                # logs = pd.DataFrame(st.session_state.logs).transpose()
                # st.session_state.log_component.table(logs)

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

        for i, prompt in enumerate(prompt_all_text_list):
            prompt_gauge_dict[prompt] = sim_scores[i]
            
        # 최고값 갱신
        for category_num, prompts in prompt_text_dict.items():
            max_similarity_prompt = None
            max_sim_score = 0
          
            try:
                for prompt in prompts:

                    if prompt not in prompt_gauge_dict:
                        print(f"Prompt '{prompt}' not in prompt_gauge_dict")
                        continue

                    sim_score = prompt_gauge_dict[prompt]

                    print(f"Prompt: {prompt}, Sim Score: {sim_score}")
                    # print(sim_score ,"?", max_sim_score)
                    if sim_score > max_sim_score:
                        max_similarity_prompt = prompt
                        max_sim_score = sim_score
            except Exception as e:
                print(f"Error {e}")
            print("뜸?22456456132")

            update_text_output(category_num, max_similarity_prompt, max_sim_score)
            # print(f'Category: {category_num}, prompt: {max_similarity_prompt}, score : {max_sim_score}')


def update_graph(scores, max_length=100):
    if scores is not None:
        # 초기화
        prompt_score_dict = st.session_state.prompt_score_dict
        prompts = st.session_state.prompt_all_text_list

        for i, prompt in enumerate(prompts):
            # print(f"scores[{i}]: {scores[i]}, type: {type(scores[i])}")
            # print(f"scores[{i}] shape: {scores[i].shape}")

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

                # if wn > 1 or hn > 1:
                #     tiles_comb = combine_image_tiles(tiles, wn, hn)
                # else:
                #     tiles_comb = frame

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
                st.session_state.frame_sampling_count.write(f"{counter}")
                print("제가 out sim 입니다 ", len(thread_var.out_sim_scores))
                print("제가 out sim 입니다 ", thread_var.out_sim_scores.shape)
                print("제가 out sim 입니다 ", thread_var.out_sim_scores[0])

                update_graph(thread_var.out_sim_scores)
                make_text_output(thread_var.out_sim_scores)
                

            # 프레임 크기 조정 후 UI 업데이트
            source_frame = st.session_state.video_output_frame
            frame = cv2.resize(frame, (MAX_WIDTH, MAX_HEIGHT), interpolation=cv2.INTER_LINEAR)
            source_frame.image(frame, use_column_width=True)
            time.sleep(0.1) 

            # 딜레이 설정이 되어 있을 경우 시간 대기
            # if delay_state:
            #     # time.sleep(1.0 / fps - 0.01)
            #     time.sleep(0.05) 

            # 프레임 카운터 증가
            counter += 1

    except Exception as e:
        print(f'error in : {e}')

    finally:
        # 비디오 캡처 객체 해제
        video_capture.release()